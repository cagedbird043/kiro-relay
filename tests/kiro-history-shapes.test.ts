import { describe, expect, test } from "bun:test";
import { anthropicToOpenAI } from "../src/anthropic.ts";
import { encodeReasoning } from "../src/kiro/reasoning.ts";
import { buildKiroPayload } from "../src/kiro/request/openai-to-kiro.ts";
import { buildKiroResponsesPayload } from "../src/responses/request.ts";
import {
  assistantText,
  blockSequences,
  kiroHistoryViolations,
  reasoningSequence,
  toolUseSequence,
  type BlockKind,
  type KiroState,
} from "./kiro-history-rules.ts";

const claude = "claude-opus-5";
const gpt = "gpt-5.6-sol";
const credentials = { providerSpecificData: { profileArn: "arn:aws:demo" } };

/** 把块排列展开成带唯一签名、正文和工具 ID 的具体块。 */
function concreteBlocks(kinds: BlockKind[]) {
  return kinds.map((kind, index) => {
    if (kind === "S") return { kind, signature: `签名${index}`, text: `思考${index}` };
    if (kind === "R") return { kind, data: `隐藏${index}` };
    if (kind === "X") return { kind, text: `正文${index}` };
    return { kind, id: `call_${index}` };
  });
}

function anthropicState(kinds: BlockKind[]): KiroState {
  const blocks = concreteBlocks(kinds);
  const content = blocks.map((block) => {
    if (block.kind === "S") return { type: "thinking", thinking: block.text, signature: block.signature };
    if (block.kind === "R") return { type: "redacted_thinking", data: block.data };
    if (block.kind === "X") return { type: "text", text: block.text };
    return { type: "tool_use", id: block.id, name: "read", input: { path: block.id } };
  });
  const toolIds = blocks.flatMap((block) => (block.kind === "U" ? [block.id!] : []));
  const messages: unknown[] = [{ role: "user", content: "任务" }, { role: "assistant", content }];
  messages.push(toolIds.length
    ? { role: "user", content: toolIds.map((id) => ({ type: "tool_result", tool_use_id: id, content: `结果 ${id}` })) }
    : { role: "user", content: "继续" });
  const body = anthropicToOpenAI({ tools: [{ name: "read", input_schema: { type: "object", properties: {} } }], messages });
  return buildKiroPayload(claude, body, true, {}).conversationState as unknown as KiroState;
}

function responsesState(kinds: BlockKind[]): KiroState {
  const blocks = concreteBlocks(kinds);
  const input: unknown[] = [{ role: "user", content: "任务" }];
  for (const block of blocks) {
    if (block.kind === "S") input.push({ type: "reasoning", summary: [], encrypted_content: encodeReasoning(gpt, { reasoningText: { text: block.text!, signature: block.signature! } }) });
    else if (block.kind === "R") input.push({ type: "reasoning", summary: [], encrypted_content: encodeReasoning(gpt, { redactedContent: block.data! }) });
    else if (block.kind === "X") input.push({ role: "assistant", content: block.text });
    else input.push({ type: "function_call", call_id: block.id, name: "read", arguments: "{}" });
  }
  const toolIds = blocks.flatMap((block) => (block.kind === "U" ? [block.id!] : []));
  if (toolIds.length) for (const id of toolIds) input.push({ type: "function_call_output", call_id: id, output: `结果 ${id}` });
  else input.push({ role: "user", content: "继续" });
  const body = { model: gpt, tools: [{ type: "function" as const, name: "read", strict: false, parameters: { type: "object", properties: {} } }], input };
  return buildKiroResponsesPayload(gpt, body, true, credentials).conversationState as unknown as KiroState;
}

/** 原始块里的签名推理、工具调用和正文，按出现顺序。 */
function expected(kinds: BlockKind[]) {
  const blocks = concreteBlocks(kinds);
  return {
    reasoning: blocks.flatMap((block): unknown[] => block.kind === "S"
      ? [{ reasoningText: { text: block.text, signature: block.signature } }]
      : block.kind === "R" ? [{ redactedContent: block.data }] : []),
    toolUses: blocks.flatMap((block) => (block.kind === "U" ? [block.id] : [])),
    texts: blocks.flatMap((block) => (block.kind === "X" ? [block.text!] : [])),
  };
}

describe("单个助手回合内思考、正文、工具调用的全部排列（长度 1~5）", () => {
  const entries: Array<[string, (kinds: BlockKind[]) => KiroState]> = [
    ["Anthropic", anthropicState],
    ["Responses", responsesState],
  ];
  for (const [entry, build] of entries) {
    test(`${entry} 入口：符合 Kiro 结构规则且推理、工具调用、正文按原序完整保留`, () => {
      const failures: string[] = [];
      let checked = 0;
      for (const kinds of blockSequences(5)) {
        const state = build(kinds);
        const want = expected(kinds);
        const label = kinds.join("");
        for (const violation of kiroHistoryViolations(state)) failures.push(`${label}: ${violation}`);
        if (JSON.stringify(reasoningSequence(state)) !== JSON.stringify(want.reasoning)) failures.push(`${label}: 推理顺序或内容变化`);
        if (JSON.stringify(toolUseSequence(state)) !== JSON.stringify(want.toolUses)) failures.push(`${label}: 工具调用顺序或内容变化`);
        const text = assistantText(state);
        let cursor = 0;
        for (const piece of want.texts) {
          const found = text.indexOf(piece, cursor);
          if (found < 0) { failures.push(`${label}: 正文 ${piece} 丢失或乱序`); break; }
          cursor = found + piece.length;
        }
        checked++;
      }
      expect(checked).toBe(1364);
      expect(failures.slice(0, 20)).toEqual([]);
    });
  }
});

describe("历史结构边界", () => {
  const tools = [{ name: "read", input_schema: { type: "object", properties: {} } }];
  const viaAnthropic = (messages: unknown[]) =>
    buildKiroPayload(claude, anthropicToOpenAI({ tools, messages }), true, {}).conversationState as unknown as KiroState;
  const viaChat = (messages: unknown[]) =>
    buildKiroPayload(claude, { tools: [{ type: "function", function: { name: "read", parameters: { type: "object", properties: {} } } }], messages }, true, {})
      .conversationState as unknown as KiroState;
  const toolUse = (id: string) => ({ type: "tool_use", id, name: "read", input: {} });
  const toolResult = (id: string, content: unknown = `结果 ${id}`) => ({ type: "tool_result", tool_use_id: id, content });

  test("以助手开头的对话补一个中性占位用户回合", () => {
    const state = viaAnthropic([{ role: "assistant", content: "我先说" }, { role: "user", content: "好" }]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(state.history[0].userInputMessage?.content).toBe("...");
  });

  test("只含空白的用户消息不会以空正文发给 Kiro", () => {
    const blank = [{ type: "text", text: " " }];
    for (const messages of [
      [{ role: "user", content: blank }, { role: "assistant", content: "一" }, { role: "user", content: "继续" }],
      [{ role: "user", content: "任务" }, { role: "assistant", content: "一" }, { role: "user", content: blank }, { role: "assistant", content: "二" }, { role: "user", content: "继续" }],
      [{ role: "user", content: "任务" }, { role: "assistant", content: "一" }, { role: "user", content: blank }],
    ]) expect(kiroHistoryViolations(viaAnthropic(messages))).toEqual([]);
    const responses = buildKiroResponsesPayload(gpt, { model: gpt, input: [
      { role: "user", content: " " }, { role: "assistant", content: "一" }, { role: "user", content: " " },
    ] }, true, credentials).conversationState as unknown as KiroState;
    expect(kiroHistoryViolations(responses)).toEqual([]);
  });

  test("以助手结尾的对话用中性占位作为当前回合", () => {
    const state = viaAnthropic([{ role: "user", content: "任务" }, { role: "assistant", content: "做完了" }]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(state.currentMessage.userInputMessage?.content).toBe("...");
  });

  test("连续多条用户消息合并为一个用户回合且内容按序保留", () => {
    const state = viaAnthropic([
      { role: "user", content: "第一句" },
      { role: "user", content: "第二句" },
      { role: "assistant", content: "收到" },
      { role: "user", content: [{ type: "text", text: "第三句" }] },
      { role: "user", content: "第四句" },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(state.history[0].userInputMessage?.content).toBe("第一句\n\n第二句");
    expect(state.currentMessage.userInputMessage?.content).toContain("第三句");
    expect(state.currentMessage.userInputMessage?.content).toContain("第四句");
  });

  test("连续多条助手纯文本合并为一个助手回合", () => {
    const state = viaAnthropic([
      { role: "user", content: "任务" },
      { role: "assistant", content: "甲" },
      { role: "assistant", content: "乙" },
      { role: "user", content: "继续" },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(state.history[1].assistantResponseMessage?.content).toBe("甲\n\n乙");
  });

  test("找不到对应工具调用的工具结果转成正文，不丢内容", () => {
    const state = viaAnthropic([
      { role: "user", content: "任务" },
      { role: "user", content: [toolResult("ghost", "孤儿结果")] },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(state.currentMessage.userInputMessage?.content).toContain("孤儿结果");
    expect(state.currentMessage.userInputMessage?.userInputMessageContext?.toolResults).toBeUndefined();
  });

  test("历史中间的孤儿工具结果同样转成正文", () => {
    const state = viaAnthropic([
      { role: "user", content: [{ type: "text", text: "任务" }, toolResult("ghost", "旧结果")] },
      { role: "assistant", content: "好" },
      { role: "user", content: "继续" },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(state.history[0].userInputMessage?.content).toContain("旧结果");
  });

  test("多个并行工具调用的结果乱序返回仍全部匹配", () => {
    const state = viaAnthropic([
      { role: "user", content: "任务" },
      { role: "assistant", content: [toolUse("a"), toolUse("b"), toolUse("c")] },
      { role: "user", content: [toolResult("c"), toolResult("a"), toolResult("b")] },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(state.currentMessage.userInputMessage?.userInputMessageContext?.toolResults?.map((r) => r.toolUseId)).toEqual(["c", "a", "b"]);
  });

  test("Chat 入口每个工具结果单独一条 tool 消息也合成一个回合", () => {
    const state = viaChat([
      { role: "user", content: "任务" },
      { role: "assistant", content: null, tool_calls: ["a", "b"].map((id) => ({ id, type: "function", function: { name: "read", arguments: "{}" } })) },
      { role: "tool", tool_call_id: "b", content: "乙" },
      { role: "tool", tool_call_id: "a", content: "甲" },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(state.currentMessage.userInputMessage?.userInputMessageContext?.toolResults?.map((r) => r.toolUseId)).toEqual(["b", "a"]);
  });

  test("工具结果之间夹着助手纯文本时不拆散这批结果", () => {
    const state = viaChat([
      { role: "user", content: "任务" },
      { role: "assistant", content: null, tool_calls: ["a", "b"].map((id) => ({ id, type: "function", function: { name: "read", arguments: "{}" } })) },
      { role: "tool", tool_call_id: "a", content: "甲" },
      { role: "assistant", content: "插一句" },
      { role: "tool", tool_call_id: "b", content: "乙" },
      { role: "user", content: "继续" },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(assistantText(state)).toContain("插一句");
  });

  test("空字符串和空数组工具结果不会变成空正文", () => {
    const state = viaAnthropic([
      { role: "user", content: "任务" },
      { role: "assistant", content: [toolUse("a"), toolUse("b")] },
      { role: "user", content: [toolResult("a", ""), toolResult("b", [])] },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
  });

  test("工具结果里的图片挂到所在用户回合，正文仍在工具结果中", () => {
    const state = viaAnthropic([
      { role: "user", content: "任务" },
      { role: "assistant", content: [toolUse("shot")] },
      { role: "user", content: [toolResult("shot", [
        { type: "text", text: "截图如下" },
        { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
      ])] },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    const current = state.currentMessage.userInputMessage as { images?: unknown[]; userInputMessageContext?: { toolResults?: Array<{ content?: Array<{ text?: string }> }> } };
    expect(current.images).toEqual([{ format: "png", source: { bytes: "iVBORw0KGgo=" } }]);
    expect(current.userInputMessageContext?.toolResults?.[0].content?.[0].text).toContain("截图如下");
  });

  test("空正文的助手消息和只有思考的助手消息不破坏交替", () => {
    const state = viaAnthropic([
      { role: "user", content: "任务" },
      { role: "assistant", content: "" },
      { role: "user", content: "再来" },
      { role: "assistant", content: [{ type: "thinking", thinking: "只想不说", signature: "签名甲" }] },
      { role: "user", content: "继续" },
    ]);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(reasoningSequence(state)).toEqual([{ reasoningText: { text: "只想不说", signature: "签名甲" } }]);
  });

  test("多轮工具循环每轮都形成完整的调用与结果配对", () => {
    const messages: unknown[] = [{ role: "user", content: "任务" }];
    for (let round = 0; round < 4; round++) {
      messages.push({ role: "assistant", content: [
        { type: "thinking", thinking: "", signature: `签名${round}a` },
        toolUse(`r${round}a`),
        toolUse(`r${round}b`),
        { type: "thinking", thinking: "", signature: `签名${round}b` },
      ] });
      messages.push({ role: "user", content: [toolResult(`r${round}b`), toolResult(`r${round}a`)] });
    }
    messages.push({ role: "user", content: "总结" });
    const state = viaAnthropic(messages);
    expect(kiroHistoryViolations(state)).toEqual([]);
    expect(toolUseSequence(state)).toEqual(["r0a", "r0b", "r1a", "r1b", "r2a", "r2b", "r3a", "r3b"]);
    expect(reasoningSequence(state)).toHaveLength(8);
  });

  test("Responses 入口：结果先于调用之外的消息到来时仍按调用配对", () => {
    const state = buildKiroResponsesPayload(gpt, {
      model: gpt,
      tools: [{ type: "function", name: "read", strict: false, parameters: { type: "object", properties: {} } }],
      input: [
        { role: "user", content: "任务" },
        { type: "function_call", call_id: "a", name: "read", arguments: "{}" },
        { type: "function_call", call_id: "b", name: "read", arguments: "{}" },
        { type: "function_call_output", call_id: "b", output: "乙" },
        { type: "function_call_output", call_id: "a", output: "甲" },
        { role: "user", content: "继续" },
      ],
    }, true, credentials).conversationState as unknown as KiroState;
    expect(kiroHistoryViolations(state)).toEqual([]);
  });
});
