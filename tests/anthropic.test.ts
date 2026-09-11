import { describe, expect, test } from "bun:test";
import { anthropicToOpenAI, collectAnthropicResponse, countAnthropicTokens, openAIStreamToAnthropic } from "../src/anthropic.ts";
import { buildKiroPayload } from "../src/kiro/request/openai-to-kiro.ts";

test("相同长系统提示和开场文本不会复用会话身份", () => {
  const body = anthropicToOpenAI({ system: "共同系统规则".repeat(1000), messages: [
    { role: "user", content: "你好" }, { role: "assistant", content: "请继续" }, { role: "user", content: "处理当前任务" },
  ] });
  const first = buildKiroPayload("claude-opus-5", body, false, {});
  const second = buildKiroPayload("claude-opus-5", body, false, {});
  expect(first.conversationState.conversationId).not.toBe(second.conversationState.conversationId);
  expect(first.conversationState.history).toEqual(second.conversationState.history);
});

test("长工具名在历史和本轮强制选择中匹配声明且保持结果关联", () => {
  const name = "tool_" + "long_name_".repeat(8);
  const payload = buildKiroPayload("claude-opus-5", anthropicToOpenAI({
    tools: [{ name, input_schema: { type: "object", properties: {} } }], tool_choice: { type: "tool", name },
    messages: [{ role: "user", content: "任务" }, { role: "assistant", content: [{ type: "tool_use", id: "a", name, input: {} }] }, { role: "user", content: [{ type: "tool_result", tool_use_id: "a", content: "完成" }] }],
  }), false, {});
  const context = payload.conversationState.currentMessage.userInputMessage.userInputMessageContext;
  const declared = context.tools[0].toolSpecification.name;
  expect(declared.length).toBeLessThanOrEqual(64);
  expect(context.toolChoice.tool.name).toBe(declared);
  const call = payload.conversationState.history[1].assistantResponseMessage.toolUses[0];
  expect(call.name).toBe(declared);
  expect(context.toolResults[0].toolUseId).toBe(call.toolUseId);
});

function upstream(lines: string[]): Response {
  return new Response(lines.map((x) => `data: ${x}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } });
}
function events(text: string) {
  return [...text.matchAll(/event: ([^\n]+)\ndata: ([^\n]+)/g)].map((m) => ({ event: m[1], data: JSON.parse(m[2]) }));
}

describe("系统提示传入 Kiro", () => {
  const first = "设备说明：向 xd://lsp 写入 JSON 调用代码导航。";
  const second = "行为约束：超出需求必须先询问用户。";
  for (const [name, system] of [
    ["字符串", `${first}\n${second}`],
    ["多文本块", [{ type: "text", text: first, cache_control: { type: "ephemeral" } }, { type: "text", text: second }]],
  ] as const) {
    test(`${name}系统提示在首轮及工具续轮保持完整顺序且仅出现一次`, () => {
      const messages = [{ role: "user", content: "用户任务" }];
      const toolHistory = [
        ...messages,
        { role: "assistant", content: [{ type: "tool_use", id: "call_1", name: "write", input: { path: "xd://lsp", content: "{}" } }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "call_1", content: "导航结果" }] },
      ];
      for (const turns of [messages, toolHistory]) {
        const body = { system, messages: turns, tools: [{ name: "write", description: "写入工具", input_schema: { type: "object", properties: {} } }] };
        const before = JSON.stringify(body);
        const payload = buildKiroPayload("claude-opus-5", anthropicToOpenAI(body), true, {});
        const text = JSON.stringify(payload);
        const wrapped = JSON.stringify(`<system-reminder>\n${first}\n${second}\n</system-reminder>`).slice(1, -1);
        expect(text).toContain(wrapped);
        expect(text.split(first)).toHaveLength(2);
        expect(text.split(second)).toHaveLength(2);
        expect(text.indexOf(first)).toBeLessThan(text.indexOf("用户任务"));
        expect(JSON.stringify(body)).toBe(before);
      }
    });
  }
});

test("低档遵循官方 effort 编码且不改写用户思考指令", () => {
  const payload = buildKiroPayload("claude-opus-5", anthropicToOpenAI({
    thinking: { type: "adaptive" }, output_config: { effort: "low" }, max_tokens: 128000,
    messages: [{ role: "user", content: "只回答结果" }],
  }), true, {});
  expect(payload.additionalModelRequestFields).toEqual({ output_config: { effort: "low" } });
  expect(payload.conversationState.currentMessage.userInputMessage.content).toBe("只回答结果");
});

test("显式关闭思考不被 effort 再次开启", () => {
  const payload = buildKiroPayload("claude-opus-5", anthropicToOpenAI({
    thinking: { type: "disabled" }, output_config: { effort: "low" },
    messages: [{ role: "user", content: "直接回答" }],
  }), true, {});
  expect(payload.additionalModelRequestFields).toEqual({ output_config: { effort: "low" }, thinking: { type: "disabled" } });
});

test("无可见摘要的分片签名在流式和非流式响应中完整保留", async () => {
  const lines = [
    JSON.stringify({ choices: [{ delta: { reasoning_signature: "签名甲" } }] }),
    JSON.stringify({ choices: [{ delta: { reasoning_signature: "签名乙" } }] }),
    JSON.stringify({ choices: [{ delta: { content: "结果" }, finish_reason: "stop" }] }),
  ];
  const streamed = events(await openAIStreamToAnthropic(upstream(lines)).text());
  expect(streamed.find(e => e.event === "content_block_start")?.data.content_block).toMatchObject({ type: "thinking", thinking: "" });
  expect(streamed.filter(e => e.data.delta?.type === "signature_delta").map(e => e.data.delta.signature).join("")).toBe("签名甲签名乙");
  const collected = await collectAnthropicResponse(upstream(lines)).then(r => r.json());
  expect(collected.content[0]).toEqual({ type: "thinking", thinking: "", signature: "签名甲签名乙" });
});

describe("Anthropic adapter", () => {
  test("emits valid ordered events for role-only and DONE", async () => {
    const out = await openAIStreamToAnthropic(upstream([
      JSON.stringify({ choices: [{ index: 0, delta: { role: "assistant" } }] }),
      "[DONE]",
    ])).text();
    expect(events(out).map((e) => e.event)).toEqual(["message_start", "message_delta", "message_stop"]);
  });

  test("DONE closes the response even when upstream stays open", async () => {
    const response = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            )
          );
        },
      })
    );

    const output = await openAIStreamToAnthropic(response).text();
    expect(events(output).at(-1)?.event).toBe("message_stop");
  });

  test("keeps contiguous blocks and fragmented thinking/text/tool JSON", async () => {
    const out = await openAIStreamToAnthropic(upstream([
      JSON.stringify({ choices: [{ delta: { reasoning_content: "think " } }] }),
      JSON.stringify({ choices: [{ delta: { reasoning_content: "more", content: "hello" } }] }),
      JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: "tu", function: { name: "lookup", arguments: '{"q":' } }] } }] }),
      JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"x"}' } }] }, finish_reason: "tool_calls" }], usage: { prompt_tokens: 4, completion_tokens: 3, cache_read_input_tokens: 0 } }),
      "[DONE]",
    ])).text();
    const es = events(out);
    expect(es.filter((e) => e.event === "content_block_start").map((e) => e.data.index)).toEqual([0, 1, 2]);
    const toolStart = es.find((e) => e.event === "content_block_start" && e.data.content_block.type === "tool_use");
    expect(toolStart?.data.content_block).toMatchObject({ type: "tool_use", id: "tu", name: "lookup", input: {} });
    expect(es.filter((e) => e.event === "content_block_start")).toHaveLength(3);
    const finalUsage = es.find((e) => e.event === "message_delta")!.data.usage;
    expect(finalUsage.cache_read_input_tokens).toBe(0);
    expect(finalUsage).not.toHaveProperty("cache_creation_input_tokens");
    const collected = await collectAnthropicResponse(upstream([
      JSON.stringify({ choices: [{ message: { role: "assistant", content: "hello", tool_calls: [{ index: 0, id: "tu", function: { name: "lookup", arguments: '{"q":"x"}' } }] }, finish_reason: "tool_calls" }], usage: { prompt_tokens: 4, completion_tokens: 3 } }),
    ])).then((r) => r.json()) as { content: unknown[] };
    const collectedTools = collected.content.filter((block): block is { type: "tool_use"; id: string; name: string; input: unknown } => {
      if (!block || typeof block !== "object") return false;
      const candidate = block as Record<string, unknown>;
      return candidate.type === "tool_use" && typeof candidate.id === "string" && typeof candidate.name === "string" && "input" in candidate;
    });
    expect(collectedTools).toHaveLength(1);
    expect(collectedTools[0]).toMatchObject({ type: "tool_use", id: "tu", name: "lookup", input: { q: "x" } });
  });
});

test("allocates fresh indexes for non-contiguous modalities", async () => {
  const out = await openAIStreamToAnthropic(upstream([
    JSON.stringify({ choices: [{ delta: { content: "a" } }] }),
    JSON.stringify({ choices: [{ delta: { reasoning_content: "b" } }] }),
    JSON.stringify({ choices: [{ delta: { content: "c" } }] }), "[DONE]",
  ])).text();
  expect(events(out).filter((e) => e.event === "content_block_start").map((e) => e.data.index)).toEqual([0, 1, 2]);
});

test("preserves thinking signature, including late arrival", async () => {
  const out = await openAIStreamToAnthropic(upstream([
    JSON.stringify({ choices: [{ delta: { reasoning_content: "secret" } }] }),
    JSON.stringify({ choices: [{ delta: { reasoning_signature: "opaque" } }] }),
    JSON.stringify({ choices: [{ delta: { content: "answer" } }] }), "[DONE]",
  ])).text();
  const es = events(out); const sig = es.find((e) => e.event === "content_block_delta" && e.data.delta.type === "signature_delta");
  expect(sig?.data.index).toBe(0); expect(sig?.data.delta.signature).toBe("opaque");
});

test("emits an error event for a truncated stream", async () => {
  const response = new Response("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n", { headers: { "content-type": "text/event-stream" } });
  const es = events(await openAIStreamToAnthropic(response).text());
  expect(es.at(-1)?.event).toBe("error"); expect(es).not.toContainEqual(expect.objectContaining({ event: "message_stop" }));
});

test("token estimator is positive, monotonic, and Unicode-aware", () => {
  const base = countAnthropicTokens({ messages: [{ role: "user", content: "hello" }] });
  const more = countAnthropicTokens({ messages: [{ role: "user", content: "hello 世界" }] });
  expect(base).toBeGreaterThan(0); expect(more).toBeGreaterThanOrEqual(base);
  expect(countAnthropicTokens({ messages: [], system: [{ type: "text", text: "policy" }], tools: [{ name: "x", input_schema: { type: "object" } }] })).toBeGreaterThan(0);
});

test("cleans up ping timer after terminal", async () => {
  const originalSet = globalThis.setInterval;
  const originalClear = globalThis.clearInterval;
  let cleared = false;
  const fakeSet = (_fn: TimerHandler, _ms?: number): number => 1;
  const fakeClear = (_timer?: number | string): void => { cleared = true; };
  globalThis.setInterval = fakeSet as unknown as typeof setInterval;
  globalThis.clearInterval = fakeClear as unknown as typeof clearInterval;
  try {
    await openAIStreamToAnthropic(upstream(["[DONE]"])).text();
    expect(cleared).toBe(true);
  } finally {
    globalThis.setInterval = originalSet;
    globalThis.clearInterval = originalClear;
  }
});

describe("Anthropic request validation", () => {
  test("rejects unsupported request semantics and content blocks", () => {
    const cases = [
      [{ output_config: { format: "json" } }, "output_config.format"],
      [{ output_config: { task_budget: 10 } }, "output_config.task_budget"],
      [{ speed: "fast" }, "speed"],
      [{ messages: [{ role: "user", content: [{ type: "document", source: {} }] }] }, "document"],
      [{ messages: [{ role: "user", content: [{ type: "image", source: { type: "url", url: "https://example.com/a.png" } }] }] }, "remote image"],
      [{ messages: [{ role: "user", content: [{ type: "tool_result", tool_use_id: "x", content: [{ type: "image", source: { type: "base64", data: "AA==" } }] }] }] }, "structured tool_result"],
      [{ messages: [{ role: "user", content: [{ type: "tool_result", content: ["raw"] }] }] }, "structured tool_result"],
      [{ messages: [{ role: "user", content: [{ type: "tool_result", content: [{ type: "text" }] }] }] }, "structured tool_result"],
      [{ messages: [{ role: "user", content: [{ type: "tool_result", content: [{ type: "text", text: 42 }] }] }] }, "structured tool_result"],
    ] as const;
    for (const [input, message] of cases) {
      expect(() => anthropicToOpenAI(input as unknown as Record<string, unknown>)).toThrow(message);
    }
  });
  test("continues accepting effort, text, base64 images, thinking replay, and image tool results", () => {
    expect(anthropicToOpenAI({
      context_management: { compact_2025_01_28: { trigger: "auto" } },
      messages: [
        { role: "assistant", content: [{ type: "thinking", thinking: "prior", signature: "sig" }] },
        { role: "user", content: [
          { type: "text", text: "hello" },
          { type: "tool_result", tool_use_id: "id", content: [
            { type: "text", text: "Image dimensions: 32x32." },
            { type: "image", source: { type: "base64", media_type: "image/png", data: "AA==" } },
          ] },
          { type: "image", source: { type: "base64", media_type: "image/png", data: "AA==" } },
        ] },
      ],
    }).messages).toBeDefined();
  });
});

