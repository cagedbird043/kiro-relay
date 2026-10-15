import test from "node:test";
import { expect } from "bun:test";
import assert from "node:assert/strict";
import { buildKiroResponsesPayload } from "../src/responses/request.ts";
import { encodeReasoning } from "../src/kiro/reasoning.ts";
import { collectResponses } from "../src/responses/response.ts";

const credentials = { providerSpecificData: { profileArn: "arn:aws:demo" } };
const model = "gpt-5.6-sol";
const req = (extra: Record<string, unknown> = {}) => ({ model, ...extra });

test("多个无签名摘要完整合并而独立签名各占一条连续助手消息", () => {
  const first = { type: "reasoning", summary: [{ type: "summary_text", text: "第一段" }] };
  const second = { type: "reasoning", summary: [{ type: "summary_text", text: "第二段" }] };
  const input = [{ role: "user", content: "任务" }, first, { role: "assistant", content: "中间正文" }, second, { role: "user", content: "继续" }];
  const merged = buildKiroResponsesPayload(model, req({ input }), false, credentials);
  assert.deepEqual(merged.conversationState.history[1].assistantResponseMessage?.reasoningContent, { reasoningText: { text: "第一段\n\n第二段" } });
  const signed = { ...first, encrypted_content: encodeReasoning(model, { reasoningText: { text: "第一段", signature: "签名甲" } }) };
  const hidden = { type: "reasoning", summary: [], encrypted_content: encodeReasoning(model, { redactedContent: "UkVEQUNURUQ=" }) };
  const single = buildKiroResponsesPayload(model, req({ input: [{ role: "user", content: "任务" }, signed, { role: "user", content: "继续" }] }), false, credentials);
  assert.deepEqual(single.conversationState.history[1].assistantResponseMessage?.reasoningContent, { reasoningText: { text: "第一段", signature: "签名甲" } });
  const assistantReasoning = (input: unknown[]) => buildKiroResponsesPayload(model, req({ input }), false, credentials)
    .conversationState.history.slice(1).map(item => item.assistantResponseMessage?.reasoningContent);
  assert.deepEqual(assistantReasoning([{ role: "user", content: "任务" }, signed, second, { role: "user", content: "继续" }]), [
    { reasoningText: { text: "第一段", signature: "签名甲" } }, { reasoningText: { text: "第二段" } },
  ]);
  assert.deepEqual(assistantReasoning([{ role: "user", content: "任务" }, signed, hidden, { role: "user", content: "继续" }]), [
    { reasoningText: { text: "第一段", signature: "签名甲" } }, { redactedContent: "UkVEQUNURUQ=" },
  ]);
  const separate = buildKiroResponsesPayload(model, req({ input: [{ role: "user", content: "任务" }, signed, { role: "user", content: "下一轮" }, hidden, { role: "user", content: "继续" }] }), false, credentials);
  assert.deepEqual(separate.conversationState.history.filter(item => item.assistantResponseMessage).map(item => item.assistantResponseMessage?.reasoningContent), [
    { reasoningText: { text: "第一段", signature: "签名甲" } }, { redactedContent: "UkVEQUNURUQ=" },
  ]);
});

test("工具调用之后的独立签名推理让工具调用移到最后一条连续助手消息", () => {
  const signed = (text: string, signature: string) => ({ type: "reasoning", summary: [], encrypted_content: encodeReasoning(model, { reasoningText: { text, signature } }) });
  const payload = buildKiroResponsesPayload(model, req({ input: [
    { role: "user", content: "任务" },
    signed("", "签名甲"),
    { type: "function_call", call_id: "a", name: "read", arguments: "{}" },
    signed("", "签名乙"),
    { type: "function_call_output", call_id: "a", output: "完成" },
  ] }), false, credentials);
  const assistants = payload.conversationState.history.filter(item => item.assistantResponseMessage).map(item => item.assistantResponseMessage);
  assert.equal(assistants.length, 2);
  assert.equal(assistants[0]?.toolUses, undefined);
  assert.deepEqual(assistants[0]?.reasoningContent, { reasoningText: { text: "", signature: "签名甲" } });
  assert.deepEqual(assistants[1]?.reasoningContent, { reasoningText: { text: "", signature: "签名乙" } });
  assert.deepEqual(assistants[1]?.toolUses?.map(use => use.toolUseId), ["a"]);
});

test("命名空间合成超长工具的声明选择和历史使用同一个上游名称", () => {
  const namespace = "repo_namespace_".repeat(3);
  const name = "read_source_file_for_analysis";
  const payload = buildKiroResponsesPayload(model, req({
    tools: [{ type: "namespace", name: namespace, tools: [{ type: "function", name, parameters: { type: "object", properties: {} } }] }],
    tool_choice: { type: "function", namespace, name },
    input: [{ role: "user", content: "任务" }, { type: "function_call", call_id: "a", namespace, name, arguments: "{}" }, { type: "function_call_output", call_id: "a", output: "完成" }],
  }), false, credentials);
  const context = payload.conversationState.currentMessage.userInputMessage.userInputMessageContext;
  const wire = JSON.parse(JSON.stringify(context));
  const declared = wire.tools[0].toolSpecification.name;
  assert.ok(declared.length <= 64);
  assert.equal(wire.toolChoice.tool.name, declared);
  assert.equal(payload.conversationState.history[1].assistantResponseMessage?.toolUses?.[0].name, declared);
  assert.equal(payload.conversationState.currentMessage.userInputMessage.userInputMessageContext?.toolResults?.[0].toolUseId, "a");
});

test("继承属性同名工具可声明和返回且真正重复仍被拒绝", async () => {
  const tools = ["constructor", "__proto__"].map(name => ({ type: "function", name, parameters: { type: "object", properties: {} } }));
  const payload = buildKiroResponsesPayload(model, req({ input: "调用工具", tools }), false, credentials);
  const response = await collectResponses(new Response(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: tools.map((tool, index) => ({ index, id: String(index), function: { name: tool.name, arguments: "{}" } })) }, finish_reason: "tool_calls" }] })}\n\n`), model, payload._responseToolMetadata);
  assert.deepEqual(response.output.filter(item => item.type === "function_call").map(item => item.name), ["constructor", "__proto__"]);
  for (const tool of tools) assert.throws(() => buildKiroResponsesPayload(model, req({ input: "调用工具", tools: [tool, tool] }), false, credentials));
});

test("builds native Kiro payload with instructions and output limit", () => {
  const p = buildKiroResponsesPayload(model, req({ input: "Hello", instructions: "Be brief", max_output_tokens: 123, stream: false }), false, credentials);
  assert.equal(p.profileArn, "arn:aws:demo");
  assert.equal(p.inferenceConfig?.maxTokens, 123);
  assert.equal(p.conversationState.currentMessage.userInputMessage.content, "<system-reminder>\nBe brief\n</system-reminder>\n\nHello");
  assert.equal(p.conversationState.currentMessage.userInputMessage.modelId, model);
  assert.match(p.conversationState.conversationId, /^[0-9a-f-]{36}$/);
});

test("encodes input_image data URL in userInputMessage.images", () => {
  const p = buildKiroResponsesPayload(model, req({ input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "what?" }, { type: "input_image", image_url: "data:image/png;base64,AAEC" }] }]}), false, {});
  assert.deepEqual(p.conversationState.currentMessage.userInputMessage.images, [{ format: "png", source: { bytes: "AAEC" } }]);
});

test("maps reasoning and function call output", () => {
  const p = buildKiroResponsesPayload(model, req({ reasoning: { effort: "high" }, input: [{ type: "message", role: "user", content: "run" }, { type: "function_call", call_id: "call_1", name: "lookup", arguments: '{"q":"x"}' }, { type: "function_call_output", call_id: "call_1", output: "result" }]}), false, {});
  assert.deepEqual(p.additionalModelRequestFields, { reasoning: { effort: "high" } });
  assert.deepEqual(p.conversationState.history[1], { assistantResponseMessage: { content: "", toolUses: [{ toolUseId: "call_1", name: "lookup", input: { q: "x" } }] } });
  assert.deepEqual(p.conversationState.currentMessage.userInputMessage.userInputMessageContext?.toolResults, [{ toolUseId: "call_1", status: "success", content: [{ text: "result" }] }]);
});

test("并行工具保持同轮且依赖工具结果的后续调用保持独立回合", () => {
  const payload = buildKiroResponsesPayload(model, req({ input: [
    { role: "user", content: "先读取两份配置再执行修改" },
    { type: "function_call", call_id: "a", name: "read", arguments: '{"path":"a"}' },
    { type: "function_call", call_id: "b", name: "read", arguments: '{"path":"b"}' },
    { type: "function_call_output", call_id: "b", output: "配置乙" },
    { type: "function_call_output", call_id: "a", output: "配置甲" },
    { role: "assistant", content: "根据两份配置修改" },
    { type: "custom_tool_call", call_id: "c", name: "edit", input: "修改指令" },
    { type: "custom_tool_call_output", call_id: "c", output: "修改成功" },
  ] }), false, {});
  assert.deepEqual(payload.conversationState.history, [
    { userInputMessage: { content: "先读取两份配置再执行修改", modelId: model, origin: "AI_EDITOR" } },
    { assistantResponseMessage: { content: "", toolUses: [
      { toolUseId: "a", name: "read", input: { path: "a" } },
      { toolUseId: "b", name: "read", input: { path: "b" } },
    ] } },
    { userInputMessage: { content: "", modelId: model, origin: "AI_EDITOR", userInputMessageContext: { toolResults: [
      { toolUseId: "b", status: "success", content: [{ text: "配置乙" }] },
      { toolUseId: "a", status: "success", content: [{ text: "配置甲" }] },
    ] } } },
    { assistantResponseMessage: { content: "根据两份配置修改", toolUses: [{ toolUseId: "c", name: "edit", input: { input: "修改指令" } }] } },
  ]);
  assert.deepEqual(payload.conversationState.currentMessage.userInputMessage.userInputMessageContext?.toolResults,
    [{ toolUseId: "c", status: "success", content: [{ text: "修改成功" }] }]);
});

test("sanitizes function tools and rejects hosted/stateful features", () => {
  const p = buildKiroResponsesPayload(model, req({ tools: [{ type: "function", name: "x", parameters: { type: "object", additionalProperties: false } }]}), false, {});
  const tool = p.conversationState.currentMessage.userInputMessage.userInputMessageContext?.tools?.[0] as { toolSpecification: { inputSchema: { json: Record<string, unknown> } } };
  assert.deepEqual(tool.toolSpecification.inputSchema.json.required, []);
  assert.throws(() => buildKiroResponsesPayload(model, req({ background: true }), false, {}), /Unsupported Responses field/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ tools: [{ type: "web_search" }]}), false, {}), /Unsupported tool type/);
});


test("accepts OMP stateless defaults and cache keys", () => {
  const first = buildKiroResponsesPayload(
    model,
    req({
      input: [{ role: "user", content: "hello" }],
      store: false,
      include: ["reasoning.encrypted_content"],
      prompt_cache_key: "session-cache-key",
      reasoning: { effort: "minimal", summary: "auto" },
    }),
    true,
    {}
  );
  const second = buildKiroResponsesPayload(
    model,
    req({ input: "different text", prompt_cache_key: "session-cache-key" }),
    false,
    {}
  );
  assert.notEqual(first.conversationState.conversationId, second.conversationState.conversationId);
  assert.deepEqual(first.additionalModelRequestFields, {
    reasoning: { effort: "low", summary: "auto" },
  });
  assert.throws(
    () => buildKiroResponsesPayload(model, req({ store: true }), false, {}),
    /storage is not supported/
  );
  assert.throws(
    () => buildKiroResponsesPayload(model, req({ include: ["file_search_call.results"] }), false, {}),
    /include value/
  );
});

test("accepts OMP reasoning history and merges assistant tool calls", () => {
  const payload = buildKiroResponsesPayload(
    "gpt-5.6-luna",
    {
      model: "gpt-5.6-luna",
      reasoning: { effort: "max", summary: "auto" },
      input: [
        { role: "user", content: [{ type: "input_text", text: "inspect this repository" }] },
        { type: "reasoning", summary: [{ type: "summary_text", text: "..." }] },
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "I will inspect it.", annotations: [] }],
        },
        {
          type: "function_call",
          call_id: "call_read",
          name: "read",
          arguments: '{"path":"."}',
        },
        {
          type: "function_call_output",
          call_id: "call_read",
          output: "package.json\nsrc/",
        },
      ],
    },
    true,
    {}
  );

  assert.deepEqual(payload.additionalModelRequestFields, {
    reasoning: { effort: "max", summary: "auto" },
  });
  assert.deepEqual(payload.conversationState.history, [
    {
      userInputMessage: {
        content: "inspect this repository",
        modelId: "gpt-5.6-luna",
        origin: "AI_EDITOR",
      },
    },
    {
      assistantResponseMessage: {
        content: "I will inspect it.",
        reasoningContent: { reasoningText: { text: "..." } },
        toolUses: [
          { toolUseId: "call_read", name: "read", input: { path: "." } },
        ],
      },
    },
  ]);
  assert.equal(payload.conversationState.currentMessage.userInputMessage.content, "");
  assert.deepEqual(
    payload.conversationState.currentMessage.userInputMessage.userInputMessageContext?.toolResults,
    [
      {
        toolUseId: "call_read",
        status: "success",
        content: [{ text: "package.json\nsrc/" }],
      },
    ]
  );
});

test("flattens Codex additional namespace and custom tools", () => {
  const payload = buildKiroResponsesPayload(
    model,
    req({
      input: [
        { role: "user", content: "inspect" },
        {
          type: "additional_tools",
          role: "developer",
          tools: [
            {
              type: "namespace",
              name: "functions",
              description: "Codex tools",
              tools: [
                {
                  type: "custom",
                  name: "exec",
                  description: "Execute JavaScript",
                  format: { type: "grammar", syntax: "lark", definition: "start: SOURCE" },
                },
              ],
            },
            {
              type: "function",
              name: "wait",
              description: "Wait for a cell",
              parameters: { type: "object", properties: { cell_id: { type: "string" } } },
            },
          ],
        },
      ],
    }),
    false,
    {}
  );
  const tools =
    payload.conversationState.currentMessage.userInputMessage.userInputMessageContext?.tools;
  expect(JSON.stringify(tools)).toContain('"name":"functions__exec"');
  expect(JSON.stringify(tools)).toContain('"name":"wait"');
  expect(JSON.stringify(tools)).toContain('"input":{"type":"string"}');
  expect(payload._responseToolMetadata).toEqual({
    functions__exec: { name: "exec", namespace: "functions", custom: true },
  });
});

test("accepts custom tool call outputs", () => {
  const payload = buildKiroResponsesPayload(
    model,
    req({
      input: [
        { type: "custom_tool_call", call_id: "call_exec", name: "exec", input: "text('ok')" },
        { type: "custom_tool_call_output", call_id: "call_exec", output: "ok" },
      ],
    }),
    false,
    {}
  );
  expect(
    payload.conversationState.currentMessage.userInputMessage.userInputMessageContext?.toolResults
  ).toEqual([
    { toolUseId: "call_exec", status: "success", content: [{ text: "ok" }] },
  ]);
});
test("rejects unsupported models and malformed boundaries", () => {
  assert.throws(() => buildKiroResponsesPayload("gpt-5.6", req(), false, {}), /native-reasoning/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ reasoning: { effort: "x" }}), false, {}), /reasoning.effort/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ max_output_tokens: 0 }), false, {}), /positive integer/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ input: [{ type: "message", role: "user", content: [{ type: "input_image", image_url: "https://example.com/a.png" }]}]}), false, {}), /base64 data URL/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ input: [{ type: "function_call_output", call_id: "c", output: 42 }]}), false, {}), /output must be/);
});

test("routes instructions, system, and developer messages in order", () => {
  const payload = buildKiroResponsesPayload(model, req({
    instructions: "top",
    input: [
      { type: "message", role: "system", content: "system" },
      { type: "message", role: "developer", content: [{ type: "input_text", text: "developer" }] },
      { type: "message", role: "user", content: "question" },
    ],
  }), false, {});
  assert.equal(payload.conversationState.currentMessage.userInputMessage.content,
    "<system-reminder>\ntop\n</system-reminder>\n\n<system-reminder>\nsystem\n</system-reminder>\n\n<system-reminder>\ndeveloper\n</system-reminder>\n\nquestion");
  assert.equal(payload.conversationState.history.length, 0);
});

test("maps namespaced and custom forced tool choices", () => {
  const tools = [{ type: "namespace", name: "functions", tools: [{ type: "custom", name: "exec" }] }];
  for (const choice of [{ type: "custom", name: "exec", namespace: "functions" }, { type: "function", name: "exec", namespace: "functions" }]) {
    const payload = buildKiroResponsesPayload(model, req({ tools, tool_choice: choice }), false, {});
    assert.deepEqual(payload.conversationState.currentMessage.userInputMessage.userInputMessageContext?.toolChoice, { tool: { name: "functions__exec" } });
  }
  assert.throws(() => buildKiroResponsesPayload(model, req({ tools, tool_choice: { type: "function", name: "missing" } }), false, {}), /undeclared tool/);
});

test("工具结果中的文本进 toolResult，图片随该批结果的用户回合原样送达", () => {
  const base = { type: "function_call_output", call_id: "call_1" };
  for (const output of ["ok", [{ type: "input_text", text: "o" }, { type: "input_text", text: "k" }]]) {
    const payload = buildKiroResponsesPayload(model, req({ input: [{ ...base, output }] }), false, {});
    assert.deepEqual(payload.conversationState.currentMessage.userInputMessage.userInputMessageContext?.toolResults?.[0], { toolUseId: "call_1", status: "success", content: [{ text: "ok" }] });
  }
  const withImages = buildKiroResponsesPayload(model, req({ input: [
    { role: "user", content: "看图" },
    { type: "function_call", call_id: "call_1", name: "read", arguments: '{"path":"a.webp"}' },
    { type: "function_call", call_id: "call_2", name: "read", arguments: '{"path":"b.png"}' },
    { ...base, output: [{ type: "input_text", text: "Read image a.webp" }, { type: "input_image", image_url: "data:image/webp;base64,UklGRg==", detail: "auto" }] },
    { type: "function_call_output", call_id: "call_2", output: [{ type: "input_image", image_url: "data:image/png;base64,iVBORw==" }] },
  ] }), false, {}).conversationState.currentMessage.userInputMessage;
  assert.deepEqual(withImages.userInputMessageContext?.toolResults, [
    { toolUseId: "call_1", status: "success", content: [{ text: "Read image a.webp\n[image: image/webp]" }] },
    { toolUseId: "call_2", status: "success", content: [{ text: "[image: image/png]" }] },
  ]);
  assert.deepEqual(withImages.images, [{ format: "webp", source: { bytes: "UklGRg==" } }, { format: "png", source: { bytes: "iVBORw==" } }]);
  assert.throws(() => buildKiroResponsesPayload(model, req({ input: [{ ...base, output: [{ type: "input_file", file_id: "f" }] }] }), false, {}), /input_text or input_image/);
});

test("carries include encrypted content as internal metadata", () => {
  assert.equal(buildKiroResponsesPayload(model, req({ include: ["reasoning.encrypted_content"] }), false, {})._includeEncryptedContent, true);
  assert.equal(buildKiroResponsesPayload(model, req(), false, {})._includeEncryptedContent, false);
});
