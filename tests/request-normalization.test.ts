import { expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { anthropicToOpenAI } from "../src/anthropic.ts";
import { createCatalog } from "../src/catalog.ts";
import { clearKiroModelCache, fetchKiroAvailableModels } from "../src/kiro/models.ts";
import { buildKiroResponsesPayload } from "../src/responses/request.ts";
import { buildKiroPayload, type KiroHistoryItem } from "../src/kiro/request/openai-to-kiro.ts";
import { collectResponses, responsesStream } from "../src/responses/response.ts";
import { createHandler } from "../src/server.ts";
import { kiroHistoryViolations, type KiroState } from "./kiro-history-rules.ts";

const model = "gpt-5.6-sol";
const inline = { type: "image", source: { type: "base64", media_type: "image/png", data: "AA==" } };
const omitted = "[image omitted: this model does not support image input]";
const remote = "[image omitted: remote image URLs are not supported, send base64 inline]";

function chat(messages: unknown[], extra: Record<string, unknown> = {}) {
  return buildKiroPayload("claude-opus-5", { messages, ...extra }, false, {}).conversationState;
}

test("consecutive assistant messages retain the text belonging to each signed reasoning slot", () => {
  const state = chat([
    { role: "assistant", content: [{ type: "thinking", thinking: "A", signature: "sigA" }] },
    { role: "assistant", content: [{ type: "text", text: "甲" }, { type: "thinking", thinking: "B", signature: "sigB" }, { type: "text", text: "乙" }] },
    { role: "user", content: "继续" },
  ]);
  expect(state.history.filter(item => item.assistantResponseMessage).map(item => ({ text: item.assistantResponseMessage?.content, reasoning: item.assistantResponseMessage?.reasoningContent }))).toEqual([
    { text: "甲", reasoning: { reasoningText: { text: "A", signature: "sigA" } } },
    { text: "乙", reasoning: { reasoningText: { text: "B", signature: "sigB" } } },
  ]);
  expect(kiroHistoryViolations(state as unknown as KiroState)).toEqual([]);
});

test("assistant messages interleaved in a tool-result batch retain signed and redacted reasoning", () => {
  const state = chat([
    { role: "assistant", tool_calls: ["a", "b"].map(id => ({ id, function: { name: "read", arguments: "{}" } })) },
    { role: "tool", tool_call_id: "a", content: "甲" },
    { role: "assistant", content: [{ type: "thinking", thinking: "checking", signature: "sig" }, { type: "text", text: "确认" }, { type: "redacted_thinking", data: "opaque" }] },
    { role: "tool", tool_call_id: "b", content: "乙" },
    { role: "user", content: "继续" },
  ]);
  expect(kiroHistoryViolations(state as unknown as KiroState)).toEqual([]);
  const assistants = state.history.flatMap(item => item.assistantResponseMessage ? [item.assistantResponseMessage] : []);
  expect(assistants.slice(1).map(item => [item.content, item.reasoningContent])).toEqual([
    ["确认", { reasoningText: { text: "checking", signature: "sig" } }],
    ["", { redactedContent: "opaque" }],
  ]);
  expect(state.history.flatMap(item => item.userInputMessage?.userInputMessageContext?.toolResults ?? []).map(result => result.toolUseId)).toEqual(["a", "b"]);
});

test("developer instructions use the system wrapper and schema literals survive sanitization", () => {
  const literal = { required: [], additionalProperties: false, nested: { $ref: "business data" } };
  const state = chat([{ role: "developer", content: "policy" }, { role: "user", content: "task" }], {
    tools: [{ type: "function", function: { name: "read", strict: true, parameters: { type: "object", properties: { value: { enum: [literal], default: literal, examples: [literal], additionalProperties: false } } } } }],
    parallel_tool_calls: false, tool_choice: { type: "any", disable_parallel_tool_use: true },
  });
  expect(state.currentMessage.userInputMessage.content).toBe("<system-reminder>\npolicy\n</system-reminder>\n\ntask");
  const schema = state.currentMessage.userInputMessage.userInputMessageContext?.tools?.[0].toolSpecification.inputSchema.json;
  expect(schema).toEqual({ type: "object", required: [], properties: { value: { enum: [literal], default: literal, examples: [literal] } } });
  expect(state.currentMessage.userInputMessage.userInputMessageContext?.toolChoice).toEqual({ any: {} });
  for (const choice of ["required", { type: "function", function: { name: "read" } }, { type: "function", name: "read" }]) {
    const context = chat([{ role: "user", content: "task" }], { tools: [{ function: { name: "read" } }], tool_choice: choice }).currentMessage.userInputMessage.userInputMessageContext;
    expect(context?.toolChoice).toEqual(choice === "required" ? { any: {} } : { tool: { name: "read" } });
  }
  expect(() => chat([], { tool_choice: "required" })).toThrow("requires declared tools");
});

test("both public adapters match result IDs, fill blank results and preserve true orphans as text", () => {
  const states = [
    chat([{ role: "assistant", tool_calls: [{ id: "a", function: { name: "read", arguments: "{}" } }] }, { role: "tool", tool_call_id: "a", content: "   " }, { role: "tool", tool_call_id: "ghost", content: "lost" }]),
    buildKiroResponsesPayload(model, { model, input: [{ type: "function_call", call_id: "a", name: "read", arguments: "{}" }, { type: "function_call_output", call_id: "a", output: "   " }, { type: "function_call_output", call_id: "ghost", output: "lost" }] }, false, {}).conversationState,
  ];
  for (const state of states) {
    expect(kiroHistoryViolations(state as unknown as KiroState)).toEqual([]);
    const user = state.currentMessage.userInputMessage;
    expect(user.content).toContain("[Tool Result (ghost)]\nlost");
    expect(user.userInputMessageContext?.toolResults).toEqual([{ toolUseId: "a", status: "success", content: [{ text: "(no output)" }] }]);
    expect(user.userInputMessageContext?.tools?.[0].toolSpecification.name).toBe("read");
    expect(state.history[0].userInputMessage?.content).toBe("...");
  }
});

test("synthesized Responses declarations restore custom and namespace output metadata in both encoders", async () => {
  const payload = buildKiroResponsesPayload(model, { model, input: [
    { type: "custom_tool_call", call_id: "a", namespace: "functions", name: "exec", input: "echo ok" },
    { type: "function_call", call_id: "b", namespace: "files", name: "read", arguments: "{}" },
    { type: "custom_tool_call_output", call_id: "a", output: "ok" },
    { type: "function_call_output", call_id: "b", output: "ok" },
  ] }, false, {});
  expect(payload._responseToolMetadata).toEqual({ functions__exec: { name: "exec", namespace: "functions", custom: true }, files__read: { name: "read", namespace: "files", custom: false } });
  expect(payload.conversationState.currentMessage.userInputMessage.userInputMessageContext?.tools?.[0].toolSpecification.inputSchema.json).toEqual({ type: "object", properties: { input: { type: "string" } }, required: ["input"] });
  const upstream = () => new Response(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [
    { index: 0, id: "new-a", function: { name: "functions__exec", arguments: '{"input":"echo ok"}' } },
    { index: 1, id: "new-b", function: { name: "files__read", arguments: '{}' } },
  ] }, finish_reason: "tool_use" }] })}\n\n`);
  const collected = await collectResponses(upstream(), model, payload._responseToolMetadata);
  expect(collected.output).toMatchObject([
    { type: "custom_tool_call", name: "exec", namespace: "functions", input: "echo ok" },
    { type: "function_call", name: "read", namespace: "files", arguments: "{}" },
  ]);
  const streamed = await responsesStream(upstream(), model, payload._responseToolMetadata).text();
  const events = streamed.split("\n").filter(line => line.startsWith("data: {")).map(line => JSON.parse(line.slice(6)));
  expect(events.find(event => event.type === "response.completed").response.output).toMatchObject(collected.output.map(({ id: _id, ...item }) => item));
});

test("all image forms and tool observations use exact positional omission text", () => {
  const imageShapes = [inline, { type: "image", image: "data:image/png;base64,AA==" }, { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } }];
  for (const image of imageShapes) {
    const user = buildKiroPayload("glm-5", { messages: [{ role: "user", content: [{ type: "text", text: "before" }, image, { type: "text", text: "after" }] }] }, false, {}).conversationState.currentMessage.userInputMessage;
    expect(user.content).toBe(`before\n${omitted}\nafter`);
    expect(user.images).toBeUndefined();
  }
  for (const image of [{ type: "image_url", image_url: "https://example.com/a.png" }, { type: "image", source: { type: "url", url: "https://example.com/a.png" } }]) {
    const user = chat([{ role: "user", content: [{ type: "text", text: "before" }, image, { type: "text", text: "after" }] }]).currentMessage.userInputMessage;
    expect(user.content).toBe(`before\n${remote}\nafter`);
    expect(user.images).toBeUndefined();
  }
  const anthropic = anthropicToOpenAI({ messages: [{ role: "assistant", content: [{ type: "tool_use", id: "a", name: "read", input: {} }] }, { role: "user", content: [{ type: "tool_result", tool_use_id: "a", content: [{ type: "text", text: "before" }, inline, { type: "image", source: { type: "url", url: "https://example.com/a.png" } }, { type: "text", text: "after" }] }] }] });
  const toolUser = buildKiroPayload("glm-5", anthropic, false, {}).conversationState.currentMessage.userInputMessage;
  expect(toolUser.userInputMessageContext?.toolResults?.[0].content[0].text).toBe(`before\n${omitted}\n${remote}\nafter`);
  expect(toolUser.images).toBeUndefined();
  for (const supported of [true, false]) {
    const user = buildKiroResponsesPayload(model, { model, input: [{ role: "user", content: [{ type: "input_text", text: "before" }, { type: "input_image", image_url: supported ? "https://example.com/a.png" : "data:image/png;base64,AA==" }, { type: "input_text", text: "after" }] }] }, false, {}, supported).conversationState.currentMessage.userInputMessage;
    expect(user.content).toBe(`before${supported ? remote : omitted}after`);
    expect(user.images).toBeUndefined();
  }
});

test("catalog modalities reach all server translators and every accepted image shape is size-gated", async () => {
  clearKiroModelCache();
  const catalog = createCatalog({ models: {}, loadCredentials: async () => ({ accessToken: "local" }), fetchModels: options => fetchKiroAvailableModels({ ...options, fetchImpl: async () => Response.json({ models: [
    { modelId: "claude-haiku-5.5", supportedInputTypes: ["TEXT"], tokenLimits: { maxInputTokens: 1000 } },
    { modelId: model, supportedInputTypes: ["TEXT"], tokenLimits: { maxInputTokens: 1000 } },
    { modelId: "glm-5", supportedInputTypes: ["TEXT", "IMAGE"], tokenLimits: { maxInputTokens: 1000 } },
    { modelId: "new-model", tokenLimits: { maxInputTokens: 1000 } },
  ] }) }) });
  await catalog.refresh();
  expect(catalog.supportsImagesFor("new-model")).toBe(true);
  expect(catalog.supportsImagesFor("glm-5")).toBe(true);
  const directory = await mkdtemp(join(tmpdir(), "kiro-request-boundary-"));
  const credentials = join(directory, "credentials.json");
  await writeFile(credentials, JSON.stringify({ accessToken: "local" }));
  const seen: unknown[] = [];
  try {
    const handler = createHandler({ listen: "127.0.0.1:0", models: {}, accounts: [{ credentialFile: credentials }] }, {
      execute: async input => { seen.push(input.body); return { response: Response.json({ choices: [{ message: { content: "ok" }, finish_reason: "end_turn" }] }) }; },
      refreshCredentials: async () => { throw new Error("refresh should not be called"); },
    }, { catalog });
    const listing = await (await handler(new Request("http://relay/v1/models"))).json();
    expect(listing.data.map((entry: { input_modalities: string[] }) => entry.input_modalities)).toEqual([["text"], ["text"], ["text", "image"], ["text", "image"]]);
    const send = (path: string, body: unknown) => handler(new Request(`http://relay${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }));
    for (const [path, body] of [
      ["/v1/messages", { model: "claude-haiku-5.5", messages: [{ role: "user", content: [inline] }] }],
      ["/v1/chat/completions", { model: "claude-haiku-5.5", messages: [{ role: "user", content: [inline] }] }],
      ["/v1/responses", { model, input: [{ role: "user", content: [{ type: "input_image", image_url: "data:image/png;base64,AA==" }] }] }],
    ] as const) expect((await send(path, body)).status).toBe(200);
    expect(seen).toHaveLength(3);
    for (const payload of seen) {
      // The server constructed this request through the typed payload builder above.
      const conversation = payload as { conversationState: { currentMessage: { userInputMessage: NonNullable<KiroHistoryItem["userInputMessage"]> } } };
      const user = conversation.conversationState.currentMessage.userInputMessage;
      expect(user.content).toBe(omitted); expect(user.images).toBeUndefined();
    }
    const data = "A".repeat(5 * 1024 * 1024 + 4);
    const oversized = [{ type: "image", source: { type: "base64", media_type: "image/png", data } }, { type: "image", image: `data:image/png;base64,${data}` }, { type: "image_url", image_url: { url: `data:image/png;base64,${data}` } }];
    for (const image of oversized) for (const nested of [false, true]) {
      const body = { model: "claude-haiku-5.5", messages: [{ role: "user", content: nested ? [{ type: "tool_result", tool_use_id: "orphan", content: [image] }] : [image] }] };
      for (const path of ["/v1/chat/completions", "/v1/messages"]) {
        const response = await send(path, body);
        expect(response.status).toBe(400); expect(await response.text()).toContain("image exceeds 5 MB maximum");
      }
    }
    const response = await send("/v1/responses", { model, input: [{ type: "function_call_output", call_id: "a", output: [{ type: "input_image", image_url: `data:image/png;base64,${data}` }] }] });
    expect(response.status).toBe(400); expect(await response.text()).toContain("image exceeds 5 MB maximum");
    expect(seen).toHaveLength(3);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
