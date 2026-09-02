import test from "node:test";
import assert from "node:assert/strict";
import { buildKiroResponsesPayload } from "../src/responses/request.ts";

const credentials = { providerSpecificData: { profileArn: "arn:aws:demo" } };
const model = "gpt-5.6-sol";
const req = (extra: Record<string, unknown> = {}) => ({ model, ...extra });

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

test("sanitizes function tools and rejects hosted/stateful features", () => {
  const p = buildKiroResponsesPayload(model, req({ tools: [{ type: "function", name: "x", parameters: { type: "object", additionalProperties: false } }]}), false, {});
  const tool = p.conversationState.currentMessage.userInputMessage.userInputMessageContext?.tools?.[0] as { toolSpecification: { inputSchema: { json: Record<string, unknown> } } };
  assert.deepEqual(tool.toolSpecification.inputSchema.json.required, []);
  assert.throws(() => buildKiroResponsesPayload(model, req({ background: true }), false, {}), /Unsupported Responses field/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ tools: [{ type: "web_search" }]}), false, {}), /Only function tools/);
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
  assert.equal(first.conversationState.conversationId, second.conversationState.conversationId);
  assert.deepEqual(first.additionalModelRequestFields, { reasoning: { effort: "low" } });
  assert.throws(
    () => buildKiroResponsesPayload(model, req({ store: true }), false, {}),
    /storage is not supported/
  );
  assert.throws(
    () => buildKiroResponsesPayload(model, req({ include: ["file_search_call.results"] }), false, {}),
    /include value/
  );
});
test("rejects unsupported models and malformed boundaries", () => {
  assert.throws(() => buildKiroResponsesPayload("gpt-5.6", req(), false, {}), /native-reasoning/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ reasoning: { effort: "x" }}), false, {}), /reasoning.effort/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ max_output_tokens: 0 }), false, {}), /positive integer/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ input: [{ type: "message", role: "user", content: [{ type: "input_image", image_url: "https://example.com/a.png" }]}]}), false, {}), /base64 data URL/);
  assert.throws(() => buildKiroResponsesPayload(model, req({ input: [{ type: "function_call_output", call_id: "c", output: 42 }]}), false, {}), /output must be/);
});
