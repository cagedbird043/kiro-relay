import test from "node:test";
import assert from "node:assert/strict";
import { anthropicToOpenAI } from "../../src/anthropic.ts";
import { buildKiroPayload, type KiroHistoryItem } from "../../src/kiro/request/openai-to-kiro.ts";
import { kiroHistoryViolations, type KiroState } from "../kiro-history-rules.ts";

const description = "Evaluate one cell and preserve kernel state.\n".repeat(500);
const schema = { type: "object", properties: { code: { type: "string" } }, required: ["code"] };
const tool = { name: "_eval", description, input_schema: schema };
const user = { role: "user", content: "Run the computation." };
const system = "Use the declared tools for computations.";

interface ConvertedConversation {
  history: KiroHistoryItem[];
  currentMessage: { userInputMessage: NonNullable<KiroHistoryItem["userInputMessage"]> };
}

function state(body: Record<string, unknown>): ConvertedConversation {
  return buildKiroPayload("claude-opus-5-5", body, true, null).conversationState;
}

function firstUser(value: ConvertedConversation) {
  return value.history.find(turn => turn.userInputMessage)?.userInputMessage
    ?? value.currentMessage.userInputMessage;
}

function assertValid(value: ConvertedConversation) {
  assert.deepEqual(kiroHistoryViolations(value as KiroState), []);
}

for (const protocol of ["Anthropic", "Chat"] as const) {
  test(`${protocol}: long eval documentation stays in the initial context across user and tool turns`, () => {
    const build = (messages: unknown[]) => protocol === "Anthropic"
      ? state(anthropicToOpenAI({ system, tools: [tool], messages }))
      : state({ messages: [{ role: "system", content: system }, ...messages], tools: [
        { type: "function", function: { name: tool.name, description, parameters: schema } },
      ] });
    const assistant = protocol === "Anthropic"
      ? { role: "assistant", content: [{ type: "tool_use", id: "call_eval", name: "_eval", input: { code: "6*7" } }] }
      : { role: "assistant", content: null, tool_calls: [{ id: "call_eval", type: "function", function: { name: "_eval", arguments: '{"code":"6*7"}' } }] };
    const result = protocol === "Anthropic"
      ? { role: "user", content: [{ type: "tool_result", tool_use_id: "call_eval", content: "42" }] }
      : { role: "tool", tool_call_id: "call_eval", content: "42" };
    const initial = build([user]);
    const continuation = build([user, assistant, result]);
    const followUp = build([user, assistant, result, { role: "assistant", content: "42" }, { role: "user", content: "Now compute 7*7." }]);
    const initialContext = firstUser(initial).content;
    assert.ok(initialContext.startsWith("<system-reminder>\n"));
    assert.ok(initialContext.includes(description));
    assert.ok(initialContext.includes(system));
    assert.ok(initialContext.endsWith(user.content));
    for (const value of [initial, continuation, followUp]) {
      assertValid(value);
      assert.equal(firstUser(value).content, initialContext, "the instruction carrier must not move or change when the conversation grows");
      const turns = [...value.history, value.currentMessage];
      assert.equal(turns.filter(turn => turn.userInputMessage?.content.includes(description)).length, 1);
      const declared = value.currentMessage.userInputMessage.userInputMessageContext?.tools?.[0]?.toolSpecification;
      assert.ok(declared && declared.description.length < 10000, "the schema must stay below the upstream description limit");
      assert.deepEqual(declared.inputSchema.json, { type: "object", properties: { code: { type: "string" } }, required: ["code"] });
    }
    const current = continuation.currentMessage.userInputMessage;
    assert.equal(current.content, "", "a tool continuation must not become a fresh documentation-only input");
    assert.deepEqual(current.userInputMessageContext?.toolResults, [
      { toolUseId: "call_eval", status: "success", content: [{ text: "42" }] },
    ]);
    assert.equal(followUp.currentMessage.userInputMessage.content, "Now compute 7*7.");
  });
}

test("description length boundary leaves accepted schemas inline and relocates only oversized descriptions", () => {
  for (const length of [10000, 10001]) {
    const body = {
      messages: [user, { role: "assistant", content: "ready" }, { role: "user", content: "continue" }],
      tools: [{ name: "boundary", description: "D".repeat(length), input_schema: schema }],
    };
    const value = state(body);
    assertValid(value);
    const spec = value.currentMessage.userInputMessage.userInputMessageContext?.tools?.[0]?.toolSpecification;
    if (length === 10000) {
      assert.equal(spec?.description.length, length);
      assert.equal(firstUser(value).content, user.content);
    } else {
      assert.ok(spec && spec.description.length < 10000);
      assert.ok(firstUser(value).content.includes("D".repeat(length)));
    }
    assert.equal(value.currentMessage.userInputMessage.content, "continue");
  }
});

test("relocated docs do not break assistant-first history, tool errors or image results", () => {
  const image = { type: "image", source: { type: "base64", media_type: "image/png", data: "aW1hZ2U=" } };
  const value = state(anthropicToOpenAI({ tools: [tool], messages: [
    { role: "assistant", content: [{ type: "tool_use", id: "call_image", name: "_eval", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "call_image", is_error: true, content: [{ type: "text", text: "Capture failed" }, image] }] },
  ] }));
  assertValid(value);
  assert.ok(firstUser(value).content.includes(description));
  assert.equal(value.currentMessage.userInputMessage.content, "");
  const result = value.currentMessage.userInputMessage.userInputMessageContext?.toolResults?.[0];
  assert.equal(result?.toolUseId, "call_image");
  assert.equal(result?.status, "error");
  assert.ok(result?.content[0].text.includes("Capture failed"));
  assert.deepEqual(value.currentMessage.userInputMessage.images, [{ format: "png", source: { bytes: "aW1hZ2U=" } }]);
  assert.equal(value.history.filter(turn => turn.assistantResponseMessage).length, 1, "document relocation must not invent an assistant acknowledgement");
});

test("tool-only requests retain complete docs without leaking internal fields", () => {
  const value = state({ messages: [], tools: [tool] });
  assertValid(value);
  assert.ok(value.currentMessage.userInputMessage.content.includes(description));
  assert.equal(value.history.length, 0);
  assert.ok(!JSON.stringify(value).includes('"_toolDocs"'), "Kiro rejects unknown transport fields");
});

test("mixed inventories relocate only long descriptions and preserve their original text", () => {
  const otherDescription = "Second tool capabilities.\n".repeat(500);
  const value = state({ messages: [user, { role: "assistant", content: "ready" }, { role: "user", content: "next" }], tools: [
    tool,
    { name: "short", description: "A short description", input_schema: schema },
    { name: "other", description: otherDescription, input_schema: schema },
  ] });
  assertValid(value);
  const content = firstUser(value).content;
  assert.ok(content.includes(description));
  assert.ok(content.includes(otherDescription));
  assert.ok(!content.includes("A short description"));
  const specs = value.currentMessage.userInputMessage.userInputMessageContext?.tools?.map(entry => entry.toolSpecification);
  assert.ok(specs?.every(spec => spec.description.length <= 10000));
  assert.equal(specs?.[1].description, "A short description");
  assert.equal(value.currentMessage.userInputMessage.content, "next");
});
