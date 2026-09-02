import { expect, test } from "bun:test";
import { readKiroEvents } from "../src/kiro/events.ts";

test("readKiroEvents handles fragmented SSE frames and DONE", async () => {
  const chunks = [
    'data: {"id":"r1","choices":[{"delta":{"role":"assistant","content":"hel',
    'lo"}}]}\n\n',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"lookup","arguments":"{\\"q\\":"}}]}}]}\n\n',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"x\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"cache_read_input_tokens":0}}\n\n',
    "data: [DONE]\n\n",
  ];
  const stream = new ReadableStream<Uint8Array>({
    start(controller) { for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk)); controller.close(); },
  });
  const events = [];
  for await (const event of readKiroEvents(new Response(stream), "gpt-5.6")) events.push(event);
  expect(events.map((event) => event.type)).toEqual(["start", "text_delta", "tool_start", "tool_input_delta", "tool_input_delta", "usage", "stop", "stop"]);
  expect(events[0]).toMatchObject({ type: "start", responseId: "r1" });
  expect(events[1]).toEqual({ type: "text_delta", text: "hello" });
  expect(events[2]).toMatchObject({ type: "tool_start", index: 0, id: "c1", name: "lookup" });
  expect(events[3]).toEqual({ type: "tool_input_delta", index: 0, delta: '{"q":' });
  expect(events[4]).toEqual({ type: "tool_input_delta", index: 0, delta: '"x"}' });
  expect(events[5]).toMatchObject({ type: "usage", inputTokens: 4, outputTokens: 2, cacheReadInputTokens: 0 });
});

test("late tool metadata starts exactly one block and preserves arguments", async () => {
  const response = new Response([
    'data: {"choices":[{"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{\\"q\\":"}}]}}]}\n\n',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":2,"id":"late-id","function":{"name":"lookup","arguments":"\\"x\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
    'data: [DONE]\n\n',
  ].join(""), { headers: { "content-type": "text/event-stream" } });
  const events = [];
  for await (const event of readKiroEvents(response, "gpt-5.6")) events.push(event);
  expect(events.filter((event) => event.type === "tool_start")).toEqual([{ type: "tool_start", index: 2, id: "late-id", name: "lookup" }]);
  expect(events.filter((event) => event.type === "tool_input_delta").map((event) => event.type === "tool_input_delta" ? event.delta : "").join("" )).toBe('{"q":"x"}');
});

test("readKiroEvents emits terminal stop on ordinary JSON response", async () => {
  const response = new Response(JSON.stringify({ choices: [{ message: { content: "ok" }, finish_reason: "stop" }] }), { headers: { "content-type": "application/json" } });
  const events = [];
  for await (const event of readKiroEvents(response, "gpt-5.6")) events.push(event);
  expect(events.at(-1)).toEqual({ type: "stop", reason: "end_turn" });
});
