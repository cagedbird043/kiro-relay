import { describe, expect, test } from "bun:test";
import { collectResponses, responsesStream } from "../src/responses/response.ts";

const upstream = (chunks: unknown[]) => new Response(chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join(""));
const itemId = (event: { item?: unknown }) => {
  if (!event.item || typeof event.item !== "object" || !("id" in event.item) || typeof event.item.id !== "string") throw new Error("missing item id");
  return event.item.id;
};
const itemFields = (event: { item?: unknown }): { id: string; call_id: string } => {
  if (!event.item || typeof event.item !== "object" || !("id" in event.item) || !("call_id" in event.item) || typeof event.item.id !== "string" || typeof event.item.call_id !== "string") throw new Error("missing function item fields");
  return { id: event.item.id, call_id: event.item.call_id };
};
const events = async (r: Response) => (await r.text()).trim().split("\n\n").map((frame) => JSON.parse(frame.split("\n").find((line) => line.startsWith("data: "))!.slice(5)) as { type: string; [key: string]: unknown });
describe("Responses formatter", () => {
  test("collects fragmented text, reasoning, tools, and usage", async () => {
    const result = await collectResponses(upstream([
      { id: "up_1", choices: [{ delta: { reasoning_content: "think " } }] },
      { choices: [{ delta: { reasoning_content: "more", content: "hel" } }] },
      { choices: [{ delta: { content: "lo", tool_calls: [{ index: 0, id: "call_1", function: { name: "lookup", arguments: '{"q":' } }] } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"x"}' } }] } }] },
      { usage: { prompt_tokens: 4, completion_tokens: 3, cache_read_input_tokens: 2 }, choices: [{ finish_reason: "tool_calls" }] },
    ]), "gpt-5.6-sol");
    expect(result.id).toBe("up_1");
    expect(result.output_text).toBe("hello");
    expect(result.output.map((item) => item.type)).toEqual(["reasoning", "message", "function_call"]);
    expect(result.output).toHaveLength(3);
    expect(result.usage?.input_tokens_details.cached_tokens).toBe(2);
    expect(result.output[2]).toMatchObject({ type: "function_call", id: expect.stringMatching(/^fc_/), call_id: "call_1", arguments: '{"q":"x"}' });
  });

  test("emits balanced added/done events, contiguous indexes, sequence numbers, and one completion", async () => {
    const output = await events(responsesStream(upstream([
      { id: "up_2", choices: [{ delta: { reasoning_content: "why" } }] },
      { choices: [{ delta: { content: "ok", tool_calls: [{ index: 1, id: "call_2", function: { name: "f", arguments: "{}" } }] } }] },
      { usage: { prompt_tokens: 1, completion_tokens: 2 }, choices: [{ finish_reason: "stop" }] },
    ]), "gpt-5.6-sol"));
    expect(output.filter((e) => e.type === "response.completed")).toHaveLength(1);
    const added = output.filter((e) => e.type === "response.output_item.added");
    const done = output.filter((e) => e.type === "response.output_item.done");
    expect(done.map(itemId)).toEqual(added.map(itemId));
    expect(added.map((e) => e.output_index)).toEqual([0, 1, 2]);
    expect(output.filter((e) => "sequence_number" in e).map((e) => e.sequence_number)).toEqual(output.map((_, i) => i));
    const callAdded = added[2]!;
    const fields = itemFields(callAdded);
    expect(callAdded.item).toMatchObject({ id: expect.stringMatching(/^fc_/), call_id: "call_2", type: "function_call", name: "f" });
    expect(fields.id).not.toBe(fields.call_id);
    expect(output.find((e) => e.type === "response.function_call_arguments.delta")).toMatchObject({ output_index: 2, item_id: fields.id, call_id: fields.call_id });
    expect(output.at(-1)?.type).toBe("response.completed");
  });
});
