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
const events = async (r: Response) => (await r.text()).trim().split("\n\n").map((frame) => frame.split("\n").find((line) => line.startsWith("data: "))?.slice(6)).filter((data): data is string => !!data && data !== "[DONE]").map((data) => JSON.parse(data) as { type: string; [key: string]: unknown });
describe("Responses formatter", () => {
  test("collects fragmented text, reasoning, tools, and usage", async () => {
    const result = await collectResponses(upstream([
      { id: "up_1", choices: [{ delta: { reasoning_content: "think ", reasoning_signature: "sig_1" } }] },
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
    expect(result.output[0]).toMatchObject({
      type: "reasoning",
      encrypted_content: "sig_1",
      content: [],
    });
    expect(result.output[2]).toMatchObject({ type: "function_call", id: expect.stringMatching(/^fc_/), call_id: "call_1", arguments: '{"q":"x"}' });
  });

  test("emits balanced added/done events, contiguous indexes, sequence numbers, and one completion", async () => {
    const output = await events(responsesStream(upstream([
      { id: "up_2", choices: [{ delta: { reasoning_content: "why", reasoning_signature: "sig_2" } }] },
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
    expect(done[0]?.item).toMatchObject({ type: "reasoning", encrypted_content: "sig_2" });
    expect(output.find((e) => e.type === "response.function_call_arguments.delta")).toMatchObject({ output_index: 2, item_id: fields.id, call_id: fields.call_id });
    expect(output.at(-1)?.type).toBe("response.completed");
  });

  test("keeps signature-only reasoning without inventing a summary", async () => {
    const response = upstream([
      { id: "up_sig", choices: [{ delta: { reasoning_signature: "opaque-signature" } }] },
      { choices: [{ delta: { content: "answer" } }] },
      { choices: [{ delta: {}, finish_reason: "stop" }] },
    ]);
    const result = await collectResponses(response, "gpt-5.6-luna");
    expect(result.output[0]).toMatchObject({
      type: "reasoning",
      encrypted_content: "opaque-signature",
      summary: [],
    });
    expect(result.output_text).toBe("answer");

    const streamed = responsesStream(
      upstream([
        { id: "up_sig_stream", choices: [{ delta: { reasoning_signature: "stream-signature" } }] },
        { choices: [{ delta: { content: "answer" } }] },
        { choices: [{ delta: {}, finish_reason: "stop" }] },
      ]),
      "gpt-5.6-luna"
    );
    const events = (await streamed.text())
      .split("\n")
      .filter((line) => line.startsWith("data: ") && line !== "data: [DONE]")
      .map((line) => JSON.parse(line.slice(6)));
    expect(events.some((event) => event.type === "response.reasoning_summary_text.delta")).toBe(false);
    const reasoningDone = events.find(
      (event) => event.type === "response.output_item.done" && event.item?.type === "reasoning"
    );
    expect(reasoningDone?.item).toMatchObject({
      encrypted_content: "stream-signature",
      summary: [],
    });
  });

  test("reconstructs Codex custom tool calls from Kiro functions", async () => {
    const chunks = [
      {
        id: "up_custom",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_exec",
                  function: { name: "functions__exec", arguments: '{"input":"text(\\\"ok\\\")"}' },
                },
              ],
            },
          },
        ],
      },
      { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
    ];
    const responseTools = {
      functions__exec: { name: "exec", namespace: "functions", custom: true },
    };
    const collected = await collectResponses(upstream(chunks), "gpt-5.6-luna", responseTools);
    expect(collected.output[0]).toMatchObject({
      type: "custom_tool_call",
      call_id: "call_exec",
      name: "exec",
      namespace: "functions",
      input: 'text("ok")',
    });

    const output = await events(
      responsesStream(upstream(chunks), "gpt-5.6-luna", responseTools)
    );
    expect(output.find((event) => event.type === "response.custom_tool_call_input.done")).toMatchObject({
      call_id: "call_exec",
      input: 'text("ok")',
    });
    expect(
      output.find(
        (event) => event.type === "response.output_item.done" &&
          typeof event.item === "object" &&
          event.item !== null &&
          "type" in event.item &&
          event.item.type === "custom_tool_call"
      )
    ).toMatchObject({ item: { namespace: "functions", input: 'text("ok")' } });
  });
});
