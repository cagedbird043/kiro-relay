import { describe, expect, test } from "bun:test";
import { collectAnthropicResponse, openAIStreamToAnthropic } from "../src/anthropic.ts";

function upstream(lines: string[]): Response {
  return new Response(lines.map((x) => `data: ${x}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } });
}
function events(text: string) {
  return [...text.matchAll(/event: ([^\n]+)\ndata: ([^\n]+)/g)].map((m) => ({ event: m[1], data: JSON.parse(m[2]) }));
}

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
    expect(es.map((e) => e.event)).toEqual([
      "message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
      "content_block_start", "content_block_delta", "content_block_stop",
      "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop",
    ]);
    expect(es.filter((e) => e.event === "content_block_start").map((e) => e.data.index)).toEqual([0, 1, 2]);
    const finalUsage = es.find((e) => e.event === "message_delta")!.data.usage;
    expect(finalUsage.cache_read_input_tokens).toBe(0);
    expect(finalUsage).not.toHaveProperty("cache_creation_input_tokens");
    const collected = await collectAnthropicResponse(upstream([
      JSON.stringify({ choices: [{ message: { role: "assistant", content: "hello", tool_calls: [{ index: 0, id: "tu", function: { name: "lookup", arguments: '{"q":"x"}' } }] }, finish_reason: "tool_calls" }], usage: { prompt_tokens: 4, completion_tokens: 3 } }),
    ])).then((r) => r.json());
    expect(collected.content.find((b: any) => b.type === "tool_use").input).toEqual({ q: "x" });
    expect(collected.content.filter((b: any) => b.type === "tool_use")).toHaveLength(1);
  });
});
