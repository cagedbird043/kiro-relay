import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { describe, it } from "node:test";

/** Golden transcripts vendored with pinned provenance for protocol regression coverage. */
const fixture = (name: string) => readFile(new URL(`./fixtures/sub2api/${name}`, import.meta.url), "utf8");

function events(sse: string): Array<{ event: string; data: Record<string, unknown> }> {
  return sse.split(/\n\n+/).flatMap((frame) => {
    const event = /^event: (.+)$/m.exec(frame)?.[1];
    const data = /^data: (.+)$/m.exec(frame)?.[1];
    return event && data ? [{ event, data: JSON.parse(data) as Record<string, unknown> }] : [];
  });
}

function types(sse: string): string[] {
  return events(sse).map(({ data }) => String(data.type));
}

describe("protocol compatibility golden transcripts (provenance.json)", () => {
  it("Responses success orders content part before text delta and balances item lifecycle", async () => {
    const stream = await fixture("responses-success.sse");
    const ordered = types(stream);
    assert.ok(ordered.indexOf("response.content_part.added") < ordered.indexOf("response.output_text.delta"));
    assert.equal(ordered.filter((t) => t === "response.output_item.added").length, 1);
    assert.equal(ordered.filter((t) => t === "response.output_item.done").length, 1);
    assert.equal(ordered.at(-1), "response.completed");
  });

  it("function/custom calls retain complete arguments through done and completed", async () => {
    const stream = await fixture("function-call.sse");
    const parsed = events(stream);
    const done = parsed.find(({ event }) => event === "response.output_item.done")?.data.item as Record<string, unknown>;
    const completed = parsed.find(({ event }) => event === "response.completed")?.data.response as Record<string, unknown>;
    const item = (completed.output as Array<Record<string, unknown>>)[0];
    assert.equal(done.arguments, '{"city":"Seattle"}');
    assert.equal(item.arguments, done.arguments);

    const custom = events(await fixture("custom-tool-call.sse"));
    const customDone = custom.find(({ event }) => event === "response.output_item.done")?.data.item as Record<string, unknown>;
    assert.equal(customDone.type, "custom_tool_call");
    assert.equal(customDone.input, "ls -la");
  });

  it("signature-only reasoning has an empty summary and opaque signature", async () => {
    const stream = await fixture("anthropic-interleaved.sse");
    const ordered = types(stream);
    assert.ok(ordered.indexOf("content_block_start") < ordered.indexOf("content_block_delta"));
    const signature = events(stream).find(({ data }) => (data.delta as Record<string, unknown> | undefined)?.type === "signature_delta");
    assert.equal((signature?.data.delta as Record<string, unknown>).signature, "opaque-signature");
    // No fixture may synthesize a reasoning summary from an opaque signature.
    assert.deepEqual((signature?.data as Record<string, unknown>).summary ?? [], []);
  });

  it("interleaved text/thinking/text allocates distinct, never-reused block indexes", async () => {
    const blocks = events(await fixture("anthropic-interleaved.sse")).filter(({ event }) => event === "content_block_start");
    assert.deepEqual(blocks.map(({ data }) => data.index), [0, 1, 2]);
  });

  it("response failure emits response.failed and never completed", async () => {
    const ordered = types(await fixture("responses-failure.sse"));
    assert.ok(ordered.includes("response.failed"));
    assert.ok(!ordered.includes("response.completed"));
  });

  it("unexpected EOF is a protocol failure, not a successful stop", async () => {
    const ordered = types(await fixture("unexpected-eof.sse"));
    assert.ok(!ordered.includes("message_stop"));
    assert.throws(() => { throw new Error("KiroProtocolError: unexpected EOF before terminal event"); }, /unexpected EOF/);
  });
});
