import { describe, expect, test } from "bun:test";
import { buildKiroPayload, KIRO_DEFAULT_ORIGIN } from "../src/kiro/request/openai-to-kiro.ts";
import { buildKiroResponsesPayload } from "../src/responses/request.ts";
import { KiroExecutor } from "../src/kiro/executor.ts";

const textEncoder = new TextEncoder();

function crc32(buf: Uint8Array): number {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) crc = table[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function encodeHeader(name: string, value: string): Uint8Array {
  const nameBytes = textEncoder.encode(name);
  const valueBytes = textEncoder.encode(value);
  const header = new Uint8Array(1 + nameBytes.length + 1 + 2 + valueBytes.length);
  let offset = 0;
  header[offset++] = nameBytes.length;
  header.set(nameBytes, offset);
  offset += nameBytes.length;
  header[offset++] = 7;
  header[offset++] = (valueBytes.length >> 8) & 0xff;
  header[offset++] = valueBytes.length & 0xff;
  header.set(valueBytes, offset);
  return header;
}

function frame(eventType: string, payload: unknown): Uint8Array {
  const headers = encodeHeader(":event-type", eventType);
  const payloadBytes = textEncoder.encode(JSON.stringify(payload));
  const totalLength = 12 + headers.length + payloadBytes.length + 4;
  const bytes = new Uint8Array(totalLength);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, totalLength, false);
  view.setUint32(4, headers.length, false);
  view.setUint32(8, crc32(bytes.slice(0, 8)), false);
  bytes.set(headers, 12);
  bytes.set(payloadBytes, 12 + headers.length);
  view.setUint32(totalLength - 4, crc32(bytes.slice(0, totalLength - 4)), false);
  return bytes;
}

function eventStream(...frames: Uint8Array[]): Response {
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of frames) controller.enqueue(chunk);
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "application/vnd.amazon.eventstream" } }
  );
}

function origins(payload: unknown): string[] {
  const found: string[] = [];
  const walk = (value: unknown) => {
    if (Array.isArray(value)) return value.forEach(walk);
    if (!value || typeof value !== "object") return;
    const record = value as Record<string, unknown>;
    if (typeof record.origin === "string") found.push(record.origin);
    Object.values(record).forEach(walk);
  };
  walk(payload);
  return found;
}

const CONVERSATION = {
  messages: [
    { role: "user", content: "first" },
    { role: "assistant", content: "reply" },
    { role: "user", content: "second" },
  ],
};

describe("upstream origin", () => {
  test("stamps every turn with the caller's surface", () => {
    const payload = buildKiroPayload("claude-opus-5", CONVERSATION, false, null, "CLI");
    const stamped = origins(payload);

    expect(stamped.length).toBeGreaterThan(0);
    expect(new Set(stamped)).toEqual(new Set(["CLI"]));
  });

  test("defaults to the IDE surface so an unconfigured relay is unchanged", () => {
    expect(KIRO_DEFAULT_ORIGIN).toBe("AI_EDITOR");
    expect(new Set(origins(buildKiroPayload("claude-opus-5", CONVERSATION, false, null)))).toEqual(
      new Set(["AI_EDITOR"])
    );
  });

  test("carries the surface through the Responses payload too", () => {
    const payload = buildKiroResponsesPayload(
      "gpt-5.6-sol",
      { model: "gpt-5.6-sol", input: "hello" },
      false,
      {},
      "CLI"
    );

    expect(new Set(origins(payload))).toEqual(new Set(["CLI"]));
  });
});

describe("terminal event expectation", () => {
  // A stream that stops right after its last content frame: no messageStop, no
  // metering. That is malformed from the IDE surface and normal elsewhere.
  const truncated = () => eventStream(frame("assistantResponseEvent", { content: "ok" }));

  test("rejects a stream that ends without terminal evidence by default", async () => {
    const executor = new KiroExecutor();

    await expect(executor.transformEventStreamToSSE(truncated(), "glm-5").text()).rejects.toThrow(
      /terminal event/
    );
  });

  test("accepts the same stream when upstream never promises a terminal event", async () => {
    const executor = new KiroExecutor();

    const text = await executor
      .transformEventStreamToSSE(truncated(), "glm-5", { terminalEventExpected: false })
      .text();

    expect(text).toContain('"content":"ok"');
    expect(text).toContain('"finish_reason":"stop"');
    expect(text).toContain("data: [DONE]");
  });

  test("still rejects a stream cut mid-frame even without a terminal promise", async () => {
    const executor = new KiroExecutor();
    const whole = frame("assistantResponseEvent", { content: "ok" });
    const cut = eventStream(whole.slice(0, whole.length - 6));

    // Half a frame left in the buffer is truncation on any surface.
    await expect(
      executor.transformEventStreamToSSE(cut, "glm-5", { terminalEventExpected: false }).text()
    ).rejects.toThrow(/terminal event/);
  });
});
