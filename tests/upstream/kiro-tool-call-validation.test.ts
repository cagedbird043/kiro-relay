import test from "node:test";
import assert from "node:assert/strict";

import { KiroExecutor } from "../../src/kiro/executor.ts";
import { withFetch } from "../fetch-stub.ts";

function createKiroErrorTransform(): TransformStream<Uint8Array, Uint8Array> {
  let buffered = "";
  return new TransformStream({
    transform(chunk, controller) {
      buffered += new TextDecoder().decode(chunk, { stream: true });
      for (const event of buffered.split("\n\n").slice(0, -1)) {
        const data = event.match(/^data:\s*(.+)$/m)?.[1];
        if (!data) continue;
        const payload = JSON.parse(data);
        if (payload.error) {
          controller.enqueue(new TextEncoder().encode(
            `event: response.failed\ndata: ${JSON.stringify(payload)}\n\n`,
          ));
        }
      }
      buffered = buffered.includes("\n\n") ? buffered.slice(buffered.lastIndexOf("\n\n") + 2) : buffered;
    },
  });
}

const textEncoder = new TextEncoder();

function crc32(bytes: Uint8Array): number {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let value = i;
    for (let bit = 0; bit < 8; bit++) {
      value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[i] = value >>> 0;
  }

  let value = 0xffffffff;
  for (const byte of bytes) {
    value = table[(value ^ byte) & 0xff] ^ (value >>> 8);
  }
  return (value ^ 0xffffffff) >>> 0;
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

function encodeEventFrame(eventType: string, payload: Record<string, unknown>): Uint8Array {
  const headers = encodeHeader(":event-type", eventType);
  const payloadBytes = textEncoder.encode(JSON.stringify(payload));
  const totalLength = 12 + headers.length + payloadBytes.length + 4;
  const frame = new Uint8Array(totalLength);
  const view = new DataView(frame.buffer);
  view.setUint32(0, totalLength, false);
  view.setUint32(4, headers.length, false);
  view.setUint32(8, crc32(frame.slice(0, 8)), false);
  frame.set(headers, 12);
  frame.set(payloadBytes, 12 + headers.length);
  view.setUint32(totalLength - 4, crc32(frame.slice(0, totalLength - 4)), false);
  return frame;
}

function buildEventStreamResponse(frames: Uint8Array[]): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const frame of frames) controller.enqueue(frame);
        controller.close();
      },
    }),
    { status: 200 }
  );
}

function collectSseJson(text: string): Array<Record<string, unknown>> {
  return text
    .split("\n")
    .filter((line) => line.startsWith("data: "))
    .map((line) => line.slice(6).trim())
    .filter((line) => line && line !== "[DONE]")
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

function toolDeltas(text: string): Array<Record<string, unknown>> {
  return collectSseJson(text).flatMap((chunk) => {
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = choices[0] as Record<string, unknown> | undefined;
    const delta = choice?.delta as Record<string, unknown> | undefined;
    return Array.isArray(delta?.tool_calls)
      ? (delta.tool_calls as Array<Record<string, unknown>>)
      : [];
  });
}

test("Kiro rejects a completed malformed tool_call wrapper without emitting a fake function", async () => {
  const executor = new KiroExecutor();
  const response = buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", {
      toolUseId: "wrapper_1",
      name: "tool_call",
      input: { arguments: { query: "router" } },
    }),
    encodeEventFrame("messageStopEvent", {}),
  ]);

  const text = await executor.transformEventStreamToSSE(response, "kiro-model").text();

  assert.match(text, /invalid_kiro_tool_call/);
  assert.match(text, /missing nested MCP tool name/);
  assert.doesNotMatch(text, /"name":"tool_call"/);
  assert.match(text, /data: \[DONE\]/);
});

test("Kiro validates a wrapper only after string fragments are complete", async () => {
  const executor = new KiroExecutor();
  const response = buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", { toolUseId: "wrapper_1", name: "tool_call" }),
    encodeEventFrame("toolUseEvent", {
      toolUseId: "wrapper_1",
      name: "tool_call",
      input: '{"name":"mcp_search",',
    }),
    encodeEventFrame("toolUseEvent", {
      toolUseId: "wrapper_1",
      name: "tool_call",
      input: '"arguments":{"query":"router"}}',
    }),
    encodeEventFrame("messageStopEvent", {}),
  ]);

  const text = await executor.transformEventStreamToSSE(response, "kiro-model").text();
  const deltas = toolDeltas(text);
  const args = deltas
    .map((delta) => {
      const fn = delta.function as Record<string, unknown> | undefined;
      return typeof fn?.arguments === "string" ? fn.arguments : "";
    })
    .join("");

  assert.doesNotMatch(text, /invalid_kiro_tool_call/);
  assert.equal((deltas[0].function as Record<string, unknown>).name, "tool_call");
  assert.deepEqual(JSON.parse(args), { name: "mcp_search", arguments: { query: "router" } });
});

test("Kiro waits for the final growing object before validating a wrapper", async () => {
  const executor = new KiroExecutor();
  const response = buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", {
      toolUseId: "wrapper_1",
      name: "tool_call",
      input: { arguments: { query: "router" } },
    }),
    encodeEventFrame("toolUseEvent", {
      toolUseId: "wrapper_1",
      name: "tool_call",
      input: { name: "mcp_search", arguments: { query: "router" } },
    }),
    encodeEventFrame("messageStopEvent", {}),
  ]);

  const text = await executor.transformEventStreamToSSE(response, "kiro-model").text();
  const deltas = toolDeltas(text);
  const args = deltas
    .map((delta) => {
      const fn = delta.function as Record<string, unknown> | undefined;
      return typeof fn?.arguments === "string" ? fn.arguments : "";
    })
    .join("");

  assert.doesNotMatch(text, /invalid_kiro_tool_call/);
  assert.deepEqual(JSON.parse(args), { name: "mcp_search", arguments: { query: "router" } });
});

test("Kiro assigns direct tools before buffered wrappers when interleaved", async () => {
  const executor = new KiroExecutor();
  const response = buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", {
      toolUseId: "wrapper_1",
      name: "tool_call",
      input: { name: "mcp_search", arguments: { query: "router" } },
    }),
    encodeEventFrame("toolUseEvent", {
      toolUseId: "direct_1",
      name: "read_file",
      input: { path: "README.md" },
    }),
    encodeEventFrame("messageStopEvent", {}),
  ]);

  const text = await executor.transformEventStreamToSSE(response, "kiro-model").text();
  const starts = toolDeltas(text).filter((delta) => typeof delta.id === "string");

  assert.deepEqual(
    starts.map((delta) => (delta.function as Record<string, unknown>).name),
    ["read_file", "tool_call"]
  );
  assert.deepEqual(
    starts.map((delta) => delta.index),
    [0, 1]
  );
});

test("Kiro cancels the upstream body after an invalid wrapper", async () => {
  const executor = new KiroExecutor();
  let cancelled = false;
  let resolveCancelled: (() => void) | undefined;
  const cancelledPromise = new Promise<void>((resolve) => {
    resolveCancelled = resolve;
  });
  const response = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encodeEventFrame("toolUseEvent", {
            toolUseId: "wrapper_1",
            name: "tool_call",
            input: { arguments: { query: "router" } },
          })
        );
        controller.enqueue(encodeEventFrame("messageStopEvent", {}));
      },
      cancel() {
        cancelled = true;
        resolveCancelled?.();
      },
    })
  );

  const textPromise = executor.transformEventStreamToSSE(response, "kiro-model").text();
  await Promise.race([cancelledPromise, new Promise((resolve) => setTimeout(resolve, 250))]);
  await textPromise;

  assert.equal(cancelled, true);
});

test("Kiro stream errors become Responses response.failed events", async () => {
  const transform = createKiroErrorTransform();
  const writer = transform.writable.getWriter();
  const responseText = new Response(transform.readable).text();

  await writer.write(
    textEncoder.encode(
      `data: ${JSON.stringify({
        error: {
          message: "Invalid Kiro tool_call payload: missing nested MCP tool name at input.name",
          type: "invalid_request_error",
          code: "invalid_kiro_tool_call",
        },
      })}\n\n`
    )
  );
  await writer.close();
  const text = await responseText;

  assert.match(text, /event: response\.failed/);
  assert.match(text, /invalid_kiro_tool_call/);
  assert.match(text, /missing nested MCP tool name/);
  assert.doesNotMatch(text, /response\.output_item\.added/);
});

test("客户端自己声明的 tool_call 工具按普通工具透传，不套用 MCP 包装校验", async () => {
  const calls = { calls: [{ name: "image_generate", arguments: { prompt: "胡同里的橘猫" } }] };
  await withFetch(async () => buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", { toolUseId: "call_1", name: "tool_call", input: calls }),
    encodeEventFrame("messageStopEvent", {}),
  ]), async () => {
    const result = await new KiroExecutor().execute({
      model: "claude-opus-5",
      body: { conversationState: { currentMessage: { userInputMessage: {
        content: "画一张",
        userInputMessageContext: { tools: [{ toolSpecification: { name: "tool_call", description: "Invoke deferred tools", inputSchema: { json: { type: "object" } } } }] },
      } } } },
      stream: true,
      credentials: { accessToken: "fake" },
    });
    const text = await result.response.text();
    assert.doesNotMatch(text, /invalid_kiro_tool_call/);
    const deltas = toolDeltas(text);
    assert.equal((deltas[0].function as Record<string, unknown>).name, "tool_call");
    const args = deltas.map((delta) => (delta.function as Record<string, unknown>).arguments ?? "").join("");
    assert.deepEqual(JSON.parse(args), calls);
  });
});
