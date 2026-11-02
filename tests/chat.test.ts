import { describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHandler, type ExecutorLike } from "../src/server.ts";
import { chatCompletionStream, collectChatCompletion } from "../src/chat.ts";

/** Mirrors what the executor emits: a full chat.completion.chunk envelope per frame. */
function sse(...chunks: Array<Record<string, unknown>>): Response {
  const body =
    chunks
      .map((chunk) =>
        `data: ${JSON.stringify({ id: "chatcmpl-upstream", object: "chat.completion.chunk", created: 0, model: "glm-5", ...chunk })}\n\n`
      )
      .join("") + "data: [DONE]\n\n";
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

const TEXT_TURN = [
  { choices: [{ delta: { role: "assistant", content: "hello" } }] },
  { choices: [{ delta: {}, finish_reason: "end_turn" }], usage: { prompt_tokens: 4, completion_tokens: 2 } },
];

async function fixture(auth?: { keys: string[] }) {
  const directory = await mkdtemp(join(tmpdir(), "kiro-relay-chat-"));
  const credentialFile = join(directory, "credentials.json");
  await writeFile(credentialFile, JSON.stringify({ accessToken: "token", refreshToken: "refresh" }));

  const seen: Array<Parameters<ExecutorLike["execute"]>[0]> = [];
  let upstream = () => sse(...TEXT_TURN);
  const handler = createHandler(
    {
      listen: "127.0.0.1:0",
      models: { "glm-5": { contextLength: 200_000 }, "gpt-5.6-sol": { contextLength: 1_000_000 } },
      accounts: [{ credentialFile }],
      ...(auth ? { auth } : {}),
    },
    {
      execute: async (input) => {
        seen.push(input);
        return { response: upstream() };
      },
      refreshCredentials: async () => ({ ok: false as const, kind: "transient" as const, code: "refresh_unavailable" }),
    }
  );
  return {
    directory,
    handler,
    seen,
    reply(next: () => Response) {
      upstream = next;
    },
  };
}

function post(body: unknown, headers: Record<string, string> = {}): Request {
  return new Request("http://relay/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

const TEXT_REQUEST = { model: "glm-5", messages: [{ role: "user", content: "hi" }] };

describe("chat completions", () => {
  test("folds a non-streaming turn into a chat.completion object", async () => {
    const f = await fixture();
    try {
      const response = await f.handler(post(TEXT_REQUEST));
      const body = await response.json();

      expect(response.status).toBe(200);
      expect(body.object).toBe("chat.completion");
      expect(body.model).toBe("glm-5");
      expect(body.choices[0].message).toEqual({ role: "assistant", content: "hello" });
      expect(body.choices[0].finish_reason).toBe("stop");
      expect(body.usage).toEqual({ prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 });
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("reports a tool-only turn the way the Chat Completions schema does", async () => {
    const f = await fixture();
    f.reply(() =>
      sse(
        { choices: [{ delta: { tool_calls: [{ index: 0, id: "call_1", function: { name: "lookup", arguments: '{"q":' } }] } }] },
        { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"x"}' } }] } }] },
        { choices: [{ delta: {}, finish_reason: "tool_use" }] }
      )
    );
    try {
      const body = await (await f.handler(post(TEXT_REQUEST))).json();

      expect(body.choices[0].message.content).toBeNull();
      expect(body.choices[0].message.tool_calls).toEqual([
        { id: "call_1", type: "function", function: { name: "lookup", arguments: '{"q":"x"}' } },
      ]);
      expect(body.choices[0].finish_reason).toBe("tool_calls");
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("maps a truncated turn to the OpenAI length reason", async () => {
    const f = await fixture();
    f.reply(() => sse({ choices: [{ delta: { content: "partial" } }] }, { choices: [{ delta: {}, finish_reason: "max_tokens" }] }));
    try {
      const body = await (await f.handler(post(TEXT_REQUEST))).json();

      expect(body.choices[0].finish_reason).toBe("length");
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("reencodes a streaming turn as public chat.completion.chunk frames", async () => {
    const f = await fixture();
    try {
      const response = await f.handler(post({ ...TEXT_REQUEST, stream: true }));
      const text = await response.text();

      expect(response.headers.get("content-type")).toContain("text/event-stream");
      expect(text).toContain('"object":"chat.completion.chunk"');
      expect(text).toContain('"content":"hello"');
      expect(text).toContain("data: [DONE]");
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("passes OpenAI messages through without the Anthropic conversion", async () => {
    const f = await fixture();
    try {
      await f.handler(
        post({
          model: "glm-5",
          messages: [
            { role: "system", content: "SYSTEM-MARKER" },
            { role: "user", content: "hi" },
          ],
        })
      );

      // A system entry in `messages` is OpenAI-native; the Anthropic surface
      // carries it as a top-level field and would have dropped it here.
      expect(JSON.stringify(f.seen[0].body)).toContain("SYSTEM-MARKER");
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("accepts an empty stop list but refuses a populated one", async () => {
    const f = await fixture();
    try {
      expect((await f.handler(post({ ...TEXT_REQUEST, stop: [] }))).status).toBe(200);
      expect((await f.handler(post({ ...TEXT_REQUEST, stop: null }))).status).toBe(200);

      const rejected = await f.handler(post({ ...TEXT_REQUEST, stop: ["END"] }));
      expect(rejected.status).toBe(400);
      expect(f.seen).toHaveLength(2);
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("reports errors in the OpenAI envelope rather than the Anthropic one", async () => {
    const f = await fixture();
    try {
      const body = await (await f.handler(post({ model: "not-a-model", messages: [] }))).json();

      expect(body.error.message).toContain("Unknown model");
      expect(body.type).toBeUndefined();
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });
});

describe("browser access", () => {
  test("rejects a request with no key once keys are configured", async () => {
    const f = await fixture({ keys: ["sk-test"] });
    try {
      const response = await f.handler(post(TEXT_REQUEST));

      expect(response.status).toBe(401);
      expect(f.seen).toHaveLength(0);
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("accepts either the OpenAI or the Anthropic key header", async () => {
    const f = await fixture({ keys: ["sk-test"] });
    try {
      expect((await f.handler(post(TEXT_REQUEST, { authorization: "Bearer sk-test" }))).status).toBe(200);
      expect((await f.handler(post(TEXT_REQUEST, { "x-api-key": "sk-test" }))).status).toBe(200);
      expect((await f.handler(post(TEXT_REQUEST, { authorization: "Bearer wrong" }))).status).toBe(401);
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("leaves the health probe reachable without a key", async () => {
    const f = await fixture({ keys: ["sk-test"] });
    try {
      const response = await f.handler(new Request("http://relay/healthz"));

      expect(response.status).toBe(200);
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("answers the preflight and marks every response cross-origin", async () => {
    const f = await fixture({ keys: ["sk-test"] });
    try {
      const preflight = await f.handler(
        new Request("http://relay/v1/chat/completions", { method: "OPTIONS" })
      );
      expect(preflight.status).toBe(204);
      expect(preflight.headers.get("access-control-allow-origin")).toBe("*");
      expect(preflight.headers.get("access-control-allow-headers")).toContain("authorization");

      // The preflight must not be the only thing carrying the headers, or the
      // browser rejects the actual response.
      const models = await f.handler(
        new Request("http://relay/v1/models", { headers: { authorization: "Bearer sk-test" } })
      );
      expect(models.headers.get("access-control-allow-origin")).toBe("*");
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });
});

test("Chat streaming and collected usage include cache reads and writes without internal fields", async () => {
  const source = () => sse(
    { choices: [{ delta: { reasoning_content: "why", reasoning_signature: "private-signature", reasoning_redacted: "private-hidden" } }] },
    { choices: [{ delta: { content: "ok" }, finish_reason: "max_tokens", kiro_stop_details: { private: true } }], usage: { prompt_tokens: 3, completion_tokens: 5, cache_read_input_tokens: 7, cache_creation_input_tokens: 11, estimated: true } },
  );
  const collected = await (await collectChatCompletion(source(), "glm-5")).json();
  const text = await chatCompletionStream(source(), "glm-5").text();
  const chunks = text.split("\n\n").filter(frame => frame.startsWith("data: {")).map(frame => JSON.parse(frame.slice(6)));
  expect(chunks[0].choices[0].delta.role).toBe("assistant");
  expect(chunks.find(chunk => chunk.choices[0]?.finish_reason)?.choices[0].finish_reason).toBe("length");
  expect(collected.usage).toEqual({ prompt_tokens: 21, completion_tokens: 5, total_tokens: 26, prompt_tokens_details: { cached_tokens: 7 } });
  expect(chunks.find(chunk => chunk.usage)?.usage).toEqual(collected.usage);
  expect(text).toContain('"reasoning_content":"why"');
  for (const privateField of ["private-signature", "private-hidden", "kiro_stop_details", "estimated", "cache_creation_input_tokens", "max_tokens"]) expect(text).not.toContain(privateField);
});

test("Chat emits content before upstream termination", async () => {
  let input!: ReadableStreamDefaultController<Uint8Array>;
  const source = new Response(new ReadableStream<Uint8Array>({ start(controller) { input = controller; } }));
  const reader = chatCompletionStream(source, "glm-5").body!.getReader();
  input.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"first"}}]}\n\n'));
  expect(new TextDecoder().decode((await reader.read()).value)).toContain('"role":"assistant"');
  expect(new TextDecoder().decode((await reader.read()).value)).toContain('"content":"first"');
  await reader.cancel();
});

test("server logs one metadata-only failure for all streaming protocols and pause_turn", async () => {
  const f = await fixture();
  const previous = console.info;
  const logs: unknown[][] = [];
  console.info = (...args: unknown[]) => { logs.push(args); };
  try {
    for (const [path, failure] of [
      ["/v1/chat/completions", "eof"], ["/v1/messages", "eof"], ["/v1/responses", "eof"], ["/v1/responses", "pause"],
      ["/v1/chat/completions", "throw"], ["/v1/messages", "throw"], ["/v1/responses", "throw"],
    ] as const) {
      logs.length = 0;
      f.reply(() => failure === "throw"
        ? new Response(new ReadableStream({ start(controller) { controller.error(new Error("private-error-message")); } }))
        : new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: "private-body" }, ...(failure === "pause" ? { finish_reason: "pause_turn" } : {}) }] })}\n\n`));
      const body = path === "/v1/responses" ? { model: "gpt-5.6-sol", input: "hello", stream: true } : { ...TEXT_REQUEST, stream: true, max_tokens: 16 };
      const response = await f.handler(new Request(`http://relay${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }));
      const output = await response.text();
      expect(output).toContain(path === "/v1/responses" ? "response.failed" : '"error"');
      const failures = logs.filter(args => JSON.stringify(args).includes("stream_failed"));
      expect(failures).toHaveLength(1);
      expect(failures[0]?.[1]).toMatchObject({ requestId: response.headers.get("request-id"), code: failure === "pause" ? "pause_turn" : failure === "throw" ? "stream_error" : "unexpected_eof" });
      expect(JSON.stringify(logs)).not.toContain("private-body");
      expect(JSON.stringify(logs)).not.toContain("private-error-message");
    }
  } finally { console.info = previous; await rm(f.directory, { recursive: true, force: true }); }
});

test("protocol parameter rejections return 400 before execute", async () => {
  const f = await fixture();
  try {
    for (const [path, body, message] of [
      ["/v1/messages", { ...TEXT_REQUEST, top_k: null, max_tokens: 16 }, "Unsupported top_k on this relay"],
      ["/v1/responses", { model: "gpt-5.6-sol", input: "hello", text: { format: { type: "json_object" } } }, "Unsupported text.format on this relay"],
    ] as const) {
      const response = await f.handler(new Request(`http://relay${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }));
      expect(response.status).toBe(400);
      expect((await response.json()).error.message).toBe(message);
    }
    expect(f.seen).toHaveLength(0);
  } finally { await rm(f.directory, { recursive: true, force: true }); }
});

test("real HTTP smoke reencodes executor events and cache-inclusive usage", async () => {
  const f = await fixture();
  f.reply(() => sse(
    { choices: [{ delta: { reasoning_content: "reason", reasoning_signature: "hidden-signature" } }] },
    { choices: [{ delta: { tool_calls: [{ index: 0, id: "call", function: { name: "read", arguments: "{}" } }] }, finish_reason: "tool_use" }], usage: { prompt_tokens: 2, completion_tokens: 3, cache_read_input_tokens: 10, cache_creation_input_tokens: 4, estimated: true } },
  ));
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: f.handler });
  try {
    for (const stream of [false, true]) {
      const response = await fetch(`http://127.0.0.1:${server.port}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ...TEXT_REQUEST, stream }) });
      expect(response.status).toBe(200);
      if (stream) {
        const text = await response.text();
        expect(text).toContain('"finish_reason":"tool_calls"');
        expect(text).toContain('"prompt_tokens":16');
        expect(text).not.toContain("hidden-signature");
        expect(text).not.toContain("estimated");
      } else {
        const body = await response.json();
        expect(body.choices[0].finish_reason).toBe("tool_calls");
        expect(body.usage).toEqual({ prompt_tokens: 16, completion_tokens: 3, total_tokens: 19, prompt_tokens_details: { cached_tokens: 10 } });
      }
    }
  } finally { await server.stop(true); await rm(f.directory, { recursive: true, force: true }); }
});
