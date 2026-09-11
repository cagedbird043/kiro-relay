import { describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHandler, type ExecutorLike } from "../src/server.ts";

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
  { choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 4, completion_tokens: 2 } },
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
      models: { "glm-5": { contextLength: 200_000 } },
      accounts: [{ credentialFile }],
      ...(auth ? { auth } : {}),
    },
    {
      execute: async (input) => {
        seen.push(input);
        return { response: upstream() };
      },
      refreshCredentials: async () => null,
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
        { choices: [{ delta: {}, finish_reason: "tool_calls" }] }
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
    f.reply(() => sse({ choices: [{ delta: { content: "partial" } }] }, { choices: [{ delta: {}, finish_reason: "length" }] }));
    try {
      const body = await (await f.handler(post(TEXT_REQUEST))).json();

      expect(body.choices[0].finish_reason).toBe("length");
    } finally {
      await rm(f.directory, { recursive: true, force: true });
    }
  });

  test("forwards a streaming turn as chat.completion.chunk frames", async () => {
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
