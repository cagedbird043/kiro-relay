import { describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHandler, type ExecutorLike } from "../src/server.ts";

const config = (credentialFile: string) => ({
  listen: "127.0.0.1:0",
  models: {
    "claude-opus-5": { contextLength: 1_000_000 },
    "gpt-5.6-sol": { contextLength: 272_000 },
  },
  accounts: [{ credentialFile }],
});
const request = (path: string, body?: unknown) => new Request(`http://relay${path}`, body === undefined ? undefined : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
const upstream = (stream = false, text = "hello") => new Response(stream ? `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n` : JSON.stringify({ choices: [{ message: { content: text }, finish_reason: "stop" }] }), { headers: { "content-type": stream ? "text/event-stream" : "application/json" } });

async function fixture() {
  const dir = await mkdtemp(join(tmpdir(), "kiro-relay-server-"));
  const credentials = join(dir, "credentials.json");
  await writeFile(credentials, JSON.stringify({ accessToken: "old", refreshToken: "refresh" }));
  return { dir, credentials, config: config(credentials) };
}

describe("server routes", () => {
  test("health, models, and token count routes", async () => {
    const f = await fixture();
    try {
      const handler = createHandler(f.config, { execute: async () => ({ response: upstream() }), refreshCredentials: async () => null });
      expect((await handler(new Request("http://relay/healthz"))).status).toBe(200);
      expect((await handler(new Request("http://relay/v1/models"))).json()).resolves.toMatchObject({
        data: [
          { id: "claude-opus-5", object: "model", context_length: 1_000_000 },
          { id: "gpt-5.6-sol", object: "model", context_length: 272_000 },
        ],
      });
      expect((await handler(request("/v1/messages/count_tokens", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }))).status).toBe(200);
    } finally { await rm(f.dir, { recursive: true, force: true }); }
  });

  test("nonstream and stream message routes", async () => {
    const f = await fixture();
    try {
      const calls: unknown[] = [];
      const handler = createHandler(f.config, { execute: async (input) => { calls.push(input); return { response: upstream(input.stream) }; }, refreshCredentials: async () => null });
      const nonstream = await handler(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }));
      expect(nonstream.status).toBe(200); expect((await nonstream.json()).content[0].text).toBe("hello");
      const stream = await handler(request("/v1/messages", { model: "claude-opus-5", stream: true, messages: [{ role: "user", content: "hi" }] }));
      expect(stream.status).toBe(200); const text = await stream.text(); expect(text).toContain("event: message_start"); expect(text).toContain("event: message_stop"); expect(calls).toHaveLength(2);
    } finally { await rm(f.dir, { recursive: true, force: true }); }
  });

  test("401 refresh retries with refreshed credentials and payload", async () => {
    const f = await fixture();
    try {
      const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
      const executor: ExecutorLike = { execute: async (input) => { seen.push(input); return { response: seen.length === 1 ? new Response("expired", { status: 401 }) : upstream() }; }, refreshCredentials: async () => ({ accessToken: "new", refreshToken: "refresh2" }) };
      const response = await createHandler(f.config, executor).call(null, request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }));
      expect(response.status).toBe(200); expect(seen).toHaveLength(2); expect(seen[1].credentials.accessToken).toBe("new"); expect(JSON.parse(await Bun.file(f.credentials).text()).accessToken).toBe("new");
    } finally { await rm(f.dir, { recursive: true, force: true }); }
  });

  test("expired credentials refresh and retry after Kiro returns 403", async () => {
    const f = await fixture();
    try {
      await writeFile(
        f.credentials,
        JSON.stringify({ accessToken: "old", refreshToken: "refresh", expiresAt: "2020-01-01T00:00:00Z" })
      );
      const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
      const executor: ExecutorLike = {
        execute: async (input) => {
          seen.push(input);
          return { response: seen.length === 1 ? new Response("expired", { status: 403 }) : upstream() };
        },
        refreshCredentials: async () => ({ accessToken: "new", refreshToken: "refresh2" }),
      };
      const response = await createHandler(f.config, executor)(
        request("/v1/messages", {
          model: "claude-opus-5",
          messages: [{ role: "user", content: "hi" }],
        })
      );
      expect(response.status).toBe(200);
      expect(seen).toHaveLength(2);
      expect(seen[1].credentials.accessToken).toBe("new");
    } finally {
      await rm(f.dir, { recursive: true, force: true });
    }
  });


  test("routes native Responses requests in stream and nonstream modes", async () => {
    const f = await fixture();
    try {
      const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
      const executor: ExecutorLike = {
        execute: async (input) => {
          seen.push(input);
          return { response: upstream(true, "responses-ok") };
        },
        refreshCredentials: async () => null,
      };
      const handler = createHandler(f.config, executor);
      const nonstream = await handler(
        request("/v1/responses", { model: "gpt-5.6-sol", input: "hello" })
      );
      const nonstreamBody: unknown = await nonstream.json();
      expect(nonstream.status).toBe(200);
      expect(nonstreamBody).toMatchObject({ object: "response", output_text: "responses-ok" });

      const stream = await handler(
        request("/v1/responses", { model: "gpt-5.6-sol", input: "hello", stream: true })
      );
      const streamText = await stream.text();
      expect(stream.status).toBe(200);
      expect(streamText).toContain("event: response.created");
      expect(streamText).toContain("event: response.completed");
      expect(seen).toHaveLength(2);
    } finally {
      await rm(f.dir, { recursive: true, force: true });
    }
  });
  test("returns OpenAI error envelopes for Responses failures", async () => {
    const f = await fixture();
    try {
      const executor: ExecutorLike = {
        execute: async () => ({ response: upstream(true) }),
        refreshCredentials: async () => null,
      };
      const handler = createHandler(f.config, executor);
      const unknown = await handler(
        request("/v1/responses", { model: "missing", input: "hello" })
      );
      const unknownBody: unknown = await unknown.json();
      expect(unknown.status).toBe(400);
      expect(unknownBody).toMatchObject({
        error: { type: "invalid_request_error", param: null, code: null },
      });

      const stateful = await handler(
        request("/v1/responses", {
          model: "gpt-5.6-sol",
          input: "hello",
          previous_response_id: "resp_previous",
        })
      );
      const statefulBody: unknown = await stateful.json();
      expect(stateful.status).toBe(400);
      expect(statefulBody).toMatchObject({ error: { type: "invalid_request_error" } });
    } finally {
      await rm(f.dir, { recursive: true, force: true });
    }
  });
  test("executor and translator errors become Anthropic JSON errors", async () => {

    const f = await fixture();
    try {
      const handler = createHandler(f.config, { execute: async () => { throw new Error("secret upstream payload"); }, refreshCredentials: async () => null });
      const response = await handler(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }));
      expect(response.status).toBe(502); expect(await response.json()).toEqual({ type: "error", error: { type: "api_error", message: "Kiro upstream request failed" } });
    } finally { await rm(f.dir, { recursive: true, force: true }); }
  });
});
