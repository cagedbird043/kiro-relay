import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
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

test("刷新重试前释放旧响应且不影响新响应内容", async () => {
  const f = await fixture();
  let calls = 0;
  let cancelled = false;
  try {
    const handler = createHandler(f.config, {
      execute: async () => {
        if (++calls === 1) return { response: new Response(new ReadableStream({ cancel() { cancelled = true; } }), { status: 401 }) };
        expect(cancelled).toBe(true);
        return { response: upstream(false, "恢复完成") };
      },
      refreshCredentials: async () => ({ accessToken: "new", refreshToken: "rotated", expiresIn: 3600 }),
    });
    const response = await handler(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "任务" }] }));
    expect(response.status).toBe(200);
    expect((await response.json()).content[0].text).toBe("恢复完成");
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test("轮换后保存失败保留新凭据并在文件恢复后共享重试保存", async () => {
  const f = await fixture();
  const backup = join(f.dir, "old.json");
  let refreshes = 0;
  let executions = 0;
  try {
    await writeFile(f.credentials, JSON.stringify({ accessToken: "expired", refreshToken: "old", expiresAt: "2020-01-01T00:00:00Z" }));
    const handler = createHandler(f.config, {
      refreshCredentials: async () => {
        refreshes++;
        if (refreshes > 1) return { error: "unrecoverable_refresh_error", code: "invalid_grant" };
        // 目录占用目标文件名，确定性地令原子替换失败，不依赖进程权限。
        await rename(f.credentials, backup);
        await mkdir(f.credentials);
        return { accessToken: "new", refreshToken: "rotated", expiresIn: 3600 };
      },
      execute: async ({ credentials }) => {
        executions++;
        return { response: credentials.accessToken === "new" ? upstream(false, "恢复完成") : new Response("expired", { status: 401 }) };
      },
    });
    const body = { model: "claude-opus-5", messages: [{ role: "user", content: "任务" }] };
    expect((await handler(request("/v1/messages", body))).status).toBe(500);
    expect((await handler(request("/v1/messages", body))).status).toBe(500);
    expect(executions).toBe(0);
    await rm(f.credentials, { recursive: true });
    await rename(backup, f.credentials);
    const responses = await Promise.all([handler(request("/v1/messages", body)), handler(request("/v1/messages", body))]);
    expect(responses.map(response => response.status)).toEqual([200, 200]);
    for (const response of responses) expect((await response.json()).content[0].text).toBe("恢复完成");
    expect(refreshes).toBe(1);
    expect(JSON.parse(await Bun.file(f.credentials).text()).refreshToken).toBe("rotated");
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test("失败响应保留协议原因与关联编号且不泄漏原始正文", async () => {
  const f = await fixture();
  const original = console.info;
  const logs: unknown[][] = [];
  console.info = (...args: unknown[]) => { logs.push(args); };
  try {
    for (const path of ["/v1/messages", "/v1/responses"]) {
      for (const reason of ["INVALID_MODEL_ID", "THINKING_SIGNATURE_INVALID"]) {
        const handler = createHandler(f.config, { execute: async () => ({ response: Response.json({ reason, message: "敏感原始正文" }, { status: 400 }) }), refreshCredentials: async () => null });
        const response = await handler(request(path, path.endsWith("responses") ? { model: "gpt-5.6-sol", input: "hi" } : { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }));
        expect(response.status).toBe(400);
        const text = await response.text();
        expect(text).toContain(reason);
        expect(text).not.toContain("敏感原始正文");
        expect(logs.at(-1)).toEqual(["[REQUEST]", expect.objectContaining({ requestId: response.headers.get("request-id"), phase: "failed", code: reason })]);
      }
    }
    expect(JSON.stringify(logs)).not.toContain("敏感原始正文");
  } finally { console.info = original; await rm(f.dir, { recursive: true, force: true }); }
});

test("不支持的停止序列在发送上游前明确拒绝", async () => {
  const f = await fixture();
  let calls = 0;
  try {
    const handler = createHandler(f.config, { execute: async () => { calls++; return { response: upstream() }; }, refreshCredentials: async () => null });
    const body = { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] };
    const rejected = await handler(request("/v1/messages", { ...body, stop_sequences: ["END"] }));
    expect(rejected.status).toBe(400);
    expect((await rejected.json()).error.type).toBe("invalid_request_error");
    expect(calls).toBe(0);
    const accepted = await handler(request("/v1/messages", { ...body, stop_sequences: [] }));
    expect(accepted.status).toBe(200);
    expect(calls).toBe(1);
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

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
        models: [
          {
            slug: "gpt-5.6-sol",
            base_instructions: "You are Codex, a coding agent based on GPT-5.",
            use_responses_lite: true,
            context_window: 272_000,
          },
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

  test("401 refresh retries once and persists the refreshed expiry", async () => {
    const f = await fixture();
    try {
      const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
      const executor: ExecutorLike = { execute: async (input) => { seen.push(input); return { response: seen.length === 1 ? new Response("expired", { status: 401 }) : upstream() }; }, refreshCredentials: async () => ({ accessToken: "new", refreshToken: "refresh2", expiresIn: 3600 }) };
      const before = Date.now();
      const response = await createHandler(f.config, executor).call(null, request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }));
      const saved = JSON.parse(await Bun.file(f.credentials).text()) as { accessToken: string; expiresAt: string };
      expect(response.status).toBe(200); expect(seen).toHaveLength(2); expect(seen[1].credentials.accessToken).toBe("new"); expect(saved.accessToken).toBe("new"); expect(Date.parse(saved.expiresAt)).toBeGreaterThanOrEqual(before + 3_599_000); expect(Date.parse(saved.expiresAt)).toBeLessThanOrEqual(Date.now() + 3_601_000);
    } finally { await rm(f.dir, { recursive: true, force: true }); }
  });

  test("expired credentials refresh before the upstream request", async () => {
    const f = await fixture();
    try {
      await writeFile(
        f.credentials,
        JSON.stringify({ accessToken: "old", refreshToken: "refresh", expiresAt: "2020-01-01T00:00:00Z" })
      );
      const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
      let refreshes = 0;
      const executor: ExecutorLike = {
        execute: async (input) => {
          seen.push(input);
          return { response: upstream() };
        },
        refreshCredentials: async () => {
          refreshes += 1;
          return { accessToken: "new", refreshToken: "refresh2", expiresIn: 3600 };
        },
      };
      const response = await createHandler(f.config, executor)(
        request("/v1/messages", {
          model: "claude-opus-5",
          messages: [{ role: "user", content: "hi" }],
        })
      );
      expect(response.status).toBe(200);
      expect(refreshes).toBe(1);
      expect(seen).toHaveLength(1);
      expect(seen[0].credentials.accessToken).toBe("new");
    } finally {
      await rm(f.dir, { recursive: true, force: true });
    }
  });

  test("concurrent expired requests share one refresh and use the persisted token", async () => {
    const f = await fixture();
    try {
      await writeFile(f.credentials, JSON.stringify({ accessToken: "old", refreshToken: "refresh", expiresAt: "2020-01-01T00:00:00Z" }));
      let releaseRefresh: (value: unknown) => void = () => undefined;
      const refreshGate = new Promise<unknown>((resolve) => {
        releaseRefresh = resolve;
      });
      let announceRefresh: () => void = () => undefined;
      const refreshStarted = new Promise<void>((resolve) => {
        announceRefresh = resolve;
      });
      let refreshes = 0;
      const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
      const executor: ExecutorLike = {
        execute: async (input) => {
          seen.push(input);
          return { response: upstream() };
        },
        refreshCredentials: async () => {
          refreshes += 1;
          announceRefresh();
          return refreshGate;
        },
      };
      const handler = createHandler(f.config, executor);
      const first = handler(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "one" }] }));
      await refreshStarted;
      const second = handler(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "two" }] }));
      releaseRefresh({ accessToken: "shared", refreshToken: "rotated", expiresIn: 3600 });
      const responses = await Promise.all([first, second]);
      expect(responses.map((response) => response.status)).toEqual([200, 200]);
      expect(refreshes).toBe(1);
      expect(seen).toHaveLength(2);
      expect(seen.every((input) => input.credentials.accessToken === "shared")).toBe(true);
      expect(JSON.parse(await Bun.file(f.credentials).text()).refreshToken).toBe("rotated");
    } finally {
      await rm(f.dir, { recursive: true, force: true });
    }
  });

  test("concurrent 401 responses share one refresh-token rotation", async () => {
    const f = await fixture();
    try {
      let releaseRefresh: (value: unknown) => void = () => undefined;
      const refreshGate = new Promise<unknown>((resolve) => {
        releaseRefresh = resolve;
      });
      let announceRefresh: () => void = () => undefined;
      const refreshStarted = new Promise<void>((resolve) => {
        announceRefresh = resolve;
      });
      let refreshes = 0;
      const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
      const executor: ExecutorLike = {
        execute: async (input) => {
          seen.push(input);
          return {
            response: input.credentials.accessToken === "old"
              ? new Response("expired", { status: 401 })
              : upstream(),
          };
        },
        refreshCredentials: async () => {
          refreshes += 1;
          announceRefresh();
          return refreshGate;
        },
      };
      const handler = createHandler(f.config, executor);
      const first = handler(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "one" }] }));
      const second = handler(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "two" }] }));
      await refreshStarted;
      releaseRefresh({ accessToken: "shared", refreshToken: "rotated", expiresIn: 3600 });
      const responses = await Promise.all([first, second]);
      expect(responses.map((response) => response.status)).toEqual([200, 200]);
      expect(refreshes).toBe(1);
      expect(seen.filter((input) => input.credentials.accessToken === "old")).toHaveLength(2);
      expect(seen.filter((input) => input.credentials.accessToken === "shared")).toHaveLength(2);
    } finally {
      await rm(f.dir, { recursive: true, force: true });
    }
  });

  test("a failed early refresh uses the still-valid access token", async () => {
    const f = await fixture();
    try {
      await writeFile(f.credentials, JSON.stringify({ accessToken: "still-valid", refreshToken: "refresh", expiresAt: new Date(Date.now() + 60_000).toISOString() }));
      const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
      const executor: ExecutorLike = {
        execute: async (input) => {
          seen.push(input);
          return { response: upstream() };
        },
        refreshCredentials: async () => null,
      };
      const response = await createHandler(f.config, executor)(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }));
      expect(response.status).toBe(200);
      expect(seen).toHaveLength(1);
      expect(seen[0].credentials.accessToken).toBe("still-valid");
    } finally {
      await rm(f.dir, { recursive: true, force: true });
    }
  });

  test("a failed refresh never sends an already expired access token", async () => {
    const f = await fixture();
    try {
      await writeFile(f.credentials, JSON.stringify({ accessToken: "expired", refreshToken: "refresh", expiresAt: "2020-01-01T00:00:00Z" }));
      let executions = 0;
      const executor: ExecutorLike = {
        execute: async () => {
          executions += 1;
          return { response: upstream() };
        },
        refreshCredentials: async () => ({ error: "unrecoverable_refresh_error", code: "invalid_grant" }),
      };
      const response = await createHandler(f.config, executor)(request("/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }));
      expect(response.status).toBe(401);
      expect(executions).toBe(0);
      expect(JSON.parse(await Bun.file(f.credentials).text()).accessToken).toBe("expired");
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
  test("preserves upstream overload statuses and Retry-After on both protocols", async () => {
    const cases = [
      { path: "/v1/messages", body: { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }, type429: "rate_limit_error", type529: "overloaded_error" },
      { path: "/v1/responses", body: { model: "gpt-5.6-sol", input: "hi" }, type429: "rate_limit_error", type529: "overloaded_error" },
    ] as const;
    for (const protocol of cases) {
      for (const [status, type] of [[429, protocol.type429], [529, protocol.type529]] as const) {
        const f = await fixture();
        try {
          const handler = createHandler(f.config, {
            execute: async () => ({ response: new Response("upstream secret", { status, headers: { "retry-after": "17", "x-upstream-secret": "no" } }) }),
            refreshCredentials: async () => null,
          });
          const response = await handler(request(protocol.path, protocol.body));
          const body = await response.json() as { error: { type: string; message: string } };
          expect(response.status).toBe(status);
          expect(body.error.type).toBe(type);
          expect(body.error.message).toBe(`Kiro upstream request failed (${status})`);
          expect(response.headers.get("retry-after")).toBe("17");
          expect(response.headers.get("x-upstream-secret")).toBeNull();
          expect(JSON.stringify(body)).not.toContain("upstream secret");
        } finally { await rm(f.dir, { recursive: true, force: true }); }
      }
    }
  });

  test("maps ordinary upstream 5xx to 502 on both protocols", async () => {
    const cases = [
      ["/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }],
      ["/v1/responses", { model: "gpt-5.6-sol", input: "hi" }],
    ] as const;
    for (const [path, body] of cases) {
      const f = await fixture();
      try {
        const handler = createHandler(f.config, {
          execute: async () => ({ response: new Response("upstream secret", { status: 503, headers: { "retry-after": "17" } }) }),
          refreshCredentials: async () => null,
        });
        const response = await handler(request(path, body));
        const payload = await response.json() as { error: { type: string } };
        expect(response.status).toBe(502);
        expect(payload.error.type).toBe("api_error");
        expect(response.headers.get("retry-after")).toBe("17");
      } finally { await rm(f.dir, { recursive: true, force: true }); }
    }
  });

  // Observed live on claude-opus-5.5: Kiro answers a capacity shortfall with a
  // plain 500 whose body names MODEL_TEMPORARILY_UNAVAILABLE.
  test("reports an upstream capacity shortfall as an overload on both protocols", async () => {
    const upstreamBody = JSON.stringify({
      message: "Encountered unexpectedly high load when processing the request, please try again.",
      reason: "MODEL_TEMPORARILY_UNAVAILABLE",
    });
    const cases = [
      ["/v1/messages", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }, undefined],
      ["/v1/chat/completions", { model: "claude-opus-5", messages: [{ role: "user", content: "hi" }] }, "MODEL_TEMPORARILY_UNAVAILABLE"],
    ] as const;
    for (const [path, body, code] of cases) {
      const f = await fixture();
      try {
        const handler = createHandler(f.config, {
          execute: async () => ({ response: new Response(upstreamBody, { status: 500 }) }),
          refreshCredentials: async () => null,
        });
        const response = await handler(request(path, body));
        const payload = await response.json() as { error: { type: string; message: string; code?: string } };
        expect(response.status).toBe(529);
        expect(payload.error.type).toBe("overloaded_error");
        expect(payload.error.message).toBe("Kiro upstream request failed (500): MODEL_TEMPORARILY_UNAVAILABLE");
        expect(payload.error.code).toBe(code);
      } finally { await rm(f.dir, { recursive: true, force: true }); }
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
