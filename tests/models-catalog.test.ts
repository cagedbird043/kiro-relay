import { describe, expect, test } from "bun:test";
import { createHandler } from "../src/server.ts";

type ModelConfig = Parameters<typeof createHandler>[0];

function config(models: ModelConfig["models"]): ModelConfig {
  return {
    listen: "127.0.0.1:0",
    models,
    accounts: [{ credentialFile: "/tmp/unused-credentials.json" }],
  };
}

const emptyExecutor = {
  execute: async () => ({ response: new Response() }),
  refreshCredentials: async () => null,
};

describe("GET /v1/models catalog", () => {
  test("returns both OpenAI data and Codex model envelopes", async () => {
    const response = await createHandler(
      config({
        "claude-opus-5": { contextLength: 1_000_000 },
        "gpt-5.6-sol": { contextLength: 272_000 },
      }),
      emptyExecutor
    )(new Request("http://relay/v1/models?source=codex"));

    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.data).toEqual([
      {
        id: "claude-opus-5",
        object: "model",
        created: 0,
        owned_by: "kiro",
        context_length: 1_000_000,
      },
      {
        id: "gpt-5.6-sol",
        object: "model",
        created: 0,
        owned_by: "kiro",
        context_length: 272_000,
      },
    ]);
    expect(body.models).toHaveLength(1);
    expect(body.models[0]).toMatchObject({
      slug: "gpt-5.6-sol",
      shell_type: "shell_command",
      visibility: "list",
      supported_in_api: true,
      base_instructions: "You are Codex, a coding agent based on GPT-5.",
      use_responses_lite: true,
      context_window: 272_000,
    });
    expect(body.models[0].supported_reasoning_levels).toEqual(
      expect.arrayContaining([{ effort: "high", description: "high" }])
    );
  });

  test("emits a stable catalog ETag regardless of query or insertion order", async () => {
    const first = await createHandler(
      config({
        "claude-opus-5": { contextLength: 1_000_000 },
        "gpt-5.6-sol": { contextLength: 272_000 },
      }),
      emptyExecutor
    )(new Request("http://relay/v1/models"));
    const second = await createHandler(
      config({
        "gpt-5.6-sol": { contextLength: 272_000 },
        "claude-opus-5": { contextLength: 1_000_000 },
      }),
      emptyExecutor
    )(new Request("http://relay/v1/models?limit=100"));

    expect(first.headers.get("etag")).toMatch(/^"[a-f0-9]{64}"$/);
    expect(second.headers.get("etag")).toBe(first.headers.get("etag"));
  });

  test("returns 304 when If-None-Match matches the catalog ETag", async () => {
    const handler = createHandler(
      config({ "gpt-5.6-sol": { contextLength: 272_000 } }),
      emptyExecutor
    );
    const initial = await handler(new Request("http://relay/v1/models"));
    const etag = initial.headers.get("etag");
    expect(etag).toBeTruthy();

    const conditional = await handler(
      new Request("http://relay/v1/models?cached=true", {
        headers: { "if-none-match": etag as string },
      })
    );
    expect(conditional.status).toBe(304);
    expect(conditional.headers.get("etag")).toBe(etag);
    expect(await conditional.text()).toBe("");
  });
});
