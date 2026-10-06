import { beforeEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createCatalog } from "../src/catalog.ts";
import { createHandler } from "../src/server.ts";
import { clearKiroModelCache, fetchKiroAvailableModels } from "../src/kiro/models.ts";

/**
 * Shape of a real `ListAvailableModels` response, trimmed to the cases that
 * matter: the `auto` router, a thinking-capable Claude (whose
 * `additionalModelRequestFieldsSchema` declares the adaptive envelope and whose
 * expansion emits a synthetic `-thinking` alias), and text-only models that
 * declare no schema at all.
 */
const ADAPTIVE_SCHEMA = {
  type: "object",
  properties: {
    thinking: {
      type: "object",
      properties: { type: { type: "string", enum: ["adaptive", "disabled"] } },
      required: ["type"],
    },
    output_config: { type: "object", properties: { effort: { type: "string" } } },
  },
};

const UPSTREAM_PAYLOAD = {
  models: [
    { modelId: "auto", modelName: "Auto", rateMultiplier: 1.0, tokenLimits: { maxInputTokens: 1_000_000 } },
    { modelId: "claude-opus-5", modelName: "Claude Opus 5", rateMultiplier: 2.2, tokenLimits: { maxInputTokens: 1_000_000 }, additionalModelRequestFieldsSchema: ADAPTIVE_SCHEMA },
    { modelId: "deepseek-3.2", modelName: "Deepseek v3.2", rateMultiplier: 0.25, tokenLimits: { maxInputTokens: 164_000 } },
    { modelId: "qwen3-coder-next", modelName: "Qwen3 Coder Next", rateMultiplier: 0.05, tokenLimits: { maxInputTokens: 256_000 } },
  ],
};

const CREDENTIALS = { accessToken: "token", providerSpecificData: { profileArn: "arn:test" } };

/**
 * A refresh pass always calls `loadCredentials` before it suspends on its first
 * await, so the counter advances synchronously with the decision to refresh.
 * That lets the TTL assertions observe the real signal instead of waiting on a
 * timer for something that may never happen.
 */
function harness(responder: () => Response, credentials: () => unknown = () => CREDENTIALS) {
  let refreshes = 0;
  const loadCredentials = async () => {
    refreshes++;
    return credentials() as never;
  };
  const fetchModels: typeof fetchKiroAvailableModels = (options) =>
    fetchKiroAvailableModels({ ...options, fetchImpl: async () => responder() });
  return { loadCredentials, fetchModels, refreshes: () => refreshes };
}

beforeEach(() => {
  clearKiroModelCache();
});

describe("live model catalog", () => {
  test("exposes the ids Kiro routes on, dropping auto and synthetic -thinking aliases", async () => {
    const { loadCredentials, fetchModels } = harness(() => Response.json(UPSTREAM_PAYLOAD));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });

    await catalog.refresh();

    expect(catalog.list()).toEqual([
      { id: "claude-opus-5", contextLength: 1_000_000, thinkingMode: "adaptive" },
      { id: "deepseek-3.2", contextLength: 164_000, thinkingMode: "none" },
      { id: "qwen3-coder-next", contextLength: 256_000, thinkingMode: "none" },
    ]);
    expect(catalog.has("auto")).toBe(false);
    expect(catalog.has("claude-opus-5-thinking")).toBe(false);
  });

  // Kiro declares the envelope it accepts per model. Reading it off the wire is
  // what lets a model that gains thinking upstream start receiving it without a
  // relay release, and keeps a model that never had it from being sent a field
  // Kiro answers with a 400.
  test("derives each model's thinking envelope from its advertised schema", async () => {
    const payload = {
      models: [
        { modelId: "claude-opus-4.6", tokenLimits: { maxInputTokens: 1_000_000 }, additionalModelRequestFieldsSchema: ADAPTIVE_SCHEMA },
        {
          modelId: "gpt-5.6-sol",
          tokenLimits: { maxInputTokens: 272_000 },
          additionalModelRequestFieldsSchema: {
            type: "object",
            properties: { reasoning: { type: "object", properties: { effort: { type: "string" } } } },
          },
        },
        { modelId: "glm-5", tokenLimits: { maxInputTokens: 200_000 }, additionalModelRequestFieldsSchema: null },
      ],
    };
    const { loadCredentials, fetchModels } = harness(() => Response.json(payload));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });

    await catalog.refresh();

    expect(catalog.thinkingModeFor("claude-opus-4.6")).toBe("adaptive");
    expect(catalog.thinkingModeFor("gpt-5.6-sol")).toBe("native");
    expect(catalog.thinkingModeFor("glm-5")).toBe("none");
  });

  // Opus/Sonnet 5.5 preview declare `thinking.type` without `disabled`; sending
  // it is a 400 REQUEST_BODY_INVALID upstream, so the catalog has to tell these
  // apart from Claude models that can switch thinking off.
  test("marks a Claude model whose thinking cannot be disabled as adaptive_only", async () => {
    const thinkingSchema = (types: string[]) => ({
      type: "object",
      properties: {
        thinking: { type: "object", properties: { type: { type: "string", enum: types } }, required: ["type"] },
        output_config: { type: "object", properties: { effort: { type: "string" } } },
      },
    });
    const payload = {
      models: [
        { modelId: "claude-opus-5.5", tokenLimits: { maxInputTokens: 1_000_000 }, additionalModelRequestFieldsSchema: thinkingSchema(["adaptive"]) },
        { modelId: "claude-sonnet-5.5", tokenLimits: { maxInputTokens: 1_000_000 }, additionalModelRequestFieldsSchema: thinkingSchema(["adaptive", "between_tools"]) },
        { modelId: "claude-sonnet-5", tokenLimits: { maxInputTokens: 1_000_000 }, additionalModelRequestFieldsSchema: thinkingSchema(["adaptive", "disabled"]) },
      ],
    };
    const { loadCredentials, fetchModels } = harness(() => Response.json(payload));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });

    await catalog.refresh();

    expect(catalog.thinkingModeFor("claude-opus-5.5")).toBe("adaptive_only");
    expect(catalog.thinkingModeFor("claude-sonnet-5.5")).toBe("adaptive_only");
    expect(catalog.thinkingModeFor("claude-sonnet-5")).toBe("adaptive");
  });

  test("falls back to the static envelope table when discovery never succeeds", async () => {
    const { loadCredentials, fetchModels } = harness(() => new Response("nope", { status: 500 }));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });

    await catalog.refresh();

    expect(catalog.thinkingModeFor("claude-opus-4.6")).toBe("adaptive");
    expect(catalog.thinkingModeFor("gpt-5.6-sol")).toBe("native");
    expect(catalog.thinkingModeFor("glm-5")).toBe("none");
  });

  test("reports the window Kiro advertises, not the hand-maintained default", async () => {
    const { loadCredentials, fetchModels } = harness(() => Response.json(UPSTREAM_PAYLOAD));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });

    await catalog.refresh();

    expect(catalog.contextLengthFor("deepseek-3.2")).toBe(164_000);
    expect(catalog.contextLengthFor("qwen3-coder-next")).toBe(256_000);
  });

  test("serves the full static table when no credentials are available", async () => {
    const { loadCredentials, fetchModels } = harness(() => Response.json(UPSTREAM_PAYLOAD), () => null);
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });

    await catalog.refresh();

    // Discovery without a token reports "fallback", so a request can still reach
    // every statically known model rather than facing an empty catalog.
    expect(catalog.has("claude-opus-4.6")).toBe(true);
    expect(catalog.contextLengthFor("claude-opus-4.6")).toBe(1_000_000);
    expect(catalog.contextLengthFor("deepseek-3.2")).toBe(164_000);
    expect(catalog.has("auto")).toBe(false);
  });

  test("keeps serving the last good snapshot once discovery starts failing", async () => {
    let healthy = true;
    const { loadCredentials, fetchModels } = harness(() =>
      healthy ? Response.json(UPSTREAM_PAYLOAD) : new Response("upstream down", { status: 503 })
    );
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });

    await catalog.refresh();
    const discovered = catalog.list();

    healthy = false;
    clearKiroModelCache();
    await catalog.refresh();

    // An upstream blip must not make models disappear mid-session.
    expect(catalog.list()).toEqual(discovered);
    expect(catalog.has("qwen3-coder-next")).toBe(true);
  });

  test("survives a malformed upstream payload without throwing", async () => {
    const { loadCredentials, fetchModels } = harness(() => new Response("<html>nope</html>", { status: 200 }));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });

    await catalog.refresh();

    expect(catalog.has("claude-opus-5")).toBe(true);
    expect(catalog.has("auto")).toBe(false);
  });

  test("survives a credential read that throws", async () => {
    const { fetchModels } = harness(() => Response.json(UPSTREAM_PAYLOAD));
    const catalog = createCatalog({
      models: {},
      loadCredentials: async () => {
        throw new Error("credential file unreadable");
      },
      fetchModels,
    });

    await catalog.refresh();

    expect(catalog.has("claude-opus-5")).toBe(true);
  });

  test("config overrides a discovered window but cannot add ids the account lacks", async () => {
    const { loadCredentials, fetchModels } = harness(() => Response.json(UPSTREAM_PAYLOAD));
    const catalog = createCatalog({
      models: {
        "deepseek-3.2": { contextLength: 111_000 },
        "claude-sonnet-4": { contextLength: 200_000 },
      },
      loadCredentials,
      fetchModels,
    });

    await catalog.refresh();

    expect(catalog.contextLengthFor("deepseek-3.2")).toBe(111_000);
    expect(catalog.has("claude-sonnet-4")).toBe(false);
  });

  test("reuses the snapshot within the TTL instead of re-querying upstream", async () => {
    const { loadCredentials, fetchModels, refreshes } = harness(() => Response.json(UPSTREAM_PAYLOAD));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels, ttlMs: 60_000 });

    await catalog.refresh();
    catalog.list();
    catalog.has("claude-opus-5");
    catalog.contextLengthFor("claude-opus-5");

    expect(refreshes()).toBe(1);
  });

  test("refreshes again once the snapshot goes stale", async () => {
    const { loadCredentials, fetchModels, refreshes } = harness(() => Response.json(UPSTREAM_PAYLOAD));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels, ttlMs: 0 });

    await catalog.refresh();
    catalog.list();

    expect(refreshes()).toBe(2);
  });
});

describe("catalog-backed server routes", () => {
  async function servedByCatalog() {
    const directory = await mkdtemp(join(tmpdir(), "kiro-relay-catalog-"));
    const credentialFile = join(directory, "credentials.json");
    await writeFile(credentialFile, JSON.stringify({ accessToken: "token", refreshToken: "refresh" }));

    const { loadCredentials, fetchModels } = harness(() => Response.json(UPSTREAM_PAYLOAD));
    const catalog = createCatalog({ models: {}, loadCredentials, fetchModels });
    await catalog.refresh();

    const seen: Array<{ model: string; contextLength?: number }> = [];
    const handler = createHandler(
      {
        listen: "127.0.0.1:0",
        // Deliberately narrow: the old behaviour gated on exactly this list.
        models: { "claude-opus-5": { contextLength: 1_000_000 } },
        accounts: [{ credentialFile }],
      },
      {
        execute: async (input) => {
          seen.push({ model: input.model, contextLength: input.contextLength });
          return {
            response: new Response(
              JSON.stringify({ choices: [{ message: { content: "hello" }, finish_reason: "stop" }] }),
              { headers: { "content-type": "application/json" } }
            ),
          };
        },
        refreshCredentials: async () => null,
      },
      { catalog }
    );
    return { directory, handler, seen };
  }

  test("serves the discovered catalog instead of the hand-listed models", async () => {
    const { directory, handler } = await servedByCatalog();
    try {
      const response = await handler(new Request("http://relay/v1/models"));
      const body = (await response.json()) as { data: Array<{ id: string; context_length: number }> };

      expect(body.data).toEqual([
        { id: "claude-opus-5", object: "model", created: 0, owned_by: "kiro", context_length: 1_000_000 },
        { id: "deepseek-3.2", object: "model", created: 0, owned_by: "kiro", context_length: 164_000 },
        { id: "qwen3-coder-next", object: "model", created: 0, owned_by: "kiro", context_length: 256_000 },
      ]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("accepts a discovered model that config never listed", async () => {
    const { directory, handler, seen } = await servedByCatalog();
    try {
      const response = await handler(
        new Request("http://relay/v1/messages", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model: "qwen3-coder-next", messages: [{ role: "user", content: "hi" }] }),
        })
      );

      expect(response.status).toBe(200);
      // The window travels with the request, so usage is scaled by Kiro's own
      // figure rather than the executor's static default.
      expect(seen).toEqual([{ model: "qwen3-coder-next", contextLength: 256_000 }]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("still rejects an id the account has no entitlement for", async () => {
    const { directory, handler, seen } = await servedByCatalog();
    try {
      const response = await handler(
        new Request("http://relay/v1/messages", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model: "claude-sonnet-4", messages: [{ role: "user", content: "hi" }] }),
        })
      );

      expect(response.status).toBe(400);
      expect(seen).toEqual([]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
