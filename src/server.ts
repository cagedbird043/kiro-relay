import { loadConfig, type AppConfig } from "./config.ts";
import { createHash } from "node:crypto";
import { loadCredentials, saveCredentials } from "./credentials.ts";
import {
  anthropicToOpenAI,
  openAIStreamToAnthropic,
  collectAnthropicResponse,
  countAnthropicTokens,
} from "./anthropic.ts";
import { chatImageSizeError } from "./imageLimit.ts";
import { buildKiroPayload } from "./kiro/request/openai-to-kiro.ts";
import { kiroModelId, publicModelId } from "./kiro/request/adaptiveThinking.ts";
import { KiroExecutor } from "./kiro/executor.ts";
import type { ProviderCredentials } from "./kiro/base.ts";
import {
  buildKiroResponsesPayload,
  type KiroResponsesRequest,
} from "./responses/request.ts";
import {
  collectResponses,
  responsesStream,
  type ResponsesOptions,
  type ResponseToolMetadata,
} from "./responses/response.ts";
import { createCatalog, type Catalog } from "./catalog.ts";
import { collectChatCompletion } from "./chat.ts";

const jsonHeaders = { "content-type": "application/json; charset=utf-8" };

/**
 * Browser-based clients call this relay directly, so every response carries CORS
 * headers and OPTIONS is answered before the real request is sent. The origin is
 * unrestricted because access is gated by an API key header rather than by an
 * ambient cookie, so a permissive origin grants no authority on its own.
 */
const CORS_HEADERS: Record<string, string> = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers":
    "authorization, content-type, x-api-key, anthropic-version, anthropic-beta, openai-beta, openai-organization, openai-project",
  "access-control-expose-headers": "request-id, etag",
  "access-control-max-age": "86400",
};

function anthropicError(message: string, status = 400, type = "invalid_request_error"): Response {
  return new Response(JSON.stringify({ type: "error", error: { type, message } }), {
    status,
    headers: jsonHeaders,
  });
}

function openAIError(message: string, status = 400, type = "invalid_request_error"): Response {
  return new Response(
    JSON.stringify({ error: { message, type, param: null, code: null } }),
    { status, headers: jsonHeaders }
  );
}

const UPSTREAM_ERROR_HEADERS = ["retry-after"] as const;

const UPSTREAM_REASONS: Record<string, true> = {
  INVALID_MODEL_ID: true, PROMPT_TOO_LONG: true, REQUEST_BODY_INVALID: true, REQUEST_BODY_INVALID_JSON: true,
  THINKING_SIGNATURE_INVALID: true, TOOL_CONFIG_MISSING: true, TOOL_DUPLICATE: true, TOOL_SCHEMA_INVALID: true,
  TOOL_USE_RESULT_MISMATCH: true, CONVERSATION_LIMIT_EXCEEDED: true, MONTHLY_REQUEST_COUNT: true,
  OVERAGE_REQUEST_LIMIT_EXCEEDED: true, INVALID_KMS_GRANT: true, MODEL_TEMPORARILY_UNAVAILABLE: true,
};

async function upstreamReason(response: Response): Promise<string | undefined> {
  try {
    const body: unknown = await response.json();
    if (!isRecord(body)) return;
    const nested = isRecord(body.error) ? body.error : {};
    for (const value of [body.reason, body.code, nested.reason, nested.code]) {
      if (typeof value === "string" && Object.hasOwn(UPSTREAM_REASONS, value)) return value;
    }
  } catch {
    // 非 JSON 错误页不包含可安全透传的协议原因。
  }
}

async function upstreamError(
  response: Response,
  isResponses: boolean,
  context: { requestId: string; model: string; stream: boolean },
): Promise<Response> {
  const upstreamStatus = response.status;
  const code = await upstreamReason(response);
  const isRateLimited = upstreamStatus === 429;
  // Kiro reports a model out of capacity as a plain 500 carrying this reason;
  // surfacing it as an overload lets clients retry instead of seeing a bare 502.
  const isOverloaded = upstreamStatus === 529 || code === "MODEL_TEMPORARILY_UNAVAILABLE";
  const status = isRateLimited
    ? upstreamStatus
    : isOverloaded
      ? 529
      : upstreamStatus >= 500
        ? 502
        : upstreamStatus;
  const type = isRateLimited
    ? "rate_limit_error"
    : isOverloaded
      ? "overloaded_error"
      : status === 401
        ? "authentication_error"
        : "api_error";
  const headers = new Headers(jsonHeaders);
  for (const name of UPSTREAM_ERROR_HEADERS) {
    const value = response.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  headers.set("request-id", context.requestId);
  metadataLog("REQUEST", { ...context, status: upstreamStatus, phase: "failed", category: type, code });
  const message = `Kiro upstream request failed (${upstreamStatus})${code ? `: ${code}` : ""}`;
  return isResponses
    ? new Response(JSON.stringify({ error: { message, type, param: null, code: code ?? null } }), { status, headers })
    : new Response(JSON.stringify({ type: "error", error: { type, message } }), { status, headers });
}

function listenAddress(value: string): { hostname: string; port: number } {
  const last = value.lastIndexOf(":");
  if (last <= 0 || last === value.length - 1) throw new Error("listen must be host:port");
  const hostname = value.slice(0, last).trim();
  const port = Number(value.slice(last + 1));
  if (!hostname || !Number.isInteger(port) || port < 1 || port > 65535) throw new Error("listen must be host:port");
  return { hostname, port };
}

const SAFE_LOG_KEYS: Record<string, true> = {
  model: true,
  stream: true,
  status: true,
  url: true,
  attempt: true,
  phase: true,
  category: true,
  code: true,
  requestId: true,
};

function metadataLog(tag: string, data: Record<string, unknown>) {
  const safe = Object.fromEntries(Object.entries(data).filter(([key, value]) => SAFE_LOG_KEYS[key] && (typeof value === "string" || typeof value === "number" || typeof value === "boolean")));
  console.info(`[${tag}]`, safe);
}

export type ExecutorLike = {
  execute(input: { model: string; body: unknown; stream: boolean; credentials: ProviderCredentials; signal?: AbortSignal; log?: unknown; contextLength?: number }): Promise<{ response: Response }>;
  refreshCredentials(credentials: ProviderCredentials, log?: unknown): Promise<unknown>;
};

type RefreshResult = {
  accessToken?: unknown;
  refreshToken?: unknown;
  expiresIn?: unknown;
  expiresAt?: unknown;
  providerSpecificData?: unknown;
  error?: unknown;
  code?: unknown;
};

function isRefreshResult(value: unknown): value is RefreshResult {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function safeErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

const REFRESH_LEAD_MS = 5 * 60_000;

type CredentialRefreshOutcome =
  | { credentials: ProviderCredentials }
  | { error: "refresh_failed" | "persist_failed"; code?: string };

function credentialExpiry(credentials: ProviderCredentials): number | undefined {
  if (typeof credentials.expiresAt !== "string") return undefined;
  const value = Date.parse(credentials.expiresAt);
  return Number.isFinite(value) ? value : undefined;
}

function expiresWithin(credentials: ProviderCredentials, leadMs: number): boolean {
  const expiry = credentialExpiry(credentials);
  return expiry !== undefined && expiry <= Date.now() + leadMs;
}

function refreshCode(value: unknown): string | undefined {
  if (!isRecord(value)) return undefined;
  for (const key of ["code", "oauthErr", "awsErrorType"]) {
    if (typeof value[key] === "string" && value[key].length <= 80) return value[key];
  }
  return undefined;
}

function normalizeRefreshedCredentials(
  current: ProviderCredentials,
  refreshed: RefreshResult,
): ProviderCredentials | undefined {
  if (refreshed.error || typeof refreshed.accessToken !== "string" || !refreshed.accessToken) {
    return undefined;
  }
  let expiresAt: string | undefined;
  if (typeof refreshed.expiresAt === "string" && Number.isFinite(Date.parse(refreshed.expiresAt))) {
    expiresAt = new Date(refreshed.expiresAt).toISOString();
  } else if (typeof refreshed.expiresIn === "number" && Number.isFinite(refreshed.expiresIn) && refreshed.expiresIn > 0) {
    const expiry = Date.now() + refreshed.expiresIn * 1000;
    if (Number.isFinite(expiry)) expiresAt = new Date(expiry).toISOString();
  }
  const { expiresAt: _staleExpiry, ...base } = current;
  return {
    ...base,
    accessToken: refreshed.accessToken,
    ...(typeof refreshed.refreshToken === "string" && refreshed.refreshToken
      ? { refreshToken: refreshed.refreshToken }
      : {}),
    ...(expiresAt ? { expiresAt } : {}),
    providerSpecificData: isRecord(refreshed.providerSpecificData)
      ? refreshed.providerSpecificData
      : current.providerSpecificData,
  };
}

type CredentialRefreshLog = {
  debug(tag: string, message: string, data?: unknown): void;
  info(tag: string, message: string, data?: unknown): void;
  warn(tag: string, message: string, data?: unknown): void;
  error(tag: string, message: string, data?: unknown): void;
};

function executorLog(model: string): CredentialRefreshLog {
  const write = (phase: string, tag: string, data?: unknown) => {
    const record = isRecord(data) ? data : {};
    const code = refreshCode(record);
    metadataLog(tag, {
      model,
      phase,
      category: "credential_refresh",
      ...(typeof record.status === "number" ? { status: record.status } : {}),
      ...(code ? { code } : {}),
    });
  };
  return {
    debug: (tag: string, _message: string, data?: unknown) => write("debug", tag, data),
    info: (tag: string, _message: string, data?: unknown) => write("info", tag, data),
    warn: (tag: string, _message: string, data?: unknown) => write("warning", tag, data),
    error: (tag: string, _message: string, data?: unknown) => write("failed", tag, data),
  };
}

function responseToolMetadata(payload: unknown): ResponseToolMetadata {
  if (!isRecord(payload) || !isRecord(payload._responseToolMetadata)) return {};
  const result: Record<string, { name: string; namespace?: string; custom: boolean }> = {};
  for (const [toolName, value] of Object.entries(payload._responseToolMetadata)) {
    if (!isRecord(value) || typeof value.name !== "string" || typeof value.custom !== "boolean") {
      continue;
    }
    if (value.namespace !== undefined && typeof value.namespace !== "string") continue;
    result[toolName] = {
      name: value.name,
      custom: value.custom,
      ...(typeof value.namespace === "string" ? { namespace: value.namespace } : {}),
    };
  }
  return result;
}

function responsesOptions(payload: unknown): ResponsesOptions {
  return {
    includeEncryptedContent:
      isRecord(payload) && payload._includeEncryptedContent === true,
  };
}
export function createHandler(
  config: AppConfig,
  executor: ExecutorLike = new KiroExecutor(),
  options: { catalog?: Catalog } = {}
) {
  const { catalog } = options;
  const account = config.accounts[0];
  const refreshFlights = new Map<string, Promise<CredentialRefreshOutcome>>();
  // 刷新令牌可能已轮换；落盘成功前由当前进程持有，后续请求先重试保存。
  let pendingCredentials: ProviderCredentials | undefined;

  const refreshAccount = async (
    model: string,
    log: CredentialRefreshLog,
    rejectedAccessToken?: string,
  ): Promise<CredentialRefreshOutcome> => {
    const path = account.credentialFile;
    const active = refreshFlights.get(path);
    if (active) return active;

    const flight = (async (): Promise<CredentialRefreshOutcome> => {
      let latest: ProviderCredentials;
      try {
        latest = pendingCredentials ?? await loadCredentials(path);
      } catch {
        metadataLog("TOKEN_REFRESH", { model, phase: "failed", category: "credential_reload", code: "credential_reload_failed" });
        return { error: "refresh_failed", code: "credential_reload_failed" };
      }
      if (pendingCredentials) {
        try {
          await saveCredentials(path, latest);
          pendingCredentials = undefined;
        } catch {
          metadataLog("TOKEN_REFRESH", { model, phase: "failed", category: "credential_persistence", code: "persist_failed" });
          return { error: "persist_failed", code: "persist_failed" };
        }
      }


      const anotherRequestRefreshed = rejectedAccessToken !== undefined &&
        latest.accessToken !== rejectedAccessToken && !expiresWithin(latest, 0);
      const proactiveRefreshNoLongerNeeded = rejectedAccessToken === undefined &&
        !expiresWithin(latest, REFRESH_LEAD_MS);
      if (anotherRequestRefreshed || proactiveRefreshNoLongerNeeded) {
        return { credentials: latest };
      }

      metadataLog("TOKEN_REFRESH", { model, phase: "started", category: "credential_refresh" });
      let value: unknown;
      try {
        value = await executor.refreshCredentials(latest, log);
      } catch {
        metadataLog("TOKEN_REFRESH", { model, phase: "failed", category: "credential_refresh", code: "refresh_exception" });
        return { error: "refresh_failed", code: "refresh_exception" };
      }
      if (!isRefreshResult(value)) {
        metadataLog("TOKEN_REFRESH", { model, phase: "failed", category: "credential_refresh", code: "refresh_failed" });
        return { error: "refresh_failed", code: "refresh_failed" };
      }
      const refreshed = normalizeRefreshedCredentials(latest, value);
      if (!refreshed) {
        const code = refreshCode(value) ?? "refresh_rejected";
        metadataLog("TOKEN_REFRESH", { model, phase: "failed", category: "credential_refresh", code });
        return { error: "refresh_failed", code };
      }
      pendingCredentials = refreshed;
      try {
        await saveCredentials(path, refreshed);
        pendingCredentials = undefined;
      } catch {
        metadataLog("TOKEN_REFRESH", { model, phase: "failed", category: "credential_persistence", code: "persist_failed" });
        return { error: "persist_failed", code: "persist_failed" };
      }
      metadataLog("TOKEN_REFRESH", { model, phase: "succeeded", category: "credential_refresh" });
      return { credentials: refreshed };
    })();

    refreshFlights.set(path, flight);
    try {
      return await flight;
    } finally {
      if (refreshFlights.get(path) === flight) refreshFlights.delete(path);
    }
  };

  /**
   * Keys are compared only when configured; an empty list leaves the relay open,
   * which is the historical behaviour and safe only on loopback. Both header
   * spellings are accepted because OpenAI clients send `Authorization: Bearer`
   * while Anthropic clients send `x-api-key`.
   */
  const authorized = (request: Request): boolean => {
    const keys = config.auth?.keys;
    if (!keys || keys.length === 0) return true;
    const bearer = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "").trim();
    const presented = bearer || request.headers.get("x-api-key")?.trim();
    return Boolean(presented) && keys.includes(presented as string);
  };

  const route = async (request: Request): Promise<Response> => {
    const url = new URL(request.url);
    const method = request.method.toUpperCase();
    if (method === "GET" && url.pathname === "/healthz") return Response.json({ status: "ok" });
    if (!authorized(request)) {
      return url.pathname.startsWith("/v1/messages")
        ? anthropicError("Invalid API key", 401, "authentication_error")
        : openAIError("Invalid API key", 401, "invalid_request_error");
    }
    if (method === "GET" && url.pathname === "/v1/models") {
      // With discovery active the catalog is Kiro's own entitlement list; without
      // it the hand-listed config models are served exactly as before.
      const entries: Array<[string, { contextLength: number }]> = catalog
        ? catalog.list().map((entry) => [publicModelId(entry.id), { contextLength: entry.contextLength }])
        : Object.entries(config.models).map(([id, spec]) => [publicModelId(id), spec]);
      const catalogFingerprint = entries
        .map(([id, spec]) => ({ id, contextLength: spec.contextLength }))
        .sort((a, b) => a.id.localeCompare(b.id));
      const etag = `"${createHash("sha256").update(JSON.stringify(catalogFingerprint)).digest("hex")}"`;
      const responseBody = {
        object: "list",
        data: entries.map(([id, spec]) => ({
          id,
          object: "model",
          created: 0,
          owned_by: "kiro",
          context_length: spec.contextLength,
        })),
        models: entries
          .filter(([id]) => id.startsWith("gpt-5.6-"))
          .map(([id, spec], priority) => ({
            slug: id,
            display_name: id,
            description: `Kiro ${id}`,
            default_reasoning_level: "high",
            supported_reasoning_levels: ["minimal", "low", "medium", "high", "xhigh", "max"].map(
              (effort) => ({ effort, description: effort })
            ),
            shell_type: "shell_command",
            visibility: "list",
            supported_in_api: true,
            priority,
            upgrade: null,
            support_verbosity: false,
            default_verbosity: null,
            apply_patch_tool_type: null,
            truncation_policy: { mode: "bytes", limit: spec.contextLength * 4 },
            supports_image_detail_original: false,
            supports_reasoning_summary_parameter: false,
            context_window: spec.contextLength,
            base_instructions: "You are Codex, a coding agent based on GPT-5.",
            experimental_supported_tools: [],
            use_responses_lite: true,
          })),
      };
      const headers = { ...jsonHeaders, etag };
      const ifNoneMatch = request.headers.get("if-none-match");
      if (
        ifNoneMatch?.split(",").some((value) => {
          const token = value.trim();
          return token === "*" || token === etag || token.replace(/^W\//, "") === etag;
        })
      ) {
        return new Response(null, { status: 304, headers: { etag } });
      }
      return new Response(JSON.stringify(responseBody), { headers });
    }

    const isResponses = url.pathname === "/v1/responses";
    const isChat = url.pathname === "/v1/chat/completions";
    const isMessages = url.pathname === "/v1/messages";
    const isCountTokens = url.pathname === "/v1/messages/count_tokens";
    if (method !== "POST" || (!isResponses && !isChat && !isMessages && !isCountTokens)) {
      return anthropicError("Not found", 404, "not_found_error");
    }
    const requestId = crypto.randomUUID();
    let model = "";
    let stream = false;
    const identify = (response: Response) => { response.headers.set("request-id", requestId); return response; };
    const requestError = (message: string, status = 400, type = "invalid_request_error") => {
      metadataLog("REQUEST", { requestId, model, stream, status, phase: "failed", category: type });
      return identify(isResponses || isChat ? openAIError(message, status, type) : anthropicError(message, status, type));
    };

    let body: Record<string, unknown>;
    try {
      const parsed: unknown = await request.json();
      if (!isRecord(parsed)) throw new Error("body must be an object");
      body = parsed;
    } catch {
      return requestError("Request body must be valid JSON");
    }

    model = typeof body.model === "string" ? body.model : "";
    // 官方 Anthropic 写法 claude-opus-5-5 与 Kiro 点号写法 claude-opus-5.5 指向同一上游模型。
    const upstreamModel = kiroModelId(model);
    const modelContextLength = catalog
      ? catalog.contextLengthFor(upstreamModel)
      : config.models[upstreamModel]?.contextLength ?? config.models[publicModelId(upstreamModel)]?.contextLength;
    // Only a live catalog knows which envelope Kiro accepts for this model;
    // with discovery off the translator's static table decides as before.
    const modelThinkingMode = catalog?.thinkingModeFor(upstreamModel);
    if (!model || modelContextLength === undefined) {
      return requestError(`Unknown model: ${model || "(missing)"}`);
    }
    if (isCountTokens) {
      try {
        return Response.json({ input_tokens: countAnthropicTokens(body) });
      } catch (error) {
        return requestError(safeErrorMessage(error, "Unable to count tokens"));
      }
    }
    if (isMessages && !Array.isArray(body.messages)) {
      return requestError("messages must be an array");
    }
    if (isChat) {
      if (!Array.isArray(body.messages)) return requestError("messages must be an array");
      // Kiro cannot honor stop sequences. An empty or null `stop` is the default
      // most web clients send and means "none", so only a populated one is a
      // request the relay has to turn down.
      const stops = body.stop === undefined || body.stop === null ? [] : Array.isArray(body.stop) ? body.stop : [body.stop];
      if (stops.length > 0) return requestError("Unsupported stop on this relay");
      // Kiro has no constrained decoding, so a JSON response_format cannot be guaranteed.
      // Plain `{type: "text"}` is the default and asks for nothing.
      const responseFormat = body.response_format;
      const formatType = responseFormat && typeof responseFormat === "object" ? (responseFormat as Record<string, unknown>).type : undefined;
      if (responseFormat !== undefined && responseFormat !== null && formatType !== "text") {
        return requestError("Unsupported response_format on this relay");
      }
      const oversizedImage = chatImageSizeError(body.messages);
      if (oversizedImage) return requestError(oversizedImage);
    }

    let credentials: ProviderCredentials;
    try {
      credentials = pendingCredentials ?? await loadCredentials(account.credentialFile);
    } catch {
      return requestError("Unable to load configured credentials", 500, "api_error");
    }

    stream = body.stream === true;
    const log = executorLog(model);
    if (pendingCredentials || expiresWithin(credentials, REFRESH_LEAD_MS)) {
      const proactive = await refreshAccount(model, log);
      if ("credentials" in proactive) {
        credentials = proactive.credentials;
      } else if (proactive.error === "persist_failed") {
        return requestError("Unable to persist refreshed credentials", 500, "api_error");
      } else if (expiresWithin(credentials, 0)) {
        return requestError("Kiro credentials expired", 401, "authentication_error");
      }
    }

    let buildPayload: (currentCredentials: ProviderCredentials) => unknown;
    let payload: unknown;
    try {
      if (isResponses) {
        const responsesBody = { ...body, model: upstreamModel } satisfies KiroResponsesRequest;
        buildPayload = (currentCredentials) =>
          buildKiroResponsesPayload(upstreamModel, responsesBody, stream, currentCredentials);
      } else {
        // Chat Completions is already the shape the Kiro translator consumes, so
        // only the Anthropic surface needs converting first.
        const openaiBody = isChat ? body : anthropicToOpenAI(body);
        buildPayload = (currentCredentials) =>
          buildKiroPayload(upstreamModel, openaiBody, stream, currentCredentials, modelThinkingMode);
      }
      payload = buildPayload(credentials);
    } catch (error) {
      return requestError(safeErrorMessage(error, "Unable to translate request"));
    }
    // Only a live catalog may override the executor's static context table, so
    // with discovery off the usage estimate resolves exactly as it did before.
    const execute = () => executor.execute({ model: upstreamModel, body: payload, stream, credentials, signal: request.signal, log, contextLength: catalog ? modelContextLength : undefined });
    let result: { response: Response };
    try {
      result = await execute();
    } catch {
      return requestError("Kiro upstream request failed", 502, "api_error");
    }

    const credentialsExpired = expiresWithin(credentials, 60_000);
    if (result.response.status === 401 || (result.response.status === 403 && credentialsExpired)) {
      await result.response.body?.cancel().catch(() => undefined);
      const rejectedAccessToken = credentials.accessToken;
      const reactive = await refreshAccount(model, log, rejectedAccessToken);
      if (!("credentials" in reactive)) {
        return reactive.error === "persist_failed"
          ? requestError("Unable to persist refreshed credentials", 500, "api_error")
          : requestError("Kiro credentials expired", 401, "authentication_error");
      }
      credentials = reactive.credentials;
      try {
        payload = buildPayload(credentials);
        result = await execute();
      } catch {
        return requestError("Kiro upstream request failed", 502, "api_error");
      }
    }

    if (!result.response.ok) {
      return upstreamError(result.response, isResponses || isChat, { requestId, model, stream });
    }
    metadataLog("REQUEST", { requestId, model, stream, status: result.response.status, phase: "accepted" });
    try {
      if (isResponses) {
        const responseTools = responseToolMetadata(payload);
        const options = responsesOptions(payload);
        return identify(stream
          ? responsesStream(result.response, model, responseTools, options)
          : Response.json(await collectResponses(result.response, model, responseTools, options)));
      }
      if (isChat) {
        // The executor already speaks Chat Completions, so a streaming turn is
        // forwarded verbatim rather than translated and rebuilt.
        return identify(stream
          ? new Response(result.response.body, {
              headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" },
            })
          : await collectChatCompletion(result.response, model));
      }
      return identify(stream
        ? openAIStreamToAnthropic(result.response, model)
        : await collectAnthropicResponse(result.response, model));
    } catch (error) {
      return requestError(
        safeErrorMessage(error, "Unable to translate upstream response"),
        502,
        "api_error"
      );
    }
  };

  return async function handler(request: Request): Promise<Response> {
    if (request.method.toUpperCase() === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }
    const response = await route(request);
    for (const [name, value] of Object.entries(CORS_HEADERS)) response.headers.set(name, value);
    return response;
  };
}

export async function startServer(configPath?: string) {
  const config = await loadConfig(configPath);
  const { hostname, port } = listenAddress(config.listen);
  const catalog = config.discovery?.enabled
    ? createCatalog({
        models: config.models,
        loadCredentials: () => loadCredentials(config.accounts[0].credentialFile),
      })
    : undefined;
  // Warm the catalog so the first request already sees a live snapshot; a failed
  // warm-up is absorbed and simply leaves the static fallback in place.
  void catalog?.refresh();
  const server = Bun.serve({
    hostname,
    port,
    idleTimeout: 255,
    fetch: createHandler(config, new KiroExecutor(), { catalog }),
  });
  return server;
}

if (import.meta.main) {
  startServer().catch((error) => { console.error(error instanceof Error ? error.message : error); process.exit(1); });
}
