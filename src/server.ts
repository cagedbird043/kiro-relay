import { loadConfig, type AppConfig } from "./config.ts";
import { loadCredentials, saveCredentials } from "./credentials.ts";
import {
  anthropicToOpenAI,
  openAIStreamToAnthropic,
  collectAnthropicResponse,
  countAnthropicTokens,
} from "./anthropic.ts";
import { buildKiroPayload } from "./kiro/request/openai-to-kiro.ts";
import { KiroExecutor } from "./kiro/executor.ts";
import type { ProviderCredentials } from "./kiro/base.ts";

const jsonHeaders = { "content-type": "application/json; charset=utf-8" };

function anthropicError(message: string, status = 400, type = "invalid_request_error"): Response {
  return new Response(JSON.stringify({ type: "error", error: { type, message } }), {
    status,
    headers: jsonHeaders,
  });
}

function listenAddress(value: string): { hostname: string; port: number } {
  const last = value.lastIndexOf(":");
  if (last <= 0 || last === value.length - 1) throw new Error("listen must be host:port");
  const hostname = value.slice(0, last).trim();
  const port = Number(value.slice(last + 1));
  if (!hostname || !Number.isInteger(port) || port < 1 || port > 65535) throw new Error("listen must be host:port");
  return { hostname, port };
}

const SAFE_LOG_KEYS: Record<string, true> = { model: true, stream: true, status: true, url: true, attempt: true, phase: true };

function metadataLog(tag: string, data: Record<string, unknown>) {
  const safe = Object.fromEntries(Object.entries(data).filter(([key, value]) => SAFE_LOG_KEYS[key] && (typeof value === "string" || typeof value === "number" || typeof value === "boolean")));
  console.info(`[${tag}]`, safe);
}

export type ExecutorLike = {
  execute(input: { model: string; body: unknown; stream: boolean; credentials: ProviderCredentials; log?: unknown }): Promise<{ response: Response }>;
  refreshCredentials(credentials: ProviderCredentials, log?: unknown): Promise<unknown>;
};

type RefreshResult = { accessToken?: unknown; refreshToken?: unknown; providerSpecificData?: unknown; error?: unknown };

function isRefreshResult(value: unknown): value is RefreshResult {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function safeErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}
export function createHandler(config: AppConfig, executor: ExecutorLike = new KiroExecutor()) {
  return async function handler(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const method = request.method.toUpperCase();
    if (method === "GET" && url.pathname === "/healthz") return Response.json({ status: "ok" });
    if (method === "GET" && url.pathname === "/v1/models") {
      return Response.json({ data: Object.entries(config.models).map(([id, spec]) => ({ id, object: "model", context_length: spec.contextLength })) });
    }
    if (method !== "POST" || !["/v1/messages", "/v1/messages/count_tokens"].includes(url.pathname)) {
      return anthropicError("Not found", 404, "not_found_error");
    }
    let body: Record<string, any>;
    try {
      const parsed = await request.json();
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("body must be an object");
      body = parsed;
    } catch { return anthropicError("Request body must be valid JSON"); }
    const model = typeof body.model === "string" ? body.model : "";
    if (!model || !config.models[model]) return anthropicError(`Unknown model: ${model || "(missing)"}`);
    if (url.pathname === "/v1/messages/count_tokens") {
      try { return Response.json({ input_tokens: countAnthropicTokens(body) }); }
      catch (error) { return anthropicError(error instanceof Error ? error.message : "Unable to count tokens"); }
    }
    if (!Array.isArray(body.messages)) return anthropicError("messages must be an array");
    const account = config.accounts[0];
    let credentials: ProviderCredentials;
    try { credentials = await loadCredentials(account.credentialFile); }
    catch { return anthropicError("Unable to load configured credentials", 500, "api_error"); }
    const stream = body.stream === true;
    let openaiBody: Record<string, any>;
    let payload: unknown;
    try {
      openaiBody = anthropicToOpenAI(body);
      payload = buildKiroPayload(model, openaiBody, stream, credentials);
    } catch (error) {
      return anthropicError(safeErrorMessage(error, "Unable to translate request"));
    }
    const log = {
      info: (tag: string, _message: string, _data?: unknown) => metadataLog(tag, { model }),
      error: (tag: string, _message: string, _data?: unknown) => metadataLog(tag, { model }),
    };
    const execute = () => executor.execute({ model, body: payload, stream, credentials, log });
    let result: { response: Response };
    try { result = await execute(); }
    catch { return anthropicError("Kiro upstream request failed", 502, "api_error"); }
    const credentialExpiry =
      typeof credentials.expiresAt === "string" ? Date.parse(credentials.expiresAt) : Number.NaN;
    const credentialsExpired =
      Number.isFinite(credentialExpiry) && credentialExpiry <= Date.now() + 60_000;
    if (result.response.status === 401 || (result.response.status === 403 && credentialsExpired)) {
      let refreshed: unknown;
      try { refreshed = await executor.refreshCredentials(credentials, log); }
      catch { return anthropicError("Kiro credentials expired", 401, "authentication_error"); }
      if (!isRefreshResult(refreshed) || refreshed.error || typeof refreshed.accessToken !== "string" || !refreshed.accessToken) return anthropicError("Kiro credentials expired", 401, "authentication_error");
      credentials = {
        ...credentials,
        accessToken: refreshed.accessToken,
        ...(typeof refreshed.refreshToken === "string" ? { refreshToken: refreshed.refreshToken } : {}),
        providerSpecificData: refreshed.providerSpecificData && typeof refreshed.providerSpecificData === "object" && !Array.isArray(refreshed.providerSpecificData) ? refreshed.providerSpecificData as ProviderCredentials["providerSpecificData"] : credentials.providerSpecificData,
      };
      try { await saveCredentials(account.credentialFile, credentials); }
      catch { return anthropicError("Unable to persist refreshed credentials", 500, "api_error"); }
      try {
        payload = buildKiroPayload(model, openaiBody, stream, credentials);
        result = await execute();
      } catch { return anthropicError("Kiro upstream request failed", 502, "api_error"); }
    }
    if (!result.response.ok) {
      const status = result.response.status >= 500 ? 502 : result.response.status;
      return anthropicError(`Kiro upstream request failed (${result.response.status})`, status, status === 401 ? "authentication_error" : "api_error");
    }
    metadataLog("REQUEST", { model, stream, status: result.response.status });
    try { return stream ? openAIStreamToAnthropic(result.response, model) : await collectAnthropicResponse(result.response, model); }
    catch (error) { return anthropicError(safeErrorMessage(error, "Unable to translate upstream response"), 502, "api_error"); }
  };
}

export async function startServer(configPath?: string) {
  const config = await loadConfig(configPath);
  const { hostname, port } = listenAddress(config.listen);
  const server = Bun.serve({ hostname, port, idleTimeout: 255, fetch: createHandler(config) });
  return server;
}

if (import.meta.main) {
  startServer().catch((error) => { console.error(error instanceof Error ? error.message : error); process.exit(1); });
}
