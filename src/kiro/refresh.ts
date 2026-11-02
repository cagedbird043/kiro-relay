import { KIRO_PROVIDER } from "./config.ts";
import type { ExecutorLog, FetchFn, JsonRecord } from "./base.ts";
import { buildExternalIdpRefreshParams, isExternalIdpAuthMethod } from "./externalIdp.ts";
import { AWS_REGION_PATTERN } from "./region.ts";

export type KiroRefreshResult =
  | { ok: true; accessToken: string; refreshToken: string; expiresIn: number }
  | { ok: false; kind: "unrecoverable" | "transient" | "invalid_config"; code: string };

function nonemptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

/** Refresh one token grant; its timeout is independent of any client request. */
export async function refreshKiroToken(
  refreshToken: string | undefined,
  providerSpecificData?: JsonRecord | null,
  log?: ExecutorLog | null,
  fetchImpl: FetchFn = globalThis.fetch,
): Promise<KiroRefreshResult> {
  const fail = (kind: "unrecoverable" | "transient" | "invalid_config", code: string): KiroRefreshResult => {
    log?.error?.("TOKEN_REFRESH", "Kiro token refresh failed", { code });
    return { ok: false, kind, code };
  };
  if (!nonemptyString(refreshToken)) return fail("invalid_config", "missing_refresh_token");
  const data = providerSpecificData ?? {};
  const { authMethod, clientId, clientSecret } = data;
  if (authMethod === "api_key") return fail("invalid_config", "unsupported_auth_method");

  let endpoint: string;
  let body: string | URLSearchParams;
  let externalIdp = false;
  if (isExternalIdpAuthMethod(authMethod)) {
    try {
      const request = buildExternalIdpRefreshParams(refreshToken, data);
      endpoint = request.tokenEndpoint;
      body = request.body;
      externalIdp = true;
    } catch {
      return fail("invalid_config", "invalid_external_idp_config");
    }
  } else if (authMethod !== "imported" &&
    (clientId !== undefined || clientSecret !== undefined || authMethod === "builder-id" || authMethod === "idc")) {
    if (!nonemptyString(clientId) || !nonemptyString(clientSecret)) {
      return fail("invalid_config", "missing_oidc_client");
    }
    const region = data.region ?? "us-east-1";
    if (typeof region !== "string" || !AWS_REGION_PATTERN.test(region)) {
      return fail("invalid_config", "invalid_region");
    }
    endpoint = `https://oidc.${region}.amazonaws.com/token`;
    body = JSON.stringify({ clientId, clientSecret, refreshToken, grantType: "refresh_token" });
  } else {
    endpoint = KIRO_PROVIDER.tokenUrl;
    if (!endpoint) return fail("invalid_config", "missing_token_endpoint");
    body = JSON.stringify({ refreshToken });
  }

  const signal = AbortSignal.timeout(15_000);
  try {
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": externalIdp ? "application/x-www-form-urlencoded" : "application/json",
        Accept: "application/json",
      },
      body,
      signal,
    });
    // Keep the same signal alive through body consumption, not just response headers.
    const text = await response.text();
    signal.throwIfAborted();
    let payload: JsonRecord;
    try {
      payload = record(JSON.parse(text));
    } catch {
      payload = {};
    }
    if (!response.ok) {
      if (response.status === 429 || response.status >= 500) {
        return fail("transient", `http_${response.status}`);
      }
      const code = typeof payload.__type === "string" ? payload.__type : payload.error;
      if (typeof code === "string" && [
        "InvalidGrantException", "ExpiredTokenException", "InvalidGrant", "ExpiredToken", "invalid_grant", "invalid_client",
      ].includes(code)) {
        return fail("unrecoverable", code);
      }
      return fail("transient", `http_${response.status}`);
    }
    const accessToken = externalIdp ? payload.access_token : payload.accessToken;
    const rotatedToken = externalIdp ? payload.refresh_token : payload.refreshToken;
    const expiresIn = (externalIdp ? payload.expires_in : payload.expiresIn) ?? 3600;
    if (!nonemptyString(accessToken) ||
      (rotatedToken !== undefined && !nonemptyString(rotatedToken)) ||
      typeof expiresIn !== "number" || !Number.isFinite(expiresIn) || expiresIn <= 0 ||
      !Number.isFinite(new Date(Date.now() + expiresIn * 1000).getTime())) {
      return fail("transient", "invalid_token_response");
    }
    log?.info?.("TOKEN_REFRESH", "Kiro token refreshed", { expiresIn });
    return { ok: true, accessToken, refreshToken: rotatedToken as string | undefined ?? refreshToken, expiresIn };
  } catch {
    return fail("transient", signal.aborted ? "refresh_timeout" : "refresh_network_error");
  }
}
