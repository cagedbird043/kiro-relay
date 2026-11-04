/**
 * Kiro 账号额度查询：CodeWhisperer `GetUsageLimits`（Kiro IDE 里“Usage”面板的数据源）。
 *
 *   GET https://q.{region}.amazonaws.com/getUsageLimits?origin=AI_EDITOR&resourceType=AGENTIC_REQUEST&profileArn=...
 *   Authorization: Bearer <accessToken>
 *
 * 接口未公开文档，字段按实测返回尽力解析；响应里的 `userInfo`（邮箱、用户 ID）一律丢弃。
 */
import type { FetchFn, ProviderCredentials } from "./base.ts";
import {
  isExternalIdpAuthMethod,
  KIRO_EXTERNAL_IDP_TOKEN_TYPE_HEADER,
  KIRO_EXTERNAL_IDP_TOKEN_TYPE_VALUE,
} from "./externalIdp.ts";
import { AWS_REGION_PATTERN, resolveKiroRuntimeRegion } from "./region.ts";

export type KiroUsageBucket = {
  resource: string;
  unit?: string;
  used: number;
  limit: number;
  remaining: number;
  percent: number;
  overage_used?: number;
  overage_cap?: number;
  free_trial?: { used: number; limit: number; expires_at?: string };
};

export type KiroUsage = {
  subscription?: string;
  overage_enabled?: boolean;
  resets_at?: string;
  buckets: KiroUsageBucket[];
};

export type KiroUsageResult =
  | { ok: true; usage: KiroUsage }
  | { ok: false; status: number; code: string };

function record(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function number(row: Record<string, unknown>, precise: string, whole: string): number | undefined {
  for (const value of [row[precise], row[whole]]) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

/** Kiro 用秒级 epoch 表示重置时间。 */
function epochSeconds(value: unknown): string | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? new Date(value * 1000).toISOString()
    : undefined;
}

export function parseKiroUsage(body: unknown): KiroUsage | undefined {
  const payload = record(body);
  if (!payload || !Array.isArray(payload.usageBreakdownList)) return undefined;
  const buckets: KiroUsageBucket[] = [];
  for (const item of payload.usageBreakdownList) {
    const row = record(item);
    if (!row) continue;
    const used = number(row, "currentUsageWithPrecision", "currentUsage");
    const limit = number(row, "usageLimitWithPrecision", "usageLimit");
    if (used === undefined || limit === undefined || used < 0 || limit < 0) continue;
    const bucket: KiroUsageBucket = {
      resource: text(row.resourceType) ?? "UNKNOWN",
      ...(text(row.unit) ? { unit: text(row.unit) } : {}),
      used,
      limit,
      remaining: Math.max(0, Math.round((limit - used) * 100) / 100),
      percent: limit > 0 ? Math.round((used / limit) * 10_000) / 100 : 0,
    };
    const overageUsed = number(row, "currentOveragesWithPrecision", "currentOverages");
    const overageCap = number(row, "overageCapWithPrecision", "overageCap");
    if (overageUsed !== undefined) bucket.overage_used = overageUsed;
    if (overageCap !== undefined) bucket.overage_cap = overageCap;
    const trial = record(row.freeTrialInfo);
    if (trial) {
      const trialUsed = number(trial, "currentUsageWithPrecision", "currentUsage");
      const trialLimit = number(trial, "usageLimitWithPrecision", "usageLimit");
      if (trialUsed !== undefined && trialLimit !== undefined) {
        const expiresAt = epochSeconds(trial.freeTrialExpiry);
        bucket.free_trial = { used: trialUsed, limit: trialLimit, ...(expiresAt ? { expires_at: expiresAt } : {}) };
      }
    }
    buckets.push(bucket);
  }
  const subscription = text(record(payload.subscriptionInfo)?.subscriptionTitle);
  const overageStatus = text(record(payload.overageConfiguration)?.overageStatus)?.toUpperCase();
  const resetsAt = epochSeconds(payload.nextDateReset);
  return {
    ...(subscription ? { subscription } : {}),
    ...(overageStatus ? { overage_enabled: overageStatus === "ENABLED" } : {}),
    ...(resetsAt ? { resets_at: resetsAt } : {}),
    buckets,
  };
}

export async function fetchKiroUsage(
  credentials: ProviderCredentials,
  fetchImpl: FetchFn = globalThis.fetch,
): Promise<KiroUsageResult> {
  const data = credentials.providerSpecificData ?? {};
  const authMethod = typeof data.authMethod === "string" ? data.authMethod : undefined;
  const token = authMethod === "api_key" ? credentials.apiKey || credentials.accessToken : credentials.accessToken;
  if (!token) return { ok: false, status: 401, code: "missing_access_token" };
  const region = resolveKiroRuntimeRegion(data as { region?: unknown; profileArn?: unknown });
  if (!AWS_REGION_PATTERN.test(region)) return { ok: false, status: 500, code: "invalid_region" };

  const url = new URL(`https://q.${region}.amazonaws.com/getUsageLimits`);
  url.searchParams.set("origin", "AI_EDITOR");
  url.searchParams.set("resourceType", "AGENTIC_REQUEST");
  if (typeof data.profileArn === "string" && data.profileArn) url.searchParams.set("profileArn", data.profileArn);

  const headers: Record<string, string> = { Accept: "application/json", Authorization: `Bearer ${token}` };
  if (authMethod === "api_key") headers.tokentype = "API_KEY";
  if (isExternalIdpAuthMethod(authMethod)) headers[KIRO_EXTERNAL_IDP_TOKEN_TYPE_HEADER] = KIRO_EXTERNAL_IDP_TOKEN_TYPE_VALUE;

  let response: Response;
  try {
    response = await fetchImpl(url, { headers, signal: AbortSignal.timeout(15_000) });
  } catch {
    return { ok: false, status: 502, code: "usage_fetch_failed" };
  }
  if (!response.ok) {
    await response.body?.cancel().catch(() => undefined);
    return { ok: false, status: response.status, code: `http_${response.status}` };
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return { ok: false, status: 502, code: "usage_invalid_json" };
  }
  const usage = parseKiroUsage(body);
  return usage ? { ok: true, usage } : { ok: false, status: 502, code: "usage_unrecognized" };
}
