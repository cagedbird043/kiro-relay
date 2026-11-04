import { expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHandler } from "../src/server.ts";
import { fetchKiroUsage } from "../src/kiro/usage.ts";
import type { ProviderCredentials } from "../src/kiro/base.ts";

// 结构取自线上 GetUsageLimits 实测返回，数值已改。
const UPSTREAM = {
  nextDateReset: 1796083200,
  overageConfiguration: { overageStatus: "DISABLED" },
  subscriptionInfo: { subscriptionTitle: "KIRO POWER", type: "Q_DEVELOPER_STANDALONE_POWER" },
  userInfo: { email: "user@example.com", userId: "private-user-id" },
  usageBreakdownList: [{
    resourceType: "CREDIT", unit: "INVOCATIONS",
    currentUsage: 4939, currentUsageWithPrecision: 4939.23,
    usageLimit: 10000, usageLimitWithPrecision: 10000,
    currentOverages: 0, currentOveragesWithPrecision: 0,
    overageCap: 10000, overageCapWithPrecision: 10000,
  }],
};

test("额度查询发往 profileArn 所在区域并丢弃用户身份", async () => {
  let seen: { url: string; headers: Headers } | undefined;
  const result = await fetchKiroUsage(
    { accessToken: "token", providerSpecificData: { authMethod: "idc", profileArn: "arn:aws:codewhisperer:eu-central-1:123:profile/X" } },
    async (input, init) => { seen = { url: String(input), headers: new Headers(init?.headers) }; return Response.json(UPSTREAM); },
  );
  const url = new URL(seen!.url);
  expect(url.host).toBe("q.eu-central-1.amazonaws.com");
  expect(url.searchParams.get("profileArn")).toBe("arn:aws:codewhisperer:eu-central-1:123:profile/X");
  expect(seen!.headers.get("authorization")).toBe("Bearer token");
  expect(result).toEqual({
    ok: true,
    usage: {
      subscription: "KIRO POWER",
      overage_enabled: false,
      resets_at: "2026-12-01T00:00:00.000Z",
      buckets: [{ resource: "CREDIT", unit: "INVOCATIONS", used: 4939.23, limit: 10000, remaining: 5060.77, percent: 49.39, overage_used: 0, overage_cap: 10000 }],
    },
  });
  expect(JSON.stringify(result)).not.toContain("private-user-id");
});

test("上游返回无法识别的结构时报错而不是给出 0 用量", async () => {
  const result = await fetchKiroUsage({ accessToken: "token" }, async () => Response.json({ limits: [] }));
  expect(result).toEqual({ ok: false, status: 502, code: "usage_unrecognized" });
});

test("/v1/usage 需要鉴权，401 时刷新凭据后重查", async () => {
  const dir = await mkdtemp(join(tmpdir(), "kiro-relay-usage-"));
  const credentialFile = join(dir, "credentials.json");
  await writeFile(credentialFile, JSON.stringify({ accessToken: "old", refreshToken: "refresh" }));
  try {
    const tokens: Array<string | undefined> = [];
    const handler = createHandler(
      { listen: "127.0.0.1:0", models: { "claude-opus-5": { contextLength: 1_000_000 } }, accounts: [{ credentialFile }], auth: { keys: ["k"] } },
      {
        execute: async () => { throw new Error("must not call upstream"); },
        refreshCredentials: async () => ({ ok: true, accessToken: "new", refreshToken: "rotated", expiresIn: 3600 }),
      },
      {
        fetchUsage: async (credentials: ProviderCredentials) => {
          tokens.push(credentials.accessToken);
          return credentials.accessToken === "new"
            ? { ok: true, usage: { buckets: [] } }
            : { ok: false, status: 401, code: "http_401" };
        },
      },
    );
    expect((await handler(new Request("http://relay/v1/usage"))).status).toBe(401);
    const response = await handler(new Request("http://relay/v1/usage", { headers: { authorization: "Bearer k" } }));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ buckets: [] });
    expect(tokens).toEqual(["old", "new"]);
    expect(JSON.parse(await Bun.file(credentialFile).text()).accessToken).toBe("new");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
