import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loginWithKiroOidc, type DeviceLoginDependencies } from "../src/auth-login.ts";

const directories: string[] = [];
afterEach(async () => {
  await Promise.all(directories.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

function json(body: unknown, status = 200): Response {
  return Response.json(body, { status });
}

async function destination(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "kiro-login-"));
  directories.push(directory);
  return join(directory, "private", "account.json");
}

describe("Builder ID device login", () => {
  test("registers an independent client, polls approval, and atomically saves credentials", async () => {
    const outputPath = await destination();
    const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
    const waits: number[] = [];
    const messages: string[] = [];
    let now = 1_000_000;
    const responses = [
      json({ clientId: "vps-client", clientSecret: "vps-secret", clientSecretExpiresAt: 9_999_999 }),
      json({ deviceCode: "device", userCode: "ABCD-EFGH", verificationUri: "https://verify", verificationUriComplete: "https://verify?user_code=ABCD-EFGH", expiresIn: 600, interval: 2 }),
      json({ __type: "AuthorizationPendingException" }, 400),
      json({ __type: "SlowDownException" }, 400),
      json({ accessToken: "access", refreshToken: "refresh", expiresIn: 3600 }),
    ];
    const dependencies: DeviceLoginDependencies = {
      fetch: (async (input, init) => {
        calls.push({ url: String(input), body: JSON.parse(String(init?.body)) as Record<string, unknown> });
        const response = responses.shift();
        if (!response) throw new Error("unexpected fetch");
        return response;
      }) as typeof fetch,
      sleep: async (milliseconds) => {
        waits.push(milliseconds);
        now += milliseconds;
      },
      now: () => now,
      output: (message) => messages.push(message),
    };

    const credentials = await loginWithKiroOidc({ outputPath }, dependencies);
    expect(calls.map((call) => new URL(call.url).pathname)).toEqual([
      "/client/register",
      "/device_authorization",
      "/token",
      "/token",
      "/token",
    ]);
    expect(calls[0].body).toEqual({
      clientName: "kiro-oauth-client",
      clientType: "public",
      scopes: ["codewhisperer:completions", "codewhisperer:analysis", "codewhisperer:conversations"],
      grantTypes: ["urn:ietf:params:oauth:grant-type:device_code", "refresh_token"],
      issuerUrl: "https://identitycenter.amazonaws.com/ssoins-722374e8c3c8e6c6",
    });
    expect(calls[1].body).toEqual({ clientId: "vps-client", clientSecret: "vps-secret", startUrl: "https://view.awsapps.com/start" });
    expect(calls[2].body).toEqual({ clientId: "vps-client", clientSecret: "vps-secret", deviceCode: "device", grantType: "urn:ietf:params:oauth:grant-type:device_code" });
    expect(waits).toEqual([2_000, 2_000, 7_000]);
    expect(messages).toContain("https://verify?user_code=ABCD-EFGH");
    expect(credentials.providerSpecificData).toEqual({
      authMethod: "builder-id",
      provider: "AWS Builder ID",
      clientId: "vps-client",
      clientSecret: "vps-secret",
      clientSecretExpiresAt: 9_999_999,
      region: "us-east-1",
    });
    expect(credentials.expiresAt).toBe(new Date(now + 3_600_000).toISOString());
    expect(JSON.parse(await readFile(outputPath, "utf8"))).toEqual(credentials);
    expect((await stat(outputPath)).mode & 0o777).toBe(0o600);
  });

  test("does not persist credentials when token polling is rejected", async () => {
    const outputPath = await destination();
    const responses = [
      json({ clientId: "client", clientSecret: "secret" }),
      json({ deviceCode: "device", userCode: "CODE", verificationUri: "https://verify", expiresIn: 600, interval: 1 }),
      json({ __type: "AccessDeniedException" }, 400),
    ];
    const dependencies: DeviceLoginDependencies = {
      fetch: (async () => {
        const response = responses.shift();
        if (!response) throw new Error("unexpected fetch");
        return response;
      }) as typeof fetch,
      sleep: async () => undefined,
      now: () => 1_000_000,
      output: () => undefined,
    };

    await expect(loginWithKiroOidc({ outputPath }, dependencies)).rejects.toThrow("AccessDeniedException");
    expect(await Bun.file(outputPath).exists()).toBe(false);
  });

  test("expires rather than polling beyond the device authorization deadline", async () => {
    const outputPath = await destination();
    let now = 10_000;
    let tokenPolls = 0;
    const responses = [
      json({ clientId: "client", clientSecret: "secret" }),
      json({ deviceCode: "device", userCode: "CODE", verificationUri: "https://verify", expiresIn: 2, interval: 2 }),
    ];
    const dependencies: DeviceLoginDependencies = {
      fetch: (async () => {
        const response = responses.shift();
        if (response) return response;
        tokenPolls += 1;
        return json({ __type: "AuthorizationPendingException" }, 400);
      }) as typeof fetch,
      sleep: async (milliseconds) => {
        now += milliseconds;
      },
      now: () => now,
      output: () => undefined,
    };

    await expect(loginWithKiroOidc({ outputPath }, dependencies)).rejects.toThrow("expired before approval");
    expect(tokenPolls).toBe(1);
  });

  test("uses an enterprise start URL without the Builder ID issuer", async () => {
    const outputPath = await destination();
    const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
    const responses = [
      json({ clientId: "idc-client", clientSecret: "idc-secret" }),
      json({ deviceCode: "device", userCode: "CODE", verificationUri: "https://signin", expiresIn: 600, interval: 1 }),
      json({ accessToken: "access", refreshToken: "refresh", expiresIn: 3600 }),
    ];
    const dependencies: DeviceLoginDependencies = {
      fetch: (async (input, init) => {
        calls.push({ url: String(input), body: JSON.parse(String(init?.body)) as Record<string, unknown> });
        const response = responses.shift();
        if (!response) throw new Error("unexpected fetch");
        return response;
      }) as typeof fetch,
      sleep: async () => undefined,
      now: () => 1_000_000,
      output: () => undefined,
    };

    const credentials = await loginWithKiroOidc({
      outputPath,
      startUrl: "https://us-east-1.signin.aws",
      region: "us-east-1",
    }, dependencies);
    expect(calls[0].body.issuerUrl).toBeUndefined();
    expect(calls[1].body.startUrl).toBe("https://us-east-1.signin.aws");
    expect(credentials.providerSpecificData).toEqual({
      authMethod: "idc",
      provider: "Enterprise",
      clientId: "idc-client",
      clientSecret: "idc-secret",
      region: "us-east-1",
    });
  });
});
