import { afterEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
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

describe("Kiro 设备登录", () => {
  test("注册独立客户端、轮询批准、原子落盘凭据", async () => {
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
      json({ profiles: [{ arn: "arn:aws:codewhisperer:us-east-1:222:profile/MINE" }] }),
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
      "/",
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
      profileArn: "arn:aws:codewhisperer:us-east-1:222:profile/MINE",
      region: "us-east-1",
    });
    expect(credentials.expiresAt).toBe(new Date(now + 3_600_000).toISOString());
    expect(JSON.parse(await readFile(outputPath, "utf8"))).toEqual(credentials);
    expect((await stat(outputPath)).mode & 0o777).toBe(0o600);
  });

  test("token 轮询被拒时不落盘任何凭据", async () => {
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

  test("超过设备授权有效期即失败，不再继续轮询", async () => {
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

  /**
   * issuerUrl 是 Kiro 各客户端注册时都会带的字段，参考实现也是无条件发送。
   * 这里保持无条件：按登录类型决定发不发，只会注册出一个与所有已验证客户端
   * 都不一样的形状，没有任何好处。
   */
  test("所有登录方式都用同一个 issuer 注册，含企业 IdC", async () => {
    const issuers: Array<unknown> = [];
    const startUrls = [
      "https://view.awsapps.com/start",
      "https://d-90660a0968.awsapps.com/start/",
      "https://example.awsapps.com/start/",
    ];

    for (const startUrl of startUrls) {
      const responses = [
        json({ clientId: "client", clientSecret: "secret" }),
        json({ deviceCode: "device", userCode: "CODE", verificationUri: "https://verify", expiresIn: 600, interval: 1 }),
        json({ accessToken: "access", refreshToken: "refresh", expiresIn: 3600 }),
        json({ profiles: [{ arn: "arn:aws:codewhisperer:us-east-1:222:profile/MINE" }] }),
      ];
      const dependencies: DeviceLoginDependencies = {
        fetch: (async (input, init) => {
          if (String(input).endsWith("/client/register")) {
            issuers.push(JSON.parse(String(init?.body)).issuerUrl);
          }
          const response = responses.shift();
          if (!response) throw new Error("unexpected fetch");
          return response;
        }) as typeof fetch,
        sleep: async () => undefined,
        now: () => 1_000_000,
        output: () => undefined,
      };
      await loginWithKiroOidc({ outputPath: await destination(), startUrl, region: "us-east-1" }, dependencies);
    }

    expect(issuers).toEqual(startUrls.map(() => "https://identitycenter.amazonaws.com/ssoins-722374e8c3c8e6c6"));
  });

  test("提供已有注册时跳过 /client/register，仍走自己的设备授权", async () => {
    const paths: string[] = [];
    const responses = [
      json({ deviceCode: "device", userCode: "CODE", verificationUri: "https://verify", expiresIn: 600, interval: 1 }),
      json({ accessToken: "access", refreshToken: "refresh", expiresIn: 3600 }),
      json({ profiles: [{ arn: "arn:aws:codewhisperer:us-east-1:222:profile/MINE" }] }),
    ];
    const bodies: Array<Record<string, unknown>> = [];
    const dependencies: DeviceLoginDependencies = {
      fetch: (async (input, init) => {
        paths.push(new URL(String(input)).pathname);
        bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
        const response = responses.shift();
        if (!response) throw new Error("unexpected fetch");
        return response;
      }) as typeof fetch,
      sleep: async () => undefined,
      now: () => 1_000_000,
      output: () => undefined,
    };

    const credentials = await loginWithKiroOidc({
      outputPath: await destination(),
      startUrl: "https://example.awsapps.com/start/",
      region: "us-east-1",
      clientId: "approved-client",
      clientSecret: "approved-secret",
    }, dependencies);

    expect(paths).toEqual(["/device_authorization", "/token", "/"]);
    expect(bodies[0]).toEqual({
      clientId: "approved-client",
      clientSecret: "approved-secret",
      startUrl: "https://example.awsapps.com/start/",
    });
    expect(credentials.providerSpecificData?.clientId).toBe("approved-client");
    expect(credentials.accessToken).toBe("access");
  });

  test("clientId 与 clientSecret 必须成对提供", async () => {
    const dependencies: DeviceLoginDependencies = {
      fetch: (async () => {
        throw new Error("must not reach the network");
      }) as typeof fetch,
      sleep: async () => undefined,
      now: () => 1_000_000,
      output: () => undefined,
    };
    await expect(loginWithKiroOidc({
      outputPath: await destination(),
      clientId: "only-id",
    }, dependencies)).rejects.toThrow("must be supplied together");
  });

  test("发现自己的 profileArn，而不是继承别的机器的", async () => {
    const outputPath = await destination();
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, JSON.stringify({
      accessToken: "old",
      refreshToken: "old-refresh",
      providerSpecificData: { profileArn: "arn:aws:codewhisperer:us-east-1:111:profile/OTHERMACHINE" },
    }));
    const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
    const responses = [
      json({ clientId: "idc-client", clientSecret: "idc-secret" }),
      json({ deviceCode: "device", userCode: "CODE", verificationUri: "https://signin", expiresIn: 600, interval: 1 }),
      json({ accessToken: "access", refreshToken: "refresh", expiresIn: 3600 }),
      json({ profiles: [{ arn: "arn:aws:codewhisperer:us-east-1:222:profile/MINE" }] }),
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
      startUrl: "https://example.awsapps.com/start/",
      region: "us-east-1",
    }, dependencies);
    expect(calls[0].body.issuerUrl).toBe("https://identitycenter.amazonaws.com/ssoins-722374e8c3c8e6c6");
    expect(calls[1].body.startUrl).toBe("https://example.awsapps.com/start/");
    expect(calls[3].url).toBe("https://codewhisperer.us-east-1.amazonaws.com/");
    expect(credentials.providerSpecificData).toEqual({
      authMethod: "idc",
      provider: "Enterprise",
      clientId: "idc-client",
      clientSecret: "idc-secret",
      profileArn: "arn:aws:codewhisperer:us-east-1:222:profile/MINE",
      region: "us-east-1",
    });
  });

  test("首个 profile 区域失败时回退下一个，查不到也照常落盘", async () => {
    const outputPath = await destination();
    const probed: string[] = [];
    const responses = [
      json({ clientId: "idc-client", clientSecret: "idc-secret" }),
      json({ deviceCode: "device", userCode: "CODE", verificationUri: "https://signin", expiresIn: 600, interval: 1 }),
      json({ accessToken: "access", refreshToken: "refresh", expiresIn: 3600 }),
      json({ __type: "AccessDeniedException" }, 403),
      json({ profiles: [] }),
    ];
    const dependencies: DeviceLoginDependencies = {
      fetch: (async (input) => {
        const url = String(input);
        if (url.endsWith("amazonaws.com/")) probed.push(url);
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
      startUrl: "https://example.awsapps.com/start/",
      region: "us-east-1",
    }, dependencies);
    expect(probed).toEqual([
      "https://codewhisperer.us-east-1.amazonaws.com/",
      "https://q.eu-central-1.amazonaws.com/",
    ]);
    expect(credentials.providerSpecificData?.profileArn).toBeUndefined();
    expect(await Bun.file(outputPath).exists()).toBe(true);
  });
});
