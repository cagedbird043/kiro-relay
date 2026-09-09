import { mkdir } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { homedir } from "node:os";
import { saveCredentials } from "./credentials.ts";
import type { ProviderCredentials } from "./kiro/base.ts";

const DEFAULT_REGION = "us-east-1";
const BUILDER_ID_START_URL = "https://view.awsapps.com/start";
const DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code";
const SCOPES = [
  "codewhisperer:completions",
  "codewhisperer:analysis",
  "codewhisperer:conversations",
];

interface OidcClientRegistration {
  clientId: string;
  clientSecret: string;
  clientSecretExpiresAt?: number;
}

interface DeviceAuthorization {
  deviceCode: string;
  userCode: string;
  verificationUri: string;
  verificationUriComplete?: string;
  expiresIn: number;
  interval: number;
}

interface DeviceTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

interface OAuthError {
  code: string;
}


export interface KiroOidcLoginOptions {
  outputPath: string;
  startUrl?: string;
  region?: string;
  /**
   * 复用已有的 OIDC 客户端注册，而不是每次新注册；参考实现就是从
   * ~/.aws/sso/cache 里读取已缓存的注册来用。
   *
   * 注册代表的是「应用身份」，不是「登录会话」：本机仍然走自己的设备授权、
   * 拿自己的 access/refresh token，不与任何其他机器共享凭据。
   */
  clientId?: string;
  clientSecret?: string;
}
export interface DeviceLoginDependencies {
  fetch: typeof fetch;
  sleep(milliseconds: number): Promise<void>;
  now(): number;
  output(message: string): void;
}

const DEFAULT_DEPENDENCIES: DeviceLoginDependencies = {
  fetch,
  sleep: (milliseconds) => Bun.sleep(milliseconds),
  now: () => Date.now(),
  output: (message) => console.info(message),
};

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("OIDC response must be a JSON object");
  }
  return value as Record<string, unknown>;
}

function requiredString(input: Record<string, unknown>, key: string): string {
  const value = input[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`OIDC response is missing ${key}`);
  }
  return value;
}

function optionalString(input: Record<string, unknown>, key: string): string | undefined {
  const value = input[key];
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`OIDC response has invalid ${key}`);
  }
  return value;
}

function requiredPositiveNumber(input: Record<string, unknown>, key: string): number {
  const value = input[key];
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`OIDC response has invalid ${key}`);
  }
  return value;
}

async function jsonResponse(response: Response, operation: string): Promise<Record<string, unknown>> {
  let value: unknown;
  try {
    value = await response.json();
  } catch {
    throw new Error(`${operation} returned invalid JSON (HTTP ${response.status})`);
  }
  const body = record(value);
  if (!response.ok) {
    const code = typeof body.__type === "string"
      ? body.__type
      : typeof body.error === "string"
        ? body.error
        : `HTTP ${response.status}`;
    throw new Error(`${operation} failed: ${code}`);
  }
  return body;
}

async function postJson(
  dependencies: DeviceLoginDependencies,
  region: string,
  path: string,
  body: Record<string, unknown>,
  operation: string,
): Promise<Record<string, unknown>> {
  const response = await dependencies.fetch(`https://oidc.${region}.amazonaws.com${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  return jsonResponse(response, operation);
}

/**
 * 注册 OIDC 客户端。issuerUrl 对所有登录方式无条件发送（含企业 IdC），
 * 保持与 Kiro 各客户端一致的注册形状，不按登录类型区分。
 */
async function registerClient(
  dependencies: DeviceLoginDependencies,
  region: string,
): Promise<OidcClientRegistration> {
  const body = await postJson(dependencies, region, "/client/register", {
    clientName: "kiro-oauth-client",
    clientType: "public",
    scopes: SCOPES,
    grantTypes: [DEVICE_GRANT, "refresh_token"],
    issuerUrl: "https://identitycenter.amazonaws.com/ssoins-722374e8c3c8e6c6",
  }, "OIDC client registration");
  const expires = body.clientSecretExpiresAt;
  if (expires !== undefined && (typeof expires !== "number" || !Number.isFinite(expires))) {
    throw new Error("OIDC response has invalid clientSecretExpiresAt");
  }
  return {
    clientId: requiredString(body, "clientId"),
    clientSecret: requiredString(body, "clientSecret"),
    ...(typeof expires === "number" ? { clientSecretExpiresAt: expires } : {}),
  };
}

async function startDeviceAuthorization(
  dependencies: DeviceLoginDependencies,
  region: string,
  startUrl: string,
  client: OidcClientRegistration,
): Promise<DeviceAuthorization> {
  const body = await postJson(dependencies, region, "/device_authorization", {
    clientId: client.clientId,
    clientSecret: client.clientSecret,
    startUrl,
  }, "OIDC device authorization");
  return {
    deviceCode: requiredString(body, "deviceCode"),
    userCode: requiredString(body, "userCode"),
    verificationUri: requiredString(body, "verificationUri"),
    verificationUriComplete: optionalString(body, "verificationUriComplete"),
    expiresIn: requiredPositiveNumber(body, "expiresIn"),
    interval: requiredPositiveNumber(body, "interval"),
  };
}

function oauthError(body: Record<string, unknown>): OAuthError {
  const code = typeof body.__type === "string"
    ? body.__type
    : typeof body.error === "string"
      ? body.error
      : "unknown_error";
  return { code };
}

async function pollForTokens(
  dependencies: DeviceLoginDependencies,
  region: string,
  client: OidcClientRegistration,
  device: DeviceAuthorization,
): Promise<DeviceTokens> {
  const deadline = dependencies.now() + device.expiresIn * 1000;
  let intervalMs = device.interval * 1000;
  while (dependencies.now() < deadline) {
    await dependencies.sleep(intervalMs);
    const response = await dependencies.fetch(`https://oidc.${region}.amazonaws.com/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        clientId: client.clientId,
        clientSecret: client.clientSecret,
        deviceCode: device.deviceCode,
        grantType: DEVICE_GRANT,
      }),
    });
    let value: unknown;
    try {
      value = await response.json();
    } catch {
      throw new Error(`OIDC token polling returned invalid JSON (HTTP ${response.status})`);
    }
    const body = record(value);
    if (response.ok) {
      return {
        accessToken: requiredString(body, "accessToken"),
        refreshToken: requiredString(body, "refreshToken"),
        expiresIn: requiredPositiveNumber(body, "expiresIn"),
      };
    }
    const { code } = oauthError(body);
    if (code === "AuthorizationPendingException" || code === "authorization_pending") continue;
    if (code === "SlowDownException" || code === "slow_down") {
      intervalMs += 5_000;
      continue;
    }
    throw new Error(`OIDC token polling failed: ${code}`);
  }
  throw new Error("OIDC device authorization expired before approval");
}

/**
 * 发现本次登录自己有权使用的 Q Developer profile ARN。
 *
 * 设备码流程不会返回 profileArn，必须用本机自己的 access token 去枚举，
 * 绝不能从别的机器继承（继承来的是别人的身份碎片，会掩盖真实问题）。
 *
 * 无论 IdC 在哪个区域，profile 只托管在 us-east-1 / eu-central-1，
 * 所以先探这两个（EMEA 登录优先探欧洲），最后把 IdC 区域作为兜底追加。
 */
async function discoverProfileArn(
  dependencies: DeviceLoginDependencies,
  accessToken: string,
  idcRegion: string,
): Promise<string | undefined> {
  const regions = /^(eu|af|me|il)-/.test(idcRegion)
    ? ["eu-central-1", "us-east-1"]
    : ["us-east-1", "eu-central-1"];
  if (!regions.includes(idcRegion)) regions.push(idcRegion);

  for (const region of regions) {
    const host = region === "us-east-1"
      ? "https://codewhisperer.us-east-1.amazonaws.com"
      : `https://q.${region}.amazonaws.com`;
    let body: unknown;
    try {
      const response = await dependencies.fetch(`${host}/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-amz-json-1.0",
          Accept: "application/json",
          "x-amz-target": "AmazonCodeWhispererService.ListAvailableProfiles",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ maxResults: 10 }),
      });
      if (!response.ok) continue;
      body = await response.json();
    } catch {
      continue;
    }
    if (!body || typeof body !== "object" || !("profiles" in body)) continue;
    const listed = body.profiles;
    if (!Array.isArray(listed)) continue;
    const arns: string[] = [];
    for (const profile of listed) {
      if (!profile || typeof profile !== "object" || !("arn" in profile)) continue;
      const arn = profile.arn;
      if (typeof arn === "string" && arn.length > 0) arns.push(arn);
    }
    const matched = arns.find((arn) => arn.split(":")[3] === region) ?? arns[0];
    if (matched) return matched;
  }
  return undefined;
}

export async function loginWithKiroOidc(
  options: KiroOidcLoginOptions,
  dependencies: DeviceLoginDependencies = DEFAULT_DEPENDENCIES,
): Promise<ProviderCredentials> {
  const region = options.region ?? DEFAULT_REGION;
  if (!/^[a-z]{2}(?:-gov)?-[a-z]+-\d$/.test(region)) {
    throw new Error("AWS region must have a canonical region name such as us-east-1");
  }
  const startUrl = options.startUrl ?? BUILDER_ID_START_URL;
  let parsedStartUrl: URL;
  try {
    parsedStartUrl = new URL(startUrl);
  } catch {
    throw new Error("IDC start URL must be a valid HTTPS URL");
  }
  if (parsedStartUrl.protocol !== "https:") {
    throw new Error("IDC start URL must be a valid HTTPS URL");
  }
  const enterprise = startUrl !== BUILDER_ID_START_URL;
  if ((options.clientId === undefined) !== (options.clientSecret === undefined)) {
    throw new Error("clientId and clientSecret must be supplied together");
  }
  const client = options.clientId && options.clientSecret
    ? { clientId: options.clientId, clientSecret: options.clientSecret }
    : await registerClient(dependencies, region);
  const device = await startDeviceAuthorization(dependencies, region, startUrl, client);
  dependencies.output("Authorize this VPS in your browser:");
  dependencies.output(device.verificationUriComplete ?? device.verificationUri);
  dependencies.output(`Device code: ${device.userCode}`);
  const tokens = await pollForTokens(dependencies, region, client, device);
  const profileArn = await discoverProfileArn(dependencies, tokens.accessToken, region);
  const credentials: ProviderCredentials = {
    accessToken: tokens.accessToken,
    refreshToken: tokens.refreshToken,
    expiresAt: new Date(dependencies.now() + tokens.expiresIn * 1000).toISOString(),
    providerSpecificData: {
      authMethod: enterprise ? "idc" : "builder-id",
      provider: enterprise ? "Enterprise" : "AWS Builder ID",
      clientId: client.clientId,
      clientSecret: client.clientSecret,
      ...(client.clientSecretExpiresAt !== undefined
        ? { clientSecretExpiresAt: client.clientSecretExpiresAt }
        : {}),
      ...(profileArn ? { profileArn } : {}),
      region,
    },
  };
  await mkdir(dirname(options.outputPath), { recursive: true, mode: 0o700 });
  await saveCredentials(options.outputPath, credentials);
  dependencies.output(`Credentials saved to ${options.outputPath}`);
  return credentials;
}

function commandOptions(arguments_: string[]): KiroOidcLoginOptions {
  let outputPath = resolve(homedir(), ".config/kiro-relay/account.json");
  let startUrl: string | undefined;
  let region: string | undefined;
  let clientId: string | undefined;
  let clientSecret: string | undefined;
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index];
    const value = arguments_[index + 1];
    if (!value) throw new Error("Every login option requires a value");
    if (flag === "--output") outputPath = resolve(value);
    else if (flag === "--start-url") startUrl = value;
    else if (flag === "--region") region = value;
    else if (flag === "--client-id") clientId = value;
    else if (flag === "--client-secret") clientSecret = value;
    else if (flag === "--client-registration") {
      const cached = JSON.parse(readFileSync(resolve(value), "utf8")) as unknown;
      if (!cached || typeof cached !== "object" || !("clientId" in cached) || !("clientSecret" in cached)) {
        throw new Error("Client registration file must contain clientId and clientSecret");
      }
      const { clientId: cachedId, clientSecret: cachedSecret } = cached;
      if (typeof cachedId !== "string" || typeof cachedSecret !== "string") {
        throw new Error("Client registration file must contain clientId and clientSecret");
      }
      clientId = cachedId;
      clientSecret = cachedSecret;
    } else throw new Error(`Unknown login option: ${flag}`);
  }
  return {
    outputPath,
    ...(startUrl ? { startUrl } : {}),
    ...(region ? { region } : {}),
    ...(clientId ? { clientId } : {}),
    ...(clientSecret ? { clientSecret } : {}),
  };
}

if (import.meta.main) {
  try {
    await loginWithKiroOidc(commandOptions(process.argv.slice(2)));
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Kiro login failed");
    process.exitCode = 1;
  }
}
