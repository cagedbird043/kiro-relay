import { mkdir } from "node:fs/promises";
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

async function registerClient(
  dependencies: DeviceLoginDependencies,
  region: string,
  enterprise: boolean,
): Promise<OidcClientRegistration> {
  const body = await postJson(dependencies, region, "/client/register", {
    clientName: "kiro-oauth-client",
    clientType: "public",
    scopes: SCOPES,
    grantTypes: [DEVICE_GRANT, "refresh_token"],
    ...(!enterprise ? { issuerUrl: "https://identitycenter.amazonaws.com/ssoins-722374e8c3c8e6c6" } : {}),
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
  const client = await registerClient(dependencies, region, enterprise);
  const device = await startDeviceAuthorization(dependencies, region, startUrl, client);
  dependencies.output("Authorize this VPS in your browser:");
  dependencies.output(device.verificationUriComplete ?? device.verificationUri);
  dependencies.output(`Device code: ${device.userCode}`);
  const tokens = await pollForTokens(dependencies, region, client, device);
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
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index];
    const value = arguments_[index + 1];
    if (!value) throw new Error("Every login option requires a value");
    if (flag === "--output") outputPath = resolve(value);
    else if (flag === "--start-url") startUrl = value;
    else if (flag === "--region") region = value;
    else throw new Error(`Unknown login option: ${flag}`);
  }
  return { outputPath, ...(startUrl ? { startUrl } : {}), ...(region ? { region } : {}) };
}

if (import.meta.main) {
  try {
    await loginWithKiroOidc(commandOptions(process.argv.slice(2)));
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Kiro login failed");
    process.exitCode = 1;
  }
}
