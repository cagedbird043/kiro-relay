import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { homedir } from "node:os";
import { saveCredentials } from "./credentials.ts";
import type { ProviderCredentials } from "./kiro/base.ts";

const REGION = "us-east-1";
const OIDC_BASE_URL = `https://oidc.${REGION}.amazonaws.com`;
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
  path: string,
  body: Record<string, unknown>,
  operation: string,
): Promise<Record<string, unknown>> {
  const response = await dependencies.fetch(`${OIDC_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  return jsonResponse(response, operation);
}

async function registerClient(dependencies: DeviceLoginDependencies): Promise<OidcClientRegistration> {
  const body = await postJson(dependencies, "/client/register", {
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
  client: OidcClientRegistration,
): Promise<DeviceAuthorization> {
  const body = await postJson(dependencies, "/device_authorization", {
    clientId: client.clientId,
    clientSecret: client.clientSecret,
    startUrl: "https://view.awsapps.com/start",
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
  client: OidcClientRegistration,
  device: DeviceAuthorization,
): Promise<DeviceTokens> {
  const deadline = dependencies.now() + device.expiresIn * 1000;
  let intervalMs = device.interval * 1000;
  while (dependencies.now() < deadline) {
    await dependencies.sleep(intervalMs);
    const response = await dependencies.fetch(`${OIDC_BASE_URL}/token`, {
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

export async function loginWithBuilderId(
  outputPath: string,
  dependencies: DeviceLoginDependencies = DEFAULT_DEPENDENCIES,
): Promise<ProviderCredentials> {
  const client = await registerClient(dependencies);
  const device = await startDeviceAuthorization(dependencies, client);
  dependencies.output("Authorize this VPS in your browser:");
  dependencies.output(device.verificationUriComplete ?? device.verificationUri);
  dependencies.output(`Device code: ${device.userCode}`);
  const tokens = await pollForTokens(dependencies, client, device);
  const credentials: ProviderCredentials = {
    accessToken: tokens.accessToken,
    refreshToken: tokens.refreshToken,
    expiresAt: new Date(dependencies.now() + tokens.expiresIn * 1000).toISOString(),
    providerSpecificData: {
      authMethod: "builder-id",
      provider: "AWS Builder ID",
      clientId: client.clientId,
      clientSecret: client.clientSecret,
      ...(client.clientSecretExpiresAt !== undefined
        ? { clientSecretExpiresAt: client.clientSecretExpiresAt }
        : {}),
      region: REGION,
    },
  };
  await mkdir(dirname(outputPath), { recursive: true, mode: 0o700 });
  await saveCredentials(outputPath, credentials);
  dependencies.output(`Credentials saved to ${outputPath}`);
  return credentials;
}

function outputArgument(arguments_: string[]): string {
  if (arguments_.length === 0) return resolve(homedir(), ".config/kiro-relay/account.json");
  if (arguments_.length === 2 && arguments_[0] === "--output" && arguments_[1]) {
    return resolve(arguments_[1]);
  }
  throw new Error("Usage: bun run auth:login [--output /path/to/account.json]");
}

if (import.meta.main) {
  try {
    await loginWithBuilderId(outputArgument(process.argv.slice(2)));
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Kiro login failed");
    process.exitCode = 1;
  }
}
