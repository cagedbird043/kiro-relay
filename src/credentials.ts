import { randomUUID } from "node:crypto";
import { chmod, open, rename, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { ProviderCredentials } from "./kiro/base.ts";

function credentialsError(message: string): Error {
  return new Error(`Invalid credentials: ${message}`);
}

function validateCredentials(value: unknown): ProviderCredentials {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw credentialsError("JSON value must be an object");
  }
  const input = value as Record<string, unknown>;
  for (const key of ["accessToken", "refreshToken", "apiKey", "expiresAt"]) {
    if (input[key] !== undefined && typeof input[key] !== "string") {
      throw credentialsError(`${key} must be a string`);
    }
  }
  if (
    input.providerSpecificData !== undefined &&
    (!input.providerSpecificData ||
      typeof input.providerSpecificData !== "object" ||
      Array.isArray(input.providerSpecificData))
  ) {
    throw credentialsError("providerSpecificData must be an object");
  }
  return {
    accessToken: input.accessToken as string | undefined,
    refreshToken: input.refreshToken as string | undefined,
    apiKey: input.apiKey as string | undefined,
    expiresAt: input.expiresAt as string | undefined,
    providerSpecificData: input.providerSpecificData as Record<string, unknown> | undefined,
  };
}

/** Read and validate a credentials JSON file. */
export async function loadCredentials(path: string): Promise<ProviderCredentials> {
  if (typeof path !== "string" || path.trim() === "") {
    throw new Error("Unable to read credentials: path must be a non-empty string");
  }
  let text: string;
  try {
    text = await Bun.file(path).text();
  } catch {
    throw new Error(`Unable to read credentials file: ${path}`);
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(`Unable to parse credentials file: ${path}`);
  }
  try {
    return validateCredentials(value);
  } catch (error) {
    if (error instanceof Error) throw new Error(`${error.message} (${path})`);
    throw new Error(`Invalid credentials (${path})`);
  }
}

/** Atomically replace a credentials JSON file with mode 0600. */
export async function saveCredentials(path: string, credentials: ProviderCredentials): Promise<void> {
  if (typeof path !== "string" || path.trim() === "") {
    throw new Error("Unable to save credentials: path must be a non-empty string");
  }
  validateCredentials(credentials);
  let content: string;
  try {
    content = `${JSON.stringify(credentials, null, 2)}\n`;
  } catch {
    throw new Error("Unable to serialize credentials");
  }

  const directory = dirname(path);
  const temporary = join(directory, `.${path.split("/").pop() || "credentials"}.${randomUUID()}.tmp`);
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    handle = await open(temporary, "w", 0o600);
    await handle.writeFile(content, "utf8");
    await chmod(temporary, 0o600);
    await handle.sync();
    await handle.close();
    handle = undefined;
    await rename(temporary, path);
    try {
      const directoryHandle = await open(directory, "r");
      try {
        await directoryHandle.sync();
      } finally {
        await directoryHandle.close();
      }
    } catch {
      // Directory fsync is unavailable on some Bun/Node platforms.
    }
  } catch (error) {
    if (handle) await handle.close().catch(() => undefined);
    await rm(temporary, { force: true }).catch(() => undefined);
    if (error instanceof Error && error.message) {
      throw new Error(`Unable to save credentials file: ${path}`);
    }
    throw new Error(`Unable to save credentials file: ${path}`);
  }
}
