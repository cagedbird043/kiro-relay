import { homedir } from "node:os";
import { parse } from "yaml";

export type AppConfig = {
  listen: string;
  models: Record<string, { contextLength: number }>;
  accounts: AccountConfig[];
  /**
   * Live model discovery. Off unless configured, so an untouched config keeps
   * exposing exactly the hand-listed `models` and nothing else.
   */
  discovery?: { enabled: boolean };
};

export type AccountConfig = {
  credentialFile: string;
};

function configError(message: string): Error {
  return new Error(`Invalid config: ${message}`);
}

function expandHome(path: string): string {
  if (path === "~" || path.startsWith("~/")) {
    const home = process.env.HOME || homedir();
    if (home) return path === "~" ? home : `${home}/${path.slice(2)}`;
  }
  return path;
}

/** Load and minimally validate the relay YAML configuration. */
export async function loadConfig(path?: string): Promise<AppConfig> {
  const configPath = path ?? process.env.KIRO_RELAY_CONFIG ?? "config.yaml";
  let text: string;
  try {
    text = await Bun.file(configPath).text();
  } catch {
    throw new Error(`Unable to read config file: ${configPath}`);
  }

  let value: unknown;
  try {
    value = parse(text);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "invalid YAML";
    throw new Error(`Unable to parse config file ${configPath}: ${detail}`);
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw configError("top level must be a mapping");
  }

  const input = value as Record<string, unknown>;
  if (typeof input.listen !== "string" || input.listen.trim() === "") {
    throw configError("listen must be a non-empty string");
  }
  if (!input.models || typeof input.models !== "object" || Array.isArray(input.models)) {
    throw configError("models must be a mapping");
  }
  const models: Record<string, { contextLength: number }> = {};
  for (const [name, model] of Object.entries(input.models as Record<string, unknown>)) {
    if (!name.trim() || !model || typeof model !== "object" || Array.isArray(model)) {
      throw configError(`model ${name || "(unnamed)"} must be a mapping`);
    }
    const contextLength = (model as Record<string, unknown>).contextLength;
    if (typeof contextLength !== "number" || !Number.isFinite(contextLength) || contextLength <= 0) {
      throw configError(`model ${name} contextLength must be a positive number`);
    }
    models[name] = { contextLength };
  }
  if (Object.keys(models).length === 0) throw configError("models must not be empty");

  if (!Array.isArray(input.accounts) || input.accounts.length === 0) {
    throw configError("accounts must be a non-empty list");
  }
  const accounts: AccountConfig[] = input.accounts.map((account, index) => {
    if (!account || typeof account !== "object" || Array.isArray(account)) {
      throw configError(`accounts[${index}] must be a mapping`);
    }
    const credentialFile = (account as Record<string, unknown>).credentialFile;
    if (typeof credentialFile !== "string" || credentialFile.trim() === "") {
      throw configError(`accounts[${index}].credentialFile must be a non-empty string`);
    }
    return { credentialFile: expandHome(credentialFile) };
  });

  let discovery = { enabled: false };
  if (input.discovery !== undefined) {
    if (!input.discovery || typeof input.discovery !== "object" || Array.isArray(input.discovery)) {
      throw configError("discovery must be a mapping");
    }
    const enabled = (input.discovery as Record<string, unknown>).enabled;
    if (enabled !== undefined && typeof enabled !== "boolean") {
      throw configError("discovery.enabled must be a boolean");
    }
    if (typeof enabled === "boolean") discovery = { enabled };
  }

  return { listen: input.listen, models, accounts, discovery };
}
