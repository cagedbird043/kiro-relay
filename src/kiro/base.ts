export type JsonRecord = Record<string, unknown>;

export type ProviderCredentials = {
  accessToken?: string;
  refreshToken?: string;
  apiKey?: string;
  expiresAt?: string;
  providerSpecificData?: JsonRecord;
};

export type ExecutorLog = {
  debug?: (tag: string, message: string, data?: unknown) => void;
  info?: (tag: string, message: string, data?: unknown) => void;
  warn?: (tag: string, message: string, data?: unknown) => void;
  error?: (tag: string, message: string, data?: unknown) => void;
};

export type ExecuteInput = {
  model: string;
  body: unknown;
  stream: boolean;
  credentials: ProviderCredentials;
  signal?: AbortSignal | null;
  log?: ExecutorLog | null;
  upstreamExtraHeaders?: Record<string, string> | null;
};

export type ProviderConfig = {
  headers: Record<string, string>;
};

export function mergeUpstreamExtraHeaders(
  headers: Record<string, string>,
  extra?: Record<string, string> | null,
): void {
  if (!extra) return;
  for (const [key, value] of Object.entries(extra)) {
    if (key && typeof value === "string") headers[key] = value;
  }
}

export class BaseExecutor {
  readonly provider: string;
  readonly config: ProviderConfig;

  constructor(provider: string, config: ProviderConfig) {
    this.provider = provider;
    this.config = config;
  }
}
