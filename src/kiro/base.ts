export type JsonRecord = Record<string, unknown>;

/** 可注入的 fetch：只需调用签名，不要求 Bun 全局 fetch 的 preconnect 等附加属性。 */
export type FetchFn = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

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
  /**
   * Context window to scale the synthesized usage estimate with. Supplied by the
   * live model catalog when discovery is active. Omitted -> the static
   * MODEL_CONTEXT_LENGTHS table is consulted, which fails closed on unknown ids.
   */
  contextLength?: number | null;
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
