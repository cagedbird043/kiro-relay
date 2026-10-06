export const KIRO_PROVIDER = {
  headers: {
    "Content-Type": "application/json",
    Accept: "application/vnd.amazon.eventstream",
    "X-Amz-Target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
    "User-Agent": "AWS-SDK-JS/3.0.0 kiro-ide/1.0.0",
    "X-Amz-User-Agent": "aws-sdk-js/3.0.0 kiro-ide/1.0.0",
  },
  tokenUrl: "https://prod.us-east-1.auth.desktop.kiro.dev/refreshToken",
} as const;

// Context windows as reported by Kiro's own ListAvailableModels catalog.
// These are the offline fallback: when live discovery succeeds, the discovered
// value wins (see src/catalog.ts). Unlisted ids still fail closed below.
export const MODEL_CONTEXT_LENGTHS: Record<string, number> = {
  "claude-opus-4.5": 200_000,
  "claude-opus-4.6": 1_000_000,
  "claude-opus-4.7": 1_000_000,
  "claude-opus-4.8": 1_000_000,
  "claude-sonnet-4": 200_000,
  "claude-sonnet-4.5": 200_000,
  "claude-sonnet-4.6": 1_000_000,
  "claude-haiku-4.5": 200_000,
  "deepseek-3.2": 164_000,
  "minimax-m2.1": 196_000,
  "minimax-m2.5": 196_000,
  "glm-5": 200_000,
  "qwen3-coder-next": 256_000,
  "claude-opus-5": 1_000_000,
  "claude-sonnet-5": 1_000_000,
  "claude-opus-5.5": 1_000_000,
  "claude-sonnet-5.5": 1_000_000,
  "gpt-5.6-sol": 272_000,
  "gpt-5.6-terra": 272_000,
  "gpt-5.6-luna": 272_000,
};

export function contextLengthForModel(model: string): number {
  const contextLength = MODEL_CONTEXT_LENGTHS[model];
  if (!contextLength) throw new Error(`Missing contextLength for Kiro model: ${model}`);
  return contextLength;
}
