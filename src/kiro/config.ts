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

export const MODEL_CONTEXT_LENGTHS: Record<string, number> = {
  // Explicit context windows for models exposed by Kiro.
  "claude-opus-4.7": 200_000,
  "claude-opus-4.8": 200_000,
  "claude-sonnet-4": 200_000,
  "claude-sonnet-4.5": 200_000,
  "claude-sonnet-4.6": 200_000,
  "claude-haiku-4.5": 200_000,
  "deepseek-3.2": 200_000,
  "minimax-m2.5": 200_000,
  "glm-5": 200_000,
  "qwen3-coder-next": 200_000,
  "claude-opus-5": 1_000_000,
  "claude-sonnet-5": 1_000_000,
  "gpt-5.6-sol": 272_000,
  "gpt-5.6-terra": 272_000,
  "gpt-5.6-luna": 272_000,
};

export function contextLengthForModel(model: string): number {
  const contextLength = MODEL_CONTEXT_LENGTHS[model];
  if (!contextLength) throw new Error(`Missing contextLength for Kiro model: ${model}`);
  return contextLength;
}
