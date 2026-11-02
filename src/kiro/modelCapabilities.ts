/** Offline context and reasoning capabilities; image support defaults to enabled. */
export type KiroModelCapabilities = {
  contextLength: number;
  supportsAdaptiveReasoning: boolean;
  supportsNativeReasoning: boolean;
};

export const KIRO_MODEL_CAPABILITIES: Readonly<Record<string, KiroModelCapabilities>> = {
  "claude-opus-4.5": { contextLength: 200_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-opus-4.6": { contextLength: 1_000_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-opus-4.7": { contextLength: 1_000_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-opus-4.8": { contextLength: 1_000_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-sonnet-4": { contextLength: 200_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-sonnet-4.5": { contextLength: 200_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-sonnet-4.6": { contextLength: 1_000_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-haiku-4.5": { contextLength: 200_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "deepseek-3.2": { contextLength: 164_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "minimax-m2.1": { contextLength: 196_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "minimax-m2.5": { contextLength: 196_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "glm-5": { contextLength: 200_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "qwen3-coder-next": { contextLength: 256_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-opus-5": { contextLength: 1_000_000, supportsAdaptiveReasoning: true, supportsNativeReasoning: false },
  "claude-sonnet-5": { contextLength: 1_000_000, supportsAdaptiveReasoning: true, supportsNativeReasoning: false },
  "claude-opus-5.5": { contextLength: 1_000_000, supportsAdaptiveReasoning: true, supportsNativeReasoning: false },
  "claude-sonnet-5.5": { contextLength: 1_000_000, supportsAdaptiveReasoning: true, supportsNativeReasoning: false },
  "gpt-5.6-sol": { contextLength: 272_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: true },
  "gpt-5.6-terra": { contextLength: 272_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: true },
  "gpt-5.6-luna": { contextLength: 272_000, supportsAdaptiveReasoning: false, supportsNativeReasoning: true },
};

export function getKiroModelCapabilities(model: string): KiroModelCapabilities | undefined {
  return KIRO_MODEL_CAPABILITIES[model];
}

const TEXT_ONLY_MODELS: Readonly<Record<string, true>> = { "deepseek-3.2": true, "minimax-m2.1": true, "minimax-m2.5": true, "glm-5": true, "qwen3-coder-next": true };

export function supportsKiroImages(model: string): boolean {
  return !Object.hasOwn(TEXT_ONLY_MODELS, model);
}
