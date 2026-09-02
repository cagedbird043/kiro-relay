/**
 * Explicit capabilities of the Kiro models exposed by this relay.
 *
 * Unknown model ids intentionally have no capabilities. This keeps newly
 * discovered or unsupported upstream models conservative (notably images are
 * omitted unless the model is explicitly listed here).
 */
export type KiroModelCapabilities = {
  contextLength: number;
  supportsImages: boolean;
  supportsAdaptiveReasoning: boolean;
  supportsNativeReasoning: boolean;
};

export const KIRO_MODEL_CAPABILITIES: Readonly<Record<string, KiroModelCapabilities>> = {
  "claude-opus-4.7": { contextLength: 200_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-opus-4.8": { contextLength: 200_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-sonnet-4": { contextLength: 200_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-sonnet-4.5": { contextLength: 200_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-sonnet-4.6": { contextLength: 200_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-haiku-4.5": { contextLength: 200_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "deepseek-3.2": { contextLength: 200_000, supportsImages: false, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "minimax-m2.5": { contextLength: 200_000, supportsImages: false, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "glm-5": { contextLength: 200_000, supportsImages: false, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "qwen3-coder-next": { contextLength: 200_000, supportsImages: false, supportsAdaptiveReasoning: false, supportsNativeReasoning: false },
  "claude-opus-5": { contextLength: 1_000_000, supportsImages: true, supportsAdaptiveReasoning: true, supportsNativeReasoning: false },
  "claude-sonnet-5": { contextLength: 1_000_000, supportsImages: true, supportsAdaptiveReasoning: true, supportsNativeReasoning: false },
  "gpt-5.6-sol": { contextLength: 272_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: true },
  "gpt-5.6-terra": { contextLength: 272_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: true },
  "gpt-5.6-luna": { contextLength: 272_000, supportsImages: true, supportsAdaptiveReasoning: false, supportsNativeReasoning: true },
};

export function getKiroModelCapabilities(model: string): KiroModelCapabilities | undefined {
  return KIRO_MODEL_CAPABILITIES[model];
}

export function supportsKiroImages(model: string): boolean {
  return getKiroModelCapabilities(model)?.supportsImages === true;
}
