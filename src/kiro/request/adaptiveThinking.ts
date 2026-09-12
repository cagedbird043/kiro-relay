/**
 * Which `additionalModelRequestFields` envelope a Kiro model accepts.
 *
 *   "adaptive" — Claude shape: `{ thinking: { type }, output_config: { effort } }`
 *   "native"   — GPT-5.6 shape: `{ reasoning: { effort } }`
 *   "none"     — the model rejects the field entirely (issue #6576: Kiro 400s
 *                `claude-sonnet-4.5` / `claude-haiku-4.5` with
 *                "additionalModelRequestFields is not supported for this model")
 *
 * Kiro publishes this per model in `ListAvailableModels` as
 * `additionalModelRequestFieldsSchema`, so the live catalog is authoritative and
 * `deriveKiroThinkingMode` reads it straight off the wire. The static tables
 * below are the offline answer only: discovery disabled, or before/after a
 * failed discovery pass. They were wrong for months — Kiro declares the adaptive
 * envelope for the whole Opus/Sonnet 4.6+ line, and hard-coding just 4.7/4.8
 * silently withheld thinking from `claude-opus-4.6` and `claude-sonnet-4.6`.
 */
export type KiroThinkingMode = "adaptive" | "native" | "none";

const KIRO_ADAPTIVE_THINKING_MODELS: Record<string, true> = {
  "claude-opus-4.6": true,
  "claude-opus-4.7": true,
  "claude-opus-4.8": true,
  "claude-opus-5": true,
  "claude-sonnet-4.6": true,
  "claude-sonnet-5": true,
};
const KIRO_NATIVE_REASONING_MODELS: Record<string, true> = {
  "gpt-5.6-sol": true,
  "gpt-5.6-terra": true,
  "gpt-5.6-luna": true,
};

export function supportsKiroAdaptiveThinking(normalizedModel: string): boolean {
  return KIRO_ADAPTIVE_THINKING_MODELS[normalizedModel] === true;
}

export function supportsKiroNativeReasoning(normalizedModel: string): boolean {
  return KIRO_NATIVE_REASONING_MODELS[normalizedModel] === true;
}

/**
 * Read the envelope a model accepts off its catalog entry. `reasoning` and
 * `thinking` are the two top-level properties Kiro declares; a null/absent
 * schema means the model takes no envelope at all.
 */
export function deriveKiroThinkingMode(schema: unknown): KiroThinkingMode {
  if (!schema || typeof schema !== "object" || !("properties" in schema)) return "none";
  const properties = schema.properties;
  if (!properties || typeof properties !== "object") return "none";
  if ("reasoning" in properties) return "native";
  if ("thinking" in properties) return "adaptive";
  return "none";
}

/** Offline answer for a model id, used when the live catalog is unavailable. */
export function staticKiroThinkingMode(normalizedModel: string): KiroThinkingMode {
  if (KIRO_NATIVE_REASONING_MODELS[normalizedModel]) return "native";
  if (KIRO_ADAPTIVE_THINKING_MODELS[normalizedModel]) return "adaptive";
  return "none";
}

const KIRO_UNSUPPORTED_AGENTIC_MESSAGE =
  "Kiro agentic aliases are not supported. The '-agentic' suffix did not change the " +
  "upstream request; select a real Kiro model instead.";
const KIRO_UNSUPPORTED_THINKING_MESSAGE =
  "This Kiro model does not support the '-thinking' alias. Use a model returned by Kiro's " +
  "live catalog with Thinking capability.";
const KIRO_REMOVED_AUTO_ALIAS_MESSAGE =
  "'auto-kiro' is not a real Kiro upstream model. Select a model returned by the live catalog.";

export function resolveKiroModelAlias(model: unknown): { upstream: string; thinking: boolean } {
  let upstream = String(model || "");
  if (upstream.endsWith("-agentic")) throw new Error(KIRO_UNSUPPORTED_AGENTIC_MESSAGE);
  if (upstream === "auto-kiro") throw new Error(KIRO_REMOVED_AUTO_ALIAS_MESSAGE);

  const thinking = upstream.endsWith("-thinking");
  if (thinking) upstream = upstream.slice(0, -"-thinking".length);
  upstream = upstream.replace(/^(claude-(?:opus|sonnet|haiku|3-\d+)-\d+)-(\d{1,2})$/, "$1.$2");

  if (thinking && !supportsKiroAdaptiveThinking(upstream)) {
    throw new Error(KIRO_UNSUPPORTED_THINKING_MESSAGE);
  }
  return { upstream, thinking };
}
