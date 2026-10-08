/**
 * Which `additionalModelRequestFields` envelope a Kiro model accepts.
 *
 *   "adaptive"      — Claude shape: `{ thinking: { type }, output_config: { effort } }`
 *   "adaptive_only" — Claude shape whose `thinking.type` enum has no `disabled`
 *                     (Opus/Sonnet 5.5 preview): thinking cannot be switched off,
 *                     so a client's "off" is ignored rather than turned into a 400
 *   "native"        — GPT-5.6 shape: `{ reasoning: { effort } }`
 *   "none"          — the model rejects the field entirely (issue #6576: Kiro 400s
 *                     `claude-sonnet-4.5` / `claude-haiku-4.5` with
 *                     "additionalModelRequestFields is not supported for this model")
 *
 * Kiro publishes this per model in `ListAvailableModels` as
 * `additionalModelRequestFieldsSchema`, so the live catalog is authoritative and
 * `deriveKiroThinkingMode` reads it straight off the wire. The static tables
 * below are the offline answer only: discovery disabled, or before/after a
 * failed discovery pass. They were wrong for months — Kiro declares the adaptive
 * envelope for the whole Opus/Sonnet 4.6+ line, and hard-coding just 4.7/4.8
 * silently withheld thinking from `claude-opus-4.6` and `claude-sonnet-4.6`.
 */
export type KiroThinkingMode = "adaptive" | "adaptive_only" | "native" | "none";

const KIRO_ADAPTIVE_THINKING_MODELS: Record<string, "adaptive" | "adaptive_only"> = {
  "claude-opus-4.6": "adaptive",
  "claude-opus-4.7": "adaptive",
  "claude-opus-4.8": "adaptive",
  "claude-opus-5": "adaptive",
  "claude-opus-5.5": "adaptive_only",
  "claude-sonnet-4.6": "adaptive",
  "claude-sonnet-5": "adaptive",
  "claude-sonnet-5.5": "adaptive_only",
};
const KIRO_NATIVE_REASONING_MODELS: Record<string, true> = {
  "gpt-5.6-sol": true,
  "gpt-5.6-terra": true,
  "gpt-5.6-luna": true,
};

export function supportsKiroAdaptiveThinking(normalizedModel: string): boolean {
  return KIRO_ADAPTIVE_THINKING_MODELS[normalizedModel] !== undefined;
}

export function supportsKiroNativeReasoning(normalizedModel: string): boolean {
  return KIRO_NATIVE_REASONING_MODELS[normalizedModel] === true;
}

/** GPT-5.6 envelope: thinking level is stated directly, `none` included. */
export type KiroNativeEnvelope = { reasoning: { effort: string; summary?: string } };
/** Claude envelope: effort and the on/off switch are separate fields. */
export type KiroAdaptiveEnvelope = {
  output_config?: { effort: string };
  thinking: { type: "adaptive" | "disabled" };
};
export type KiroThinkingEnvelope = KiroNativeEnvelope | KiroAdaptiveEnvelope;

/**
 * Build the `additionalModelRequestFields` envelope, or null when the model
 * takes none.
 *
 * Every public interface — Chat Completions, Anthropic Messages, Responses —
 * routes through here, so each one can keep its own protocol's standard way of
 * asking for (or refusing) thinking and still produce the same upstream request.
 * Envelope choice is a property of the model, never of the client protocol.
 *
 * @param effort      normalized Kiro level, or "" when the client said nothing
 * @param thinkingOff the client explicitly asked for no thinking; ignored for
 *                    "adaptive_only" models, which cannot turn thinking off
 */
export function buildKiroThinkingEnvelope(
  normalizedModel: string,
  effort: string,
  thinkingOff: boolean,
  thinkingMode?: KiroThinkingMode,
  summary?: string,
): KiroThinkingEnvelope | null {
  const mode = thinkingMode ?? staticKiroThinkingMode(normalizedModel);
  const usesNative = mode === "native";
  const usesAdaptive = mode === "adaptive" || mode === "adaptive_only";

  if (usesNative) {
    // GPT-5.6 defaults to `high` upstream, so "off" has to be stated as a level.
    const level = thinkingOff ? "none" : effort;
    if (!level) return null;
    return { reasoning: { effort: level, ...(summary ? { summary } : {}) } };
  }

  if (usesAdaptive) {
    // Claude's effort enum has no "none"; thinking is switched off via `thinking.type`.
    // `thinking.type` is required by the schema — sending only `output_config`
    // leaves adaptive thinking off, which is why Claude 4.6+ stayed silent no
    // matter what effort was requested.
    return {
      ...(effort && effort !== "none" ? { output_config: { effort } } : {}),
      thinking: { type: thinkingOff && mode !== "adaptive_only" ? "disabled" : "adaptive" },
    };
  }

  return null;
}

/**
 * Read the envelope a model accepts off its catalog entry. `reasoning` and
 * `thinking` are the two top-level properties Kiro declares; a null/absent
 * schema means the model takes no envelope at all. A declared `thinking.type`
 * enum without `disabled` marks a model whose thinking cannot be switched off.
 */
export function deriveKiroThinkingMode(schema: unknown): KiroThinkingMode {
  if (!schema || typeof schema !== "object" || !("properties" in schema)) return "none";
  const properties = schema.properties;
  if (!properties || typeof properties !== "object") return "none";
  if ("reasoning" in properties) return "native";
  if ("thinking" in properties) {
    const typeEnum = (properties.thinking as { properties?: { type?: { enum?: unknown } } } | null)
      ?.properties?.type?.enum;
    return Array.isArray(typeEnum) && !typeEnum.includes("disabled") ? "adaptive_only" : "adaptive";
  }
  return "none";
}

/** Offline answer for a model id, used when the live catalog is unavailable. */
export function staticKiroThinkingMode(normalizedModel: string): KiroThinkingMode {
  if (KIRO_NATIVE_REASONING_MODELS[normalizedModel]) return "native";
  return KIRO_ADAPTIVE_THINKING_MODELS[normalizedModel] ?? "none";
}

const KIRO_UNSUPPORTED_AGENTIC_MESSAGE =
  "Kiro agentic aliases are not supported. The '-agentic' suffix did not change the " +
  "upstream request; select a real Kiro model instead.";
const KIRO_UNSUPPORTED_THINKING_MESSAGE =
  "This Kiro model does not support the '-thinking' alias. Use a model returned by Kiro's " +
  "live catalog with Thinking capability.";
const KIRO_REMOVED_AUTO_ALIAS_MESSAGE =
  "'auto-kiro' is not a real Kiro upstream model. Select a model returned by the live catalog.";

/** Anthropic API id (`claude-opus-5-5`) → the dotted id Kiro routes on (`claude-opus-5.5`). */
export function kiroModelId(model: string): string {
  return model.replace(/^(claude-(?:opus|sonnet|haiku|3-\d+)-\d+)-(\d{1,2})$/, "$1.$2");
}

/** Kiro's dotted id → the Anthropic API id clients see (`claude-opus-5.5` → `claude-opus-5-5`). */
export function publicModelId(model: string): string {
  return model.replace(/^(claude-(?:opus|sonnet|haiku)-\d+)\.(\d{1,2})$/, "$1-$2");
}

export function resolveKiroModelAlias(model: unknown): { upstream: string; thinking: boolean } {
  let upstream = String(model || "");
  if (upstream.endsWith("-agentic")) throw new Error(KIRO_UNSUPPORTED_AGENTIC_MESSAGE);
  if (upstream === "auto-kiro") throw new Error(KIRO_REMOVED_AUTO_ALIAS_MESSAGE);

  const thinking = upstream.endsWith("-thinking");
  if (thinking) upstream = upstream.slice(0, -"-thinking".length);
  upstream = kiroModelId(upstream);

  if (thinking && !supportsKiroAdaptiveThinking(upstream)) {
    throw new Error(KIRO_UNSUPPORTED_THINKING_MESSAGE);
  }
  return { upstream, thinking };
}
