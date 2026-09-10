/**
 * Kiro tool-schema sanitizer.
 *
 * Kiro / AWS CodeWhisperer rejects requests with HTTP 400 "Improperly formed
 * request" when a tool's `inputSchema.json` contains JSON-Schema keywords it
 * does not understand (`additionalProperties`, `anyOf`/`oneOf`/`allOf`/`not`,
 * `$ref`/`$defs`/`definitions`, `if`/`then`/`else`, etc.) or an empty
 * `required` array. It also rejects tool names longer than 64 characters.
 *
 * This pure helper strips the unsupported keys recursively, ensures a
 * top-level `required: []` is present, and hash-truncates over-long names —
 * returning a `nameMap` (truncated → original) so the streamed tool-call name
 * can be mapped back for the client. See chatCore.ts (request wiring) and
 * translator/response/kiro-to-openai.ts (response reverse-mapping).
 */
import { createHash } from "node:crypto";

/** Max tool-name length Kiro accepts before it rejects the request. */
const MAX_TOOL_NAME_LENGTH = 64;

/** JSON-Schema keywords Kiro rejects anywhere in a tool schema. */
const STRIP_KEYS: Readonly<Record<string, true>> = {
  additionalProperties: true,
  anyOf: true,
  oneOf: true,
  allOf: true,
  not: true,
  $schema: true,
  $id: true,
  $ref: true,
  $defs: true,
  definitions: true,
  if: true,
  then: true,
  else: true,
  unevaluatedProperties: true,
  unevaluatedItems: true,
  contentEncoding: true,
  contentMediaType: true,
};

/**
 * Recursively drop unsupported JSON-Schema keys and empty `required` arrays.
 * Non-object/array values are returned untouched.
 */
function stripKeys(value: unknown): unknown {
  if (!value || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(stripKeys);

  const cleaned: Record<string, unknown> = {};
  for (const [key, val] of Object.entries(value)) {
    if (Object.hasOwn(STRIP_KEYS, key) || (key === "required" && Array.isArray(val) && val.length === 0)) continue;
    let result = val;
    // 属性映射的键是业务名称；枚举、默认值等普通数据不能按 Schema 清洗。
    if ((key === "properties" || key === "patternProperties" || key === "dependentSchemas" || key === "dependencies") && val && typeof val === "object" && !Array.isArray(val)) {
      result = Object.fromEntries(Object.entries(val).map(([name, schema]) => [name, stripKeys(schema)]));
    } else if (key === "items" || key === "prefixItems" || key === "contains" || key === "propertyNames" || key === "additionalItems") {
      result = stripKeys(val);
    }
    Object.defineProperty(cleaned, key, { value: result, enumerable: true, writable: true, configurable: true });
  }
  return cleaned;
}

/** A single Kiro tool entry (loose shape — upstream payloads vary). */
type KiroTool = {
  toolSpecification?: {
    name?: string;
    inputSchema?: { json?: unknown };
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

export interface SanitizeKiroToolsResult<T = unknown> {
  tools: T;
  /** Map of truncated tool name → original name, for response reverse-mapping. */
  nameMap: Map<string, string>;
}

/**
 * Sanitize a Kiro `tools` array in place-safely (returns new objects).
 *
 * - Strips JSON-Schema keys Kiro rejects from each `inputSchema.json`.
 * - Ensures the top-level schema carries `required: []` (Kiro requires the key).
 * - Hash-truncates tool names > 64 chars and records the mapping in `nameMap`.
 * - Deterministically disambiguates generated names that would collide with
 *   another generated name or an existing short tool name.
 *
 * Inputs that are not arrays are returned untouched with an empty `nameMap`,
 * so this is safe to call unconditionally on the Kiro tools slot.
 */
export function sanitizeKiroTools<T>(tools: T): SanitizeKiroToolsResult<T> {
  const nameMap = new Map<string, string>();

  if (!tools || !Array.isArray(tools)) {
    return { tools, nameMap };
  }

  const entries = tools as KiroTool[];
  // Short names are emitted unchanged, so reserve them before generating any
  // truncated names. This prevents a generated name from shadowing a real one.
  const usedNames = new Set<string>();
  for (const tool of entries) {
    const name = tool?.toolSpecification?.name;
    if (typeof name === "string" && name.length <= MAX_TOOL_NAME_LENGTH) {
      usedNames.add(name);
    }
  }

  const generatedName = (original: string, attempt: number): string => {
    const hashInput = attempt === 0 ? original : `${original}\u0000${attempt}`;
    const hash = createHash("sha256").update(hashInput).digest("hex").slice(0, 7);
    return `${original.slice(0, 56)}_${hash}`;
  };

  const sanitized = entries.map((tool) => {
    const spec = tool?.toolSpecification;
    if (!spec) return tool;

    const originalName = spec.name;
    let name = originalName;
    if (typeof name === "string" && name.length > MAX_TOOL_NAME_LENGTH) {
      let attempt = 0;
      let candidate = generatedName(name, attempt);
      while (usedNames.has(candidate)) {
        candidate = generatedName(name, ++attempt);
      }
      name = candidate;
      usedNames.add(candidate);
      // Every generated key is unique, so this mapping can never overwrite a
      // different original name (including duplicate declarations).
      nameMap.set(candidate, originalName);
    }

    const schema = spec.inputSchema?.json;
    if (schema && typeof schema === "object" && !Array.isArray(schema)) {
      const cleaned = stripKeys(schema) as Record<string, unknown>;
      if (!cleaned.required) {
        cleaned.required = [];
      }
      return {
        ...tool,
        toolSpecification: {
          ...spec,
          name,
          inputSchema: { json: cleaned },
        },
      };
    }

    return {
      ...tool,
      toolSpecification: { ...spec, name },
    };
  });

  return { tools: sanitized as T, nameMap };
}

type KiroToolHistoryItem = { assistantResponseMessage?: { toolUses?: Array<{ name: string }> } };

/** 将新构造历史中的工具名称映射到本次声明使用的上游名称。 */
export function remapKiroToolHistory(history: readonly KiroToolHistoryItem[], nameMap: ReadonlyMap<string, string>): void {
  if (!nameMap.size) return;
  const forward = new Map<string, string>();
  for (const [sanitized, original] of nameMap) if (!forward.has(original)) forward.set(original, sanitized);
  for (const item of history) {
    for (const tool of item.assistantResponseMessage?.toolUses ?? []) tool.name = forward.get(tool.name) ?? tool.name;
  }
}
