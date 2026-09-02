import { v5 as uuidv5 } from "uuid";
import type { ResponseCreateParamsBase } from "openai/resources/responses/responses";
import { sanitizeKiroTools } from "../kiro/sanitizer.ts";
import type { ProviderCredentials } from "../kiro/base.ts";
import { getKiroModelCapabilities } from "../kiro/modelCapabilities.ts";

/** Responses request accepted by the native Kiro adapter. */
export type KiroResponsesRequest = Partial<ResponseCreateParamsBase> & Record<string, unknown> & {
  model: string;
  input?: unknown;
  instructions?: string | null;
  tools?: unknown;
  tool_choice?: unknown;
  reasoning?: unknown;
  max_output_tokens?: unknown;
  stream?: boolean;
};

export type KiroResponsesPayload = {
  conversationState: { chatTriggerType: "MANUAL"; conversationId: string; currentMessage: { userInputMessage: { content: string; modelId: string; origin: "AI_EDITOR"; images?: Array<{ format: string; source: { bytes: string } }>; userInputMessageContext?: { tools?: unknown[]; toolResults?: unknown[]; toolChoice?: unknown } } }; history: unknown[] };
  profileArn?: string;
  inferenceConfig?: { maxTokens?: number };
  additionalModelRequestFields?: { reasoning: { effort: string } };
  _toolNameMap?: Record<string, string>;
};

const UUID_NAMESPACE = "34f7193f-561d-4050-bc84-9547d953d6bf";
const UNSUPPORTED_FIELDS: Record<string, true> = {
  previous_response_id: true,
  conversation: true,
  background: true,
  prompt: true,
  truncation: true,
};
const EFFORTS: Readonly<Record<string, string>> = {
  minimal: "low",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
  max: "max",
  none: "",
};
type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj =>
  !!value && typeof value === "object" && !Array.isArray(value);

function imageOf(value: unknown): { format: string; bytes: string } {
  if (!isObj(value) || value.type !== "input_image" || typeof value.image_url !== "string") throw new Error("input_image requires image_url");
  const m = /^data:image\/(png|jpe?g|gif|webp);base64,([A-Za-z0-9+/]+={0,2})$/i.exec(value.image_url);
  if (!m) throw new Error("input_image requires a base64 data URL");
  return { format: m[1].toLowerCase() === "jpg" ? "jpeg" : m[1].toLowerCase(), bytes: m[2] };
}
function contentParts(value: unknown): { text: string; images: Array<{ format: string; bytes: string }> } {
  if (typeof value === "string") return { text: value, images: [] };
  if (!Array.isArray(value)) throw new Error("message content must be a string or array");
  const text: string[] = []; const images: Array<{ format: string; bytes: string }> = [];
  for (const part of value) {
    if (!isObj(part)) throw new Error("message content parts must be objects");
    if (part.type === "input_text" || part.type === "output_text") {
      if (typeof part.text !== "string") throw new Error("Text content must be a string");
      text.push(part.text);
    }
    else if (part.type === "input_image") images.push(imageOf(part));
    else throw new Error(`Unsupported content part type: ${String(part.type)}`);
  }
  return { text: text.join(""), images };
}
function functionInput(value: unknown): Obj {
  if (typeof value !== "string") { if (isObj(value)) return value; throw new Error("function_call arguments must be a JSON object string"); }
  let parsed: unknown; try { parsed = JSON.parse(value); } catch { throw new Error("function_call arguments must be valid JSON"); }
  if (!isObj(parsed)) throw new Error("function_call arguments must be a JSON object");
  return parsed;
}
function outputText(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return contentParts(value).text;
  if (isObj(value) && typeof value.text === "string") return value.text;
  throw new Error("function_call_output output must be text or content parts");
}

/** Build the exact conversationState wire payload consumed by the Kiro executor. */
export function buildKiroResponsesPayload(model: string, body: KiroResponsesRequest, stream: boolean, credentials: ProviderCredentials): KiroResponsesPayload {
  void stream;
  const capabilities = getKiroModelCapabilities(model);
  if (!capabilities?.supportsNativeReasoning) throw new Error("Responses endpoint supports only native-reasoning GPT-5.6 models");
  for (const key of Object.keys(UNSUPPORTED_FIELDS)) if (body[key] !== undefined) throw new Error(`Unsupported Responses field: ${key}`);
  if (body.store !== undefined && body.store !== false) {
    throw new Error("Responses storage is not supported");
  }
  if (
    body.include !== undefined &&
    (!Array.isArray(body.include) ||
      body.include.some((item) => item !== "reasoning.encrypted_content"))
  ) {
    throw new Error("Unsupported Responses include value");
  }
  if (body.prompt_cache_key !== undefined && typeof body.prompt_cache_key !== "string") {
    throw new Error("prompt_cache_key must be a string");
  }
  let reasoningEffort = "";
  if (body.reasoning !== undefined && body.reasoning !== null) {
    if (
      !isObj(body.reasoning) ||
      typeof body.reasoning.effort !== "string" ||
      !Object.hasOwn(EFFORTS, body.reasoning.effort)
    ) {
      throw new Error(
        "reasoning.effort must be one of minimal, low, medium, high, xhigh, max, or none"
      );
    }
    reasoningEffort = EFFORTS[body.reasoning.effort];
  }
  if (body.max_output_tokens !== undefined && body.max_output_tokens !== null && (typeof body.max_output_tokens !== "number" || !Number.isInteger(body.max_output_tokens) || body.max_output_tokens < 1)) throw new Error("max_output_tokens must be a positive integer");
  const rawInput = body.input ?? "";
  if (typeof rawInput !== "string" && !Array.isArray(rawInput)) throw new Error("input must be a string or array");
  const items: Obj[] = typeof rawInput === "string" ? [{ type: "message", role: "user", content: [{ type: "input_text", text: rawInput }] }] : rawInput.map((v) => { if (!isObj(v)) throw new Error("input items must be objects"); return v; });
  const history: unknown[] = []; const toolResults: unknown[] = []; let current: { text: string; images: Array<{ format: string; bytes: string }> } = { text: "", images: [] }; let firstUser = "";
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const type = typeof item.type === "string"
      ? item.type
      : typeof item.role === "string"
        ? "message"
        : undefined;
    if (type === "function_call_output") { if (typeof item.call_id !== "string" || !item.call_id) throw new Error("function_call_output requires call_id"); toolResults.push({ toolUseId: item.call_id, status: "success", content: [{ text: outputText(item.output) }] }); continue; }
    if (type === "function_call") { if (typeof item.call_id !== "string" || !item.call_id || typeof item.name !== "string" || !item.name) throw new Error("function_call requires call_id and name"); history.push({ assistantResponseMessage: { content: "", toolUses: [{ toolUseId: item.call_id, name: item.name, input: functionInput(item.arguments) }] } }); continue; }
    if (type !== "message") throw new Error(`Unsupported input item type: ${String(type)}`);
    if (item.role !== "user" && item.role !== "assistant" && item.role !== "system" && item.role !== "developer") throw new Error("message role is invalid");
    const parts = contentParts(item.content); const role = item.role === "assistant" ? "assistant" : "user";
    if (role === "user" && !firstUser) firstUser = parts.text;
    if (i === items.length - 1) current = parts; else if (role === "assistant") history.push({ assistantResponseMessage: { content: parts.text } }); else history.push({ userInputMessage: { content: parts.text, modelId: model, origin: "AI_EDITOR", ...(parts.images.length ? { images: parts.images.map((x) => ({ format: x.format, source: { bytes: x.bytes } })) } : {}) } });
  }
  const instruction = typeof body.instructions === "string" && body.instructions ? `<system-reminder>\n${body.instructions}\n</system-reminder>\n\n` : "";
  const context: { tools?: unknown[]; toolResults?: unknown[]; toolChoice?: unknown } = {};
  if (toolResults.length) context.toolResults = toolResults;
  if (body.tool_choice !== undefined) { const c = body.tool_choice; if (c === "auto") context.toolChoice = { auto: {} }; else if (c === "required") context.toolChoice = { any: {} }; else if (c === "none") context.toolChoice = { none: {} }; else if (isObj(c) && c.type === "function" && typeof c.name === "string" && c.name) context.toolChoice = { tool: { name: c.name } }; else throw new Error("Invalid tool_choice"); }
  let nameMap: Record<string, string> | undefined;
  if (body.tools !== undefined) { if (!Array.isArray(body.tools)) throw new Error("tools must be an array"); for (const tool of body.tools) { if (!isObj(tool) || tool.type !== "function") throw new Error("Only function tools are supported"); }
    const tools = body.tools.map((tool) => { const raw: unknown = tool; const fn = isObj(raw) && isObj(raw.function) ? raw.function : raw; if (!isObj(fn) || typeof fn.name !== "string" || !fn.name) throw new Error("function tool requires a name"); return { toolSpecification: { name: fn.name, description: typeof fn.description === "string" ? fn.description : undefined, inputSchema: { json: isObj(fn.parameters) ? fn.parameters : { type: "object", properties: {} } } } }; });
    const sanitized = sanitizeKiroTools(tools); context.tools = sanitized.tools; if (sanitized.nameMap.size) nameMap = Object.fromEntries(sanitized.nameMap);
  }
  const conversationSeed =
    typeof body.prompt_cache_key === "string" && body.prompt_cache_key
      ? body.prompt_cache_key
      : (firstUser || current.text).slice(0, 4000);
  const payload: KiroResponsesPayload = {
    conversationState: {
      chatTriggerType: "MANUAL",
      conversationId: uuidv5(conversationSeed, UUID_NAMESPACE),
      currentMessage: {
        userInputMessage: {
          content: instruction + current.text,
          modelId: model,
          origin: "AI_EDITOR",
          ...(current.images.length
            ? {
                images: current.images.map((image) => ({
                  format: image.format,
                  source: { bytes: image.bytes },
                })),
              }
            : {}),
          ...(Object.keys(context).length ? { userInputMessageContext: context } : {}),
        },
      },
      history,
    },
  };
  const profileArn = credentials.providerSpecificData?.profileArn; if (typeof profileArn === "string" && profileArn) payload.profileArn = profileArn;
  if (typeof body.max_output_tokens === "number") payload.inferenceConfig = { maxTokens: body.max_output_tokens };
  if (reasoningEffort) {
    payload.additionalModelRequestFields = { reasoning: { effort: reasoningEffort } };
  }
  if (nameMap) payload._toolNameMap = nameMap; return payload;
}
