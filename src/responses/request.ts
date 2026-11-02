import { v4 as uuidv4 } from "uuid";
import type { ResponseCreateParamsBase } from "openai/resources/responses/responses";
import { imageSizeViolation, parseInlineImage, REMOTE_IMAGE, UNSUPPORTED_MODEL_IMAGE } from "../imageLimit.ts";
import { remapKiroToolHistory, sanitizeKiroTools } from "../kiro/sanitizer.ts";
import type { ProviderCredentials } from "../kiro/base.ts";
import { getKiroModelCapabilities, supportsKiroImages } from "../kiro/modelCapabilities.ts";
import { decodeReasoning, mergeKiroReasoning, type KiroReasoningContent } from "../kiro/reasoning.ts";
import { buildKiroThinkingEnvelope } from "../kiro/request/adaptiveThinking.ts";
import { normalizeKiroConversation, synthesizeHistoryToolSpecs, lowerKiroToolChoice, type KiroCurrentMessage } from "../kiro/request/history.ts";
import type { KiroHistoryItem, KiroUserInputMessageContext } from "../kiro/request/openai-to-kiro.ts";

/**
 * Responses request accepted by the native Kiro adapter. `input` 由
 * {@link buildKiroResponsesPayload} 逐项校验，客户端回放的输出项（无 id 的
 * reasoning、output_text 消息等）比 SDK 的输入类型宽，因此不继承 SDK 的 input。
 */
export type KiroResponsesRequest = Omit<Partial<ResponseCreateParamsBase>, "input"> & Record<string, unknown> & {
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
  conversationState: { chatTriggerType: "MANUAL"; conversationId: string; currentMessage: KiroCurrentMessage; history: KiroHistoryItem[] };
  profileArn?: string;
  inferenceConfig?: { maxTokens?: number };
  additionalModelRequestFields?: { reasoning: { effort: string; summary?: string } };
  _toolNameMap?: Record<string, string>;
  _responseToolMetadata?: Record<string, { name: string; namespace?: string; custom: boolean }>;
  /** Internal response wiring: expose opaque reasoning content when requested. */
  _includeEncryptedContent?: boolean;
};

const UNSUPPORTED_FIELDS: Record<string, true> = {
  previous_response_id: true,
  conversation: true,
  background: true,
  prompt: true,
  truncation: true,
};
/**
 * Responses `reasoning.effort` → the Kiro level to send.
 *
 * `none` is a real GPT-5.6 level, not an absence: these models think at `high`
 * by default upstream, so mapping it to "" would drop the field and silently
 * give the client the opposite of what it asked for.
 */
const EFFORTS: Readonly<Record<string, string>> = {
  minimal: "low",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
  max: "max",
  none: "none",
};
type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj =>
  !!value && typeof value === "object" && !Array.isArray(value);

type InputParts = {
  text: string;
  images: Array<{ format: string; bytes: string }>;
};
type KiroToolUse = { toolUseId: string; name: string; input: Obj };

function historyUserMessage(parts: InputParts, model: string): KiroHistoryItem {
  return {
    userInputMessage: {
      content: parts.text,
      modelId: model,
      origin: "AI_EDITOR",
      ...(parts.images.length
        ? {
            images: parts.images.map((image) => ({
              format: image.format,
              source: { bytes: image.bytes },
            })),
          }
        : {}),
    },
  };
}

function appendAssistantHistory(
  history: KiroHistoryItem[],
  content: string,
  toolUse?: KiroToolUse
): void {
  const previous = history.at(-1)?.assistantResponseMessage;
  if (previous) {
    if (content) previous.content = [previous.content, content].filter(Boolean).join("\n\n");
    if (toolUse) (previous.toolUses ??= []).push(toolUse);
    return;
  }
  history.push({
    assistantResponseMessage: {
      content,
      ...(toolUse ? { toolUses: [toolUse] } : {}),
    },
  });
}
function reasoningHistoryItem(item: Obj, model: string): KiroReasoningContent | undefined {
  const summary: string[] = [];
  if (item.summary !== undefined) {
    if (!Array.isArray(item.summary)) throw new Error("reasoning summary must be an array");
    for (const part of item.summary) {
      if (!isObj(part) || part.type !== "summary_text" || typeof part.text !== "string") {
        throw new Error("reasoning summary items must contain summary_text");
      }
      summary.push(part.text);
    }
  }
  if (item.encrypted_content !== undefined && item.encrypted_content !== null && typeof item.encrypted_content !== "string") {
    throw new Error("reasoning encrypted_content must be a string");
  }
  if (typeof item.encrypted_content === "string" && item.encrypted_content) return decodeReasoning(item.encrypted_content, model);
  const text = summary.join("");
  return text ? { reasoningText: { text } } : undefined;
}

function imageOf(value: unknown, path: string, supportsImages: boolean, toolResult = false): InputParts {
  if (!isObj(value) || value.type !== "input_image" || typeof value.image_url !== "string") throw new Error("input_image requires image_url");
  const image = parseInlineImage(value);
  if (!image) {
    if (/^https?:\/\//i.test(value.image_url)) return { text: REMOTE_IMAGE, images: [] };
    throw new Error("input_image requires a base64 data URL");
  }
  const violation = imageSizeViolation(image.bytes);
  if (violation) throw new Error(`Invalid '${path}.image_url': ${violation}`);
  if (!supportsImages) return { text: UNSUPPORTED_MODEL_IMAGE, images: [] };
  return { text: toolResult ? `[image: image/${image.format}]` : "", images: [image] };
}
function contentParts(value: unknown, path: string, supportsImages: boolean): InputParts {
  if (typeof value === "string") return { text: value, images: [] };
  if (!Array.isArray(value)) throw new Error("message content must be a string or array");
  const text: string[] = []; const images: Array<{ format: string; bytes: string }> = [];
  for (const [index, part] of value.entries()) {
    if (!isObj(part)) throw new Error("message content parts must be objects");
    if (part.type === "input_text" || part.type === "output_text") {
      if (typeof part.text !== "string") throw new Error("Text content must be a string");
      text.push(part.text);
    }
    else if (part.type === "input_image") {
      const image = imageOf(part, `${path}[${index}]`, supportsImages);
      text.push(image.text); images.push(...image.images);
    }
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
/**
 * 工具结果的文本进 toolResult；图片与 Anthropic 入口一致，挂到承载这批
 * toolResults 的 user 回合 images 上，并在文本中留下 `[image: …]` 占位。
 */
function toolOutput(value: unknown, path: string, supportsImages: boolean): InputParts {
  if (typeof value === "string") return { text: value, images: [] };
  if (Array.isArray(value)) {
    const text: string[] = [];
    const images: InputParts["images"] = [];
    for (const [index, part] of value.entries()) {
      if (isObj(part) && part.type === "input_text" && typeof part.text === "string") text.push(part.text);
      else if (isObj(part) && part.type === "input_image") {
        const image = imageOf(part, `${path}[${index}]`, supportsImages, true);
        images.push(...image.images);
        text.push(`${text.length ? "\n" : ""}${image.text}`);
      } else throw new Error("function_call_output output must contain input_text or input_image parts");
    }
    return { text: text.join(""), images };
  }
  if (isObj(value) && typeof value.text === "string") return { text: value.text, images: [] };
  throw new Error("function_call_output output must be text or content parts");
}


function flattenToolDefinitions(
  definitions: unknown[],
  output: Obj[],
  metadata: Record<string, { name: string; namespace?: string; custom: boolean }>,
  seenNames: Record<string, true>,
  namespace?: string
): void {
  for (const value of definitions) {
    if (!isObj(value) || typeof value.type !== "string") {
      throw new Error("Invalid tool definition");
    }
    if (value.type === "namespace") {
      if (typeof value.name !== "string" || !value.name || !Array.isArray(value.tools)) {
        throw new Error("namespace tool requires name and tools");
      }
      flattenToolDefinitions(value.tools, output, metadata, seenNames, value.name);
      continue;
    }
    if (value.type !== "function" && value.type !== "custom") {
      throw new Error(`Unsupported tool type: ${value.type}`);
    }
    if (typeof value.name !== "string" || !value.name) {
      throw new Error(`${value.type} tool requires a name`);
    }
    const originalName = value.name;
    const flattenedName = namespace ? `${namespace}__${originalName}` : originalName;
    if (Object.hasOwn(seenNames, flattenedName)) throw new Error(`Duplicate tool: ${flattenedName}`);
    seenNames[flattenedName] = true;
    if (namespace || value.type === "custom") {
      metadata[flattenedName] = {
        name: originalName,
        ...(namespace ? { namespace } : {}),
        custom: value.type === "custom",
      };
    }
    if (value.type === "function") {
      output.push({ ...value, name: flattenedName });
      continue;
    }
    const formatDescription = isObj(value.format)
      ? `\n\nCustom tool input format: ${JSON.stringify(value.format)}`
      : "";
    output.push({
      type: "function",
      name: flattenedName,
      description:
        (typeof value.description === "string" ? value.description : "") + formatDescription,
      parameters: {
        type: "object",
        properties: { input: { type: "string" } },
        required: ["input"],
        additionalProperties: false,
      },
    });
  }
}
/** Build the exact conversationState wire payload consumed by the Kiro executor. */
export function buildKiroResponsesPayload(model: string, body: KiroResponsesRequest, stream: boolean, credentials: ProviderCredentials, supportsImages = supportsKiroImages(model)): KiroResponsesPayload {
  void stream;
  const capabilities = getKiroModelCapabilities(model);
  const includeEncryptedContent = Array.isArray(body.include) && body.include.includes("reasoning.encrypted_content");
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
  let reasoningSummary = "";
  if (body.reasoning !== undefined && body.reasoning !== null) {
    if (!isObj(body.reasoning)) throw new Error("reasoning must be an object");
    if (body.reasoning.effort !== undefined && body.reasoning.effort !== null) {
      if (typeof body.reasoning.effort !== "string" || !Object.hasOwn(EFFORTS, body.reasoning.effort)) {
        throw new Error("reasoning.effort must be one of minimal, low, medium, high, xhigh, max, or none");
      }
      reasoningEffort = EFFORTS[body.reasoning.effort];
    }
    if (body.reasoning.summary !== undefined && body.reasoning.summary !== null) {
      const validSummaries: Readonly<Record<string, true>> = {
        auto: true,
        concise: true,
        detailed: true,
      };
      if (
        typeof body.reasoning.summary !== "string" ||
        !Object.hasOwn(validSummaries, body.reasoning.summary)
      ) {
        throw new Error("reasoning.summary must be auto, concise, or detailed");
      }
      reasoningSummary = body.reasoning.summary;
    }
  }
  if (body.max_output_tokens !== undefined && body.max_output_tokens !== null && (typeof body.max_output_tokens !== "number" || !Number.isInteger(body.max_output_tokens) || body.max_output_tokens < 1)) throw new Error("max_output_tokens must be a positive integer");
  if (isObj(body.text) && isObj(body.text.format) && body.text.format.type !== undefined && body.text.format.type !== "text") {
    throw new Error("Unsupported text.format on this relay");
  }
  const rawInput = body.input ?? "";
  if (typeof rawInput !== "string" && !Array.isArray(rawInput)) throw new Error("input must be a string or array");
  const items: Obj[] = typeof rawInput === "string" ? [{ type: "message", role: "user", content: [{ type: "input_text", text: rawInput }] }] : rawInput.map((v) => { if (!isObj(v)) throw new Error("input items must be objects"); return v; });
  const history: KiroHistoryItem[] = [];
  const instructionTexts: string[] = [];
  if (typeof body.instructions === "string" && body.instructions) instructionTexts.push(body.instructions);
  const additionalToolDefinitions: unknown[] = [];
  const historyToolMetadata: Record<string, { name: string; namespace?: string; custom: boolean }> = Object.create(null);
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const type = typeof item.type === "string" ? item.type : typeof item.role === "string" ? "message" : undefined;
    if (type === "additional_tools") {
      if (item.role !== "developer" || !Array.isArray(item.tools)) throw new Error("additional_tools requires developer role and tools");
      additionalToolDefinitions.push(...item.tools);
      continue;
    }
    if (type === "reasoning") {
      const reasoningContent = reasoningHistoryItem(item, model);
      if (reasoningContent) {
        const previous = history.at(-1)?.assistantResponseMessage;
        const merged = previous && !previous.toolUses?.length ? mergeKiroReasoning(previous.reasoningContent, reasoningContent) : undefined;
        // 一个推理槽放不下时另起连续助手消息；Kiro 会按顺序合并回同一回合。
        // Kiro 只接受连续助手消息的最后一条带 toolUses，前一段的工具调用随之移到新消息。
        if (previous && merged) previous.reasoningContent = merged;
        else {
          const carriedToolUses = previous?.toolUses;
          if (previous && carriedToolUses?.length) delete previous.toolUses;
          history.push({ assistantResponseMessage: { content: "", reasoningContent, ...(carriedToolUses?.length ? { toolUses: carriedToolUses } : {}) } });
        }
      }
      continue;
    }
    if (type === "function_call_output" || type === "custom_tool_call_output") {
      if (typeof item.call_id !== "string" || !item.call_id) throw new Error(`${type} requires call_id`);
      let user = history.at(-1)?.userInputMessage;
      if (!user) {
        const turn = historyUserMessage({ text: "", images: [] }, model);
        history.push(turn);
        user = turn.userInputMessage!;
      }
      const output = toolOutput(item.output, `input[${i}].output`, supportsImages);
      ((user.userInputMessageContext ??= {}).toolResults ??= []).push({ toolUseId: item.call_id, status: "success", content: [{ text: output.text }] });
      if (output.images.length) (user.images ??= []).push(...output.images.map(image => ({ format: image.format, source: { bytes: image.bytes } })));
      continue;
    }
    if (type === "function_call" || type === "custom_tool_call") {
      if (typeof item.call_id !== "string" || !item.call_id || typeof item.name !== "string" || !item.name) throw new Error(`${type} requires call_id and name`);
      const input = type === "custom_tool_call" ? (typeof item.input === "string" ? { input: item.input } : (() => { throw new Error("custom_tool_call input must be a string"); })()) : functionInput(item.arguments);
      const historyName = typeof item.namespace === "string" && item.namespace ? `${item.namespace}__${item.name}` : item.name;
      historyToolMetadata[historyName] = { name: item.name, ...(typeof item.namespace === "string" && item.namespace ? { namespace: item.namespace } : {}), custom: type === "custom_tool_call" };
      appendAssistantHistory(history, "", { toolUseId: item.call_id, name: historyName, input });
      continue;
    }
    if (type !== "message") throw new Error(`Unsupported input item type: ${String(type)}`);
    if (item.role !== "user" && item.role !== "assistant" && item.role !== "system" && item.role !== "developer") throw new Error("message role is invalid");
    const parts = contentParts(item.content, `input[${i}].content`, supportsImages);
    if (item.role === "system" || item.role === "developer") { instructionTexts.push(parts.text); continue; }
    if (item.role === "assistant") { appendAssistantHistory(history, parts.text); continue; }
    const previous = history.at(-1)?.userInputMessage;
    if (previous) {
      previous.content = [previous.content, parts.text].filter(Boolean).join("\n\n");
      if (parts.images.length) (previous.images ??= []).push(...parts.images.map(image => ({ format: image.format, source: { bytes: image.bytes } })));
    } else history.push(historyUserMessage(parts, model));
  }
  const lastUser = history.at(-1)?.userInputMessage;
  if (lastUser) history.pop();
  const currentText = lastUser?.content ?? "";
  const currentImages = lastUser?.images?.map(image => ({ format: image.format, bytes: image.source.bytes })) ?? [];
  const current: InputParts = {
    text: currentText.trim() || currentImages.length || lastUser?.userInputMessageContext?.toolResults?.length ? currentText : "...",
    images: currentImages,
  };
  const instruction = instructionTexts.length ? instructionTexts.map((text) => `<system-reminder>\n${text}\n</system-reminder>`).join("\n\n") + "\n\n" : "";
  const context: KiroUserInputMessageContext = {};
  if (lastUser?.userInputMessageContext) context.toolResults = lastUser.userInputMessageContext.toolResults;
  const declaredTools: unknown[] = [];
  if (body.tools !== undefined && body.tools !== null) {
    if (!Array.isArray(body.tools)) throw new Error("tools must be an array");
    declaredTools.push(...body.tools);
  }
  declaredTools.push(...additionalToolDefinitions);
  let nameMap: Record<string, string> | undefined;
  let responseToolMetadata: Record<string, { name: string; namespace?: string; custom: boolean }> | undefined;
  const metadata: Record<string, { name: string; namespace?: string; custom: boolean }> = Object.create(null);
  let tools: NonNullable<KiroUserInputMessageContext["tools"]>;
  if (declaredTools.length) {
    const flattened: Obj[] = [];
    flattenToolDefinitions(declaredTools, flattened, metadata, Object.create(null));
    tools = flattened.map((tool) => {
      const fn = isObj(tool.function) ? tool.function : tool;
      if (typeof fn.name !== "string" || !fn.name) throw new Error("function tool requires a name");
      return { toolSpecification: { name: fn.name, description: typeof fn.description === "string" ? fn.description : `Tool: ${fn.name}`, inputSchema: { json: isObj(fn.parameters) ? fn.parameters : { type: "object", properties: {} } } } };
    });
  } else {
    Object.assign(metadata, historyToolMetadata);
    tools = synthesizeHistoryToolSpecs(history).map(tool => metadata[tool.toolSpecification.name]?.custom
      ? { toolSpecification: { ...tool.toolSpecification, inputSchema: { json: { type: "object", properties: { input: { type: "string" } }, required: ["input"], additionalProperties: false } } } }
      : tool);
  }
  const sanitized = sanitizeKiroTools(tools);
  remapKiroToolHistory(history, sanitized.nameMap);
  if (sanitized.tools.length) context.tools = sanitized.tools;
  if (sanitized.nameMap.size) nameMap = Object.fromEntries(sanitized.nameMap);
  if (Object.keys(metadata).length) responseToolMetadata = metadata;
  if (body.tool_choice !== undefined) {
    const choice = body.tool_choice;
    const flattened = isObj(choice) && typeof choice.name === "string" && typeof choice.namespace === "string" && choice.namespace
      ? { type: choice.type, name: `${choice.namespace}__${choice.name}` } : choice;
    context.toolChoice = lowerKiroToolChoice(flattened, { sanitizedNames: new Set(sanitized.tools.map(tool => tool.toolSpecification.name)), nameMap: sanitized.nameMap });
  }
  const payload: KiroResponsesPayload = {
    conversationState: {
      chatTriggerType: "MANUAL",
      conversationId: uuidv4(),
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
  const normalized = normalizeKiroConversation(history, payload.conversationState.currentMessage, model);
  payload.conversationState.history = normalized.history;
  payload.conversationState.currentMessage = normalized.currentMessage;
  payload._includeEncryptedContent = includeEncryptedContent;
  const profileArn = credentials.providerSpecificData?.profileArn; if (typeof profileArn === "string" && profileArn) payload.profileArn = profileArn;
  if (typeof body.max_output_tokens === "number") payload.inferenceConfig = { maxTokens: body.max_output_tokens };
  // Shared with Chat Completions and Anthropic Messages so all three surfaces
  // produce the same upstream envelope. This endpoint is native-only (asserted
  // above), but routing through the same builder keeps the shapes from drifting.
  const envelope = buildKiroThinkingEnvelope(
    model,
    reasoningEffort,
    reasoningEffort === "none",
    "native",
    reasoningSummary,
  );
  // Native-only endpoint, so the builder can only produce the `reasoning` shape;
  // the guard is what proves that to the type system.
  if (envelope && "reasoning" in envelope) payload.additionalModelRequestFields = envelope;
  if (nameMap) payload._toolNameMap = nameMap;
  if (responseToolMetadata) payload._responseToolMetadata = responseToolMetadata;
  return payload;
}
