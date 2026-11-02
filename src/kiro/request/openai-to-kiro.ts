/**
 * OpenAI to Kiro Request Translator
 * Converts OpenAI Chat Completions format to Kiro/AWS CodeWhisperer format
 */
import { v4 as uuidv4, v5 as uuidv5 } from "uuid";
import {
  parseToolInput,
  serializeToolResultContent,
} from "./messageHelpers.ts";
import {
  buildKiroThinkingEnvelope,
  resolveKiroModelAlias,
  type KiroAdaptiveEnvelope,
  type KiroNativeEnvelope,
  type KiroThinkingMode,
} from "./adaptiveThinking.ts";
import type { ProviderCredentials } from "../base.ts";
import { supportsKiroImages } from "../modelCapabilities.ts";
import { remapKiroToolHistory, sanitizeKiroTools } from "../sanitizer.ts";
import { mergeKiroReasoning, type KiroReasoningContent } from "../reasoning.ts";
import { normalizeKiroConversation, synthesizeHistoryToolSpecs, lowerKiroToolChoice, type KiroCurrentMessage } from "./history.ts";
import { parseInlineImage, imageSizeViolation, REMOTE_IMAGE, UNSUPPORTED_MODEL_IMAGE } from "../../imageLimit.ts";

/** 用户回合附带的工具声明、工具结果与强制工具选择。 */
export type KiroUserInputMessageContext = {
  toolResults?: Array<{ toolUseId: string; status: string; content: Array<{ text: string }> }>;
  tools?: Array<{ toolSpecification: { name: string; description: string; inputSchema: { json: unknown } } }>;
  toolChoice?: { auto: object } | { any: object } | { none: object } | { tool: { name: string } };
};

/** 原生历史中的一项：用户回合或助手回合。 */
export type KiroHistoryItem = {
  userInputMessage?: {
    content: string;
    modelId: string;
    origin: string;
    images?: Array<{ format: string; source: { bytes: string } }>;
    userInputMessageContext?: KiroUserInputMessageContext;
  };
  assistantResponseMessage?: {
    content: string;
    reasoningContent?: KiroReasoningContent;
    toolUses?: Array<{ toolUseId: string; name: string; input: unknown }>;
  };
};

/**
 * Anthropic's direct-provider `[1m]` context-1m beta suffix. Kiro is AWS
 * Bedrock-backed and does not honor it, so forwarding a `kr/*` model id that
 * carries `[1m]` produces a malformed upstream model id at Bedrock.
 */
export const KIRO_UNSUPPORTED_CONTEXT_1M_SUFFIX = "[1m]";
export const KIRO_UNSUPPORTED_CONTEXT_1M_MESSAGE =
  "[kr/*] '[1m]' suffix is not supported by Kiro upstream. Kiro is AWS " +
  "Bedrock-backed and does not honor Anthropic's context-1m beta. Use a " +
  "direct-Anthropic provider for 1M-context routing.";

/**
 * Kiro is AWS Bedrock-backed, so Anthropic's direct-provider `[1m]` context
 * beta cannot be forwarded as part of a `kr/*` model id.
 */
export function hasUnsupportedKiroContextSuffix(model: unknown): boolean {
  return (
    typeof model === "string" && model.toLowerCase().includes(KIRO_UNSUPPORTED_CONTEXT_1M_SUFFIX)
  );
}

/**
 * Wrap system-prompt content in <system-reminder> tags before it is merged into
 * a Kiro user message. Kiro/CodeWhisperer has no `system` role, so without this
 * the system prompt would appear as raw user text (issue #2306).
 */
function wrapSystemReminder(text: string): string {
  return `<system-reminder>\n${text}\n</system-reminder>`;
}

/** Kiro rejects a `toolSpecification.description` longer than ~10000 chars. */
const KIRO_TOOL_DESC_MAX = 10000;

/** OpenAI- and Anthropic-shaped tool declarations, as clients actually send them. */
type KiroToolInput = {
  name?: string;
  description?: string;
  parameters?: unknown;
  input_schema?: unknown;
  function?: { name?: string; description?: string; parameters?: unknown };
};

/**
 * Build Kiro `toolSpecification` entries, relocating any oversized description
 * out of the schema and returning it separately.
 *
 * Kiro rejects oversized descriptions, so schemas retain a pointer while the
 * complete text is carried by the initial instruction context. The docs are
 * returned separately so history promotion cannot drop them or leak an
 * unsupported field into the upstream payload.
 */
function buildKiroToolSpecs(tools: KiroToolInput[]): {
  specs: NonNullable<KiroUserInputMessageContext["tools"]>;
  docs: string;
} {
  const docs: string[] = [];
  const specs = tools.map((t) => {
    const name = t.function?.name || t.name || "";
    let description = t.function?.description || t.description || "";

    if (!description.trim()) {
      description = `Tool: ${name}`;
    }

    if (description.length > KIRO_TOOL_DESC_MAX) {
      docs.push(`## Tool: ${name}\n\n${description}`);
      description = `[Full documentation in initial context under '## Tool: ${name}']`;
    }
    const schema = t.function?.parameters ?? t.parameters ?? t.input_schema;

    return {
      toolSpecification: {
        name,
        description,
        inputSchema: {
          json: schema && typeof schema === "object" && !Array.isArray(schema) ? schema : { type: "object", properties: {} },
        },
      },
    };
  });

  return { specs, docs: docs.join("\n\n---\n\n") };
}

/**
 * Does this message carry Anthropic-style `tool_result` content blocks? Such a
 * user message is part of an open tool-result batch rather than new user input.
 */
function carriesToolResults(msg: RequestMessage): boolean {
  return Array.isArray(msg?.content) && msg.content.some((c) => c.type === "tool_result");
}

/**
 * Lookahead for issue #8903: is the text-only assistant message at `index`
 * genuinely sandwiched inside a tool-result batch?
 *
 * True only when a later `tool` message (or a `tool_result` content block on a
 * user message) still belongs to the same assistant turn — i.e. it appears
 * before the conversation moves on with real user text or a new assistant
 * tool-call turn. Consecutive text-only assistant messages are skipped so a
 * `tool -> assistant -> assistant -> tool` run still counts as interleaved.
 *
 * Returning false for the ordinary `tool -> assistant(final reply)` shape is
 * what keeps that reply on the normal flush path instead of being deferred.
 */
function hasFollowingToolResult(messages: RequestMessage[], index: number): boolean {
  for (let j = index + 1; j < messages.length; j++) {
    const next = messages[j];
    if (next.role === "tool") return true;

    if (next.role === "user") {
      const blocks = Array.isArray(next.content) ? next.content : [];
      // A user message carrying only tool_result blocks is still part of the
      // batch; one with real text ends it.
      if (blocks.some((c) => c.type === "tool_result")) {
        const hasText = blocks.some((c) => (c.type === "text" || c.text) && c.text?.trim());
        if (!hasText) return true;
      }
      return false;
    }

    if (next.role === "assistant") {
      const isTextOnly =
        (!next.tool_calls || next.tool_calls.length === 0) &&
        !(Array.isArray(next.content) && next.content.some((c) => c.type === "tool_use"));
      // Skip further text-only assistant messages; a new tool-call turn ends
      // the current batch.
      if (isTextOnly) continue;
      return false;
    }

    // system or any other role ends the batch
    return false;
  }
  return false;
}


/**
 * Convert OpenAI messages to Kiro format
 * Rules: system/tool/user -> user role, merge consecutive same roles
 */
type ToolCall = { id?: string; name?: string; input?: unknown; function?: { name: string; arguments?: unknown } };
type ContentBlock = ToolCall & { type?: string; text?: string; thinking?: string; signature?: string; data?: string; tool_use_id?: string; is_error?: boolean; content?: unknown };
type RequestMessage = { role?: string; content?: string | ContentBlock[]; tool_calls?: ToolCall[]; tool_call_id?: string };

/** Keep entire assistant messages, including opaque reasoning, outside an open result batch. */
function reorderInterleavedAssistant(messages: RequestMessage[]): RequestMessage[] {
  const ordered: RequestMessage[] = [];
  const deferred: RequestMessage[] = [];
  let inBatch = false;
  for (let index = 0; index < messages.length; index++) {
    const message = messages[index];
    const toolResult = message.role === "tool" || carriesToolResults(message);
    const assistantOnly = message.role === "assistant" && !message.tool_calls?.length &&
      !(Array.isArray(message.content) && message.content.some(block => block.type === "tool_use"));
    if (inBatch && assistantOnly && hasFollowingToolResult(messages, index)) {
      deferred.push(message);
      continue;
    }
    if (!toolResult && deferred.length) ordered.push(...deferred.splice(0));
    ordered.push(message);
    inBatch = toolResult;
  }
  ordered.push(...deferred);
  return ordered;
}

function convertMessages(messages: RequestMessage[], tools: KiroToolInput[], model: string, supportsImages: boolean) {
  const history: KiroHistoryItem[] = [];
  const ordered = reorderInterleavedAssistant(messages);
  for (let i = 0; i < ordered.length; i++) {
    const message = ordered[i];
    if (message.role === "assistant") {
      const blocks: ContentBlock[] = [];
      do {
        const assistant = ordered[i];
        const content = typeof assistant.content === "string" ? [{ type: "text", text: assistant.content }] : assistant.content ?? [];
        if (blocks.length && content.length) blocks.push({ type: "text", text: "" });
        blocks.push(...content);
        if (!content.some(block => block.type === "tool_use")) {
          blocks.push(...(assistant.tool_calls ?? []).map(call => ({ ...call, type: "tool_use" })));
        }
        i++;
      } while (i < ordered.length && ordered[i].role === "assistant");
      i--;
      let texts: string[] = [];
      let reasoning: KiroReasoningContent | undefined;
      let uses: NonNullable<NonNullable<KiroHistoryItem["assistantResponseMessage"]>["toolUses"]> = [];
      let toolIndex = 0;
      const flush = () => {
        history.push({ assistantResponseMessage: { content: texts.join("\n").trim(), ...(reasoning ? { reasoningContent: reasoning } : {}), ...(uses.length ? { toolUses: uses } : {}) } });
        texts = []; reasoning = undefined; uses = [];
      };
      for (const block of blocks) {
        const nextReasoning: KiroReasoningContent | undefined = block.type === "thinking" && typeof block.thinking === "string" && block.signature
          ? { reasoningText: { text: block.thinking, signature: block.signature } }
          : block.type === "redacted_thinking" && block.data ? { redactedContent: block.data } : undefined;
        if (nextReasoning) {
          const merged = uses.length ? undefined : mergeKiroReasoning(reasoning, nextReasoning);
          if (merged) reasoning = merged;
          else {
            if (reasoning || texts.length || uses.length) flush();
            reasoning = nextReasoning;
          }
        } else if (block.type === "text") texts.push(block.text ?? "");
        else if (block.type === "tool_use") {
          const name = block.function?.name ?? block.name ?? "";
          uses.push({ toolUseId: block.id || uuidv5(`${name}:${toolIndex}`, "a1b2c3d4-e5f6-7890-abcd-ef1234567890"), name, input: parseToolInput(block.function ? block.function.arguments : block.input) });
          toolIndex++;
        }
      }
      flush();
      continue;
    }
    if (!["system", "developer", "user", "tool"].includes(message.role ?? "")) continue;
    const user: KiroCurrentMessage = { userInputMessage: { content: "", modelId: model, origin: "AI_EDITOR" } };
    const images: NonNullable<typeof user.userInputMessage.images> = [];
    const lowerContent = (content: unknown, toolResult = false): string => {
      if (!Array.isArray(content)) return toolResult ? serializeToolResultContent(content) : typeof content === "string" ? content : "";
      const text: string[] = [];
      for (const block of content as ContentBlock[]) {
        if (block.type === "image" || block.type === "image_url") {
          const image = parseInlineImage(block);
          const violation = image && imageSizeViolation(image.bytes);
          if (violation) throw new Error(violation);
          if (!image) text.push(REMOTE_IMAGE);
          else if (!supportsImages) text.push(UNSUPPORTED_MODEL_IMAGE);
          else {
            images.push({ format: image.format, source: { bytes: image.bytes } });
            if (toolResult) text.push(`[image: image/${image.format}]`);
          }
        } else if (block.type === "tool_result") {
          const result = lowerContent(block.content, true);
          ((user.userInputMessage.userInputMessageContext ??= {}).toolResults ??= []).push({ toolUseId: block.tool_use_id ?? "", status: block.is_error ? "error" : "success", content: [{ text: result }] });
        } else if (typeof block.text === "string") text.push(block.text);
        else if (toolResult) text.push(serializeToolResultContent([block]));
      }
      return text.join("\n");
    };
    if (message.role === "tool") {
      user.userInputMessage.userInputMessageContext = { toolResults: [{ toolUseId: message.tool_call_id ?? "", status: "success", content: [{ text: lowerContent(message.content, true) }] }] };
    } else {
      const text = lowerContent(message.content);
      user.userInputMessage.content = (message.role === "system" || message.role === "developer") && text ? wrapSystemReminder(text) : text;
    }
    if (images.length) user.userInputMessage.images = images;
    history.push(user);
  }
  const last = history.at(-1);
  const currentMessage: KiroCurrentMessage = last?.userInputMessage
    ? { userInputMessage: history.pop()!.userInputMessage! }
    : { userInputMessage: { content: "...", modelId: model, origin: "AI_EDITOR" } };
  const built = buildKiroToolSpecs(tools);
  const specs = built.specs.length ? built.specs : synthesizeHistoryToolSpecs(history);
  if (specs.length) (currentMessage.userInputMessage.userInputMessageContext ??= {}).tools = specs;
  return { ...normalizeKiroConversation(history, currentMessage, model), toolDocs: built.docs };
}

/**
 * Reasoning-effort levels Kiro accepts, across both envelopes.
 *
 * `none` appears only in the GPT-5.6 `reasoning.effort` enum, and is how a client
 * turns thinking off on those models — without it they fall through to Kiro's own
 * `high` default. Claude's `output_config.effort` has no such level; it is switched
 * off through `thinking.type` instead, so the adaptive branch below drops `none`
 * rather than forwarding a value Kiro would reject.
 */
const KIRO_EFFORT_LEVELS = ["none", "low", "medium", "high", "xhigh", "max"];

/**
 * Resolve the Kiro effort level for a request, or "" when no reasoning was asked
 * for. Effort sources, in priority order:
 *   1. OpenAI-style `reasoning_effort`
 *   2. Anthropic adaptive-thinking `output_config.effort` (the canonical field)
 *   3. Anthropic `thinking` block — `{type:"enabled", budget_tokens}` mapped to a
 *      level via {@link effortFromBudget}; `{type:"adaptive"}` (no explicit
 *      effort) defaults to `high`, matching Anthropic's documented default
 *      (omitting `effort` ≡ `high`).
 * OpenAI's `minimal` collapses to `low` (Kiro has no `minimal`).
 */
function resolveKiroEffort(body: Record<string, unknown>): string {
  let effort = typeof body.reasoning_effort === "string" ? body.reasoning_effort.toLowerCase() : "";

  if (!effort) {
    const outputConfig = body.output_config as Record<string, unknown> | undefined;
    if (
      outputConfig &&
      typeof outputConfig === "object" &&
      typeof outputConfig.effort === "string"
    ) {
      effort = outputConfig.effort.toLowerCase();
    }
  }

  if (!effort) {
    const thinking = body.thinking as Record<string, unknown> | undefined;
    if (thinking && typeof thinking === "object") {
      if (thinking.type === "enabled") {
        effort = effortFromBudget(Number(thinking.budget_tokens) || 0);
      } else if (thinking.type === "adaptive") {
        effort = "high";
      }
    }
  }

  if (effort === "minimal") effort = "low";
  return KIRO_EFFORT_LEVELS.includes(effort) ? effort : "";
}

/** Map an Anthropic `thinking.budget_tokens` to a coarse Kiro effort level. */
function effortFromBudget(budget: number): string {
  if (budget >= 32000) return "high";
  if (budget >= 16000) return "medium";
  if (budget > 0) return "low";
  return "";
}


/**
 * Build Kiro payload from OpenAI format.
 *
 * `thinkingMode` is the envelope Kiro's live catalog says this model accepts.
 * Omitted -> the static table decides, which is what discovery-disabled
 * deployments and the offline fallback path get.
 */
export function buildKiroPayload(
  model: string,
  // OpenAI Chat 形状的请求体，翻译器按字段动态读取，入口已校验结构。
  body: Record<string, any>,
  stream: boolean,
  credentials: ProviderCredentials | null,
  thinkingMode?: KiroThinkingMode,
  supportsImages?: boolean,
) {
  // Reject the Anthropic-only `[1m]` context beta before it reaches Bedrock —
  // Kiro cannot honor it and a forwarded `kr/*[1m]` id is malformed upstream.
  if (hasUnsupportedKiroContextSuffix(model)) {
    throw new Error(KIRO_UNSUPPORTED_CONTEXT_1M_MESSAGE);
  }
  // Normalize model name: Claude Code sends dashes (claude-sonnet-4-6),
  // Kiro API expects dots (claude-sonnet-4.6). Convert trailing version segment.
  // The minor group is bounded to 1-2 digits so date-suffixed ids (e.g.
  // claude-opus-4-20250514) are never mistaken for a dash-separated minor
  // version and corrupted into claude-opus-4.20250514 (upstream 9router #2270).
  // The supported `-thinking` selector is a local alias: strip it before the request leaves
  // OmniRoute so Kiro only receives a real upstream model ID. Non-functional agentic and
  // auto-kiro aliases are rejected above instead of silently degrading to another model.
  const { upstream: normalizedModel, thinking: modelRequestsThinking } =
    resolveKiroModelAlias(model);
  const messages = body.messages || [];
  const tools = body.tools || [];
  const maxTokens = body.max_tokens ?? body.max_completion_tokens ?? 32000;
  const temperature = body.temperature;
  const topP = body.top_p;

  const { history, currentMessage, toolDocs } = convertMessages(messages, tools, normalizedModel, supportsImages ?? supportsKiroImages(normalizedModel));

  const rawProfileArn = credentials?.providerSpecificData?.profileArn;
  const profileArn = typeof rawProfileArn === "string" ? rawProfileArn : "";

  // Kiro has no system role. Use the same instruction wrapper as system
  // prompts, anchored to the first user carrier so it stays at the same
  // history position across requests. A tool-result continuation must keep
  // its current content for observations, not newly injected documentation.
  if (toolDocs) {
    const firstUser = history.find(item => item.userInputMessage)?.userInputMessage
      ?? currentMessage.userInputMessage;
    const docs = wrapSystemReminder(`# Tool Documentation\n\n${toolDocs}`);
    firstUser.content = `${docs}\n\n${firstUser.content}`;
  }

  const finalContent = currentMessage?.userInputMessage?.content || "";

  const payload: {
    conversationState: {
      chatTriggerType: string;
      conversationId: string;
      currentMessage: {
        userInputMessage: {
          content: string;
          modelId: string;
          origin: string;
          images?: Array<{ format: string; source: { bytes: string } }>;
          userInputMessageContext?: KiroUserInputMessageContext;
        };
      };
      history: KiroHistoryItem[];
    };
    profileArn?: string;
    inferenceConfig?: {
      maxTokens?: number;
      temperature?: number;
      topP?: number;
    };
    additionalModelRequestFields?: {
      reasoning?: KiroNativeEnvelope["reasoning"];
      thinking?: KiroAdaptiveEnvelope["thinking"];
      output_config?: KiroAdaptiveEnvelope["output_config"];
    };
  } = {
    conversationState: {
      chatTriggerType: "MANUAL",
      conversationId: uuidv4(),
      currentMessage: {
        userInputMessage: {
          content: finalContent,
          modelId: normalizedModel,
          origin: "AI_EDITOR",
          ...(currentMessage?.userInputMessage?.images?.length && {
            images: currentMessage.userInputMessage.images,
          }),
          ...(currentMessage?.userInputMessage?.userInputMessageContext && {
            userInputMessageContext: currentMessage.userInputMessage.userInputMessageContext,
          }),
        },
      },
      history: history,
    },
  };


  if (profileArn) {
    payload.profileArn = profileArn;
  }

  if (maxTokens || temperature !== undefined || topP !== undefined) {
    payload.inferenceConfig = {};
    if (maxTokens) payload.inferenceConfig.maxTokens = maxTokens;
    if (temperature !== undefined) payload.inferenceConfig.temperature = temperature;
    if (topP !== undefined) payload.inferenceConfig.topP = topP;
  }

  // 官方客户端按模型目录声明的路径发送 effort，不向消息注入思考指令。
  // The live catalog wins when present: Kiro publishes each model's accepted
  // envelope as `additionalModelRequestFieldsSchema`, so a model that gains
  // thinking upstream starts getting it here without a code change.
  const requestedEffort = resolveKiroEffort(body) || (modelRequestsThinking ? "high" : "");
  // Adaptive thinking is the model deciding per turn whether it needs to think,
  // so the compatible default is on — a client that wants it off says so. Native
  // GPT-5.6 models default to `high` upstream, so "off" must be sent explicitly;
  // {@link buildKiroThinkingEnvelope} owns that asymmetry.
  const thinkingTurnedOff =
    body.thinking?.type === "disabled" ||
    (typeof body.reasoning_effort === "string" && body.reasoning_effort.toLowerCase() === "none");
  const envelope = buildKiroThinkingEnvelope(
    normalizedModel,
    requestedEffort,
    thinkingTurnedOff,
    thinkingMode,
  );
  if (envelope) {
    payload.additionalModelRequestFields = envelope;

    // Adaptive-only Claude models (Opus 4.7/4.8, Sonnet 5, Fable 5) reject a
    // non-default temperature / top_p with a 400 while thinking is active, so
    // strip both. Drop inferenceConfig entirely if nothing else remains.
    // Thinking that was actually turned off imposes no such restriction; read it
    // off the envelope, since models that cannot disable thinking ignore "off".
    const thinkingActive =
      "thinking" in envelope ? envelope.thinking.type !== "disabled" : envelope.reasoning.effort !== "none";
    if (payload.inferenceConfig && thinkingActive) {
      delete payload.inferenceConfig.temperature;
      delete payload.inferenceConfig.topP;
      if (Object.keys(payload.inferenceConfig).length === 0) {
        delete payload.inferenceConfig;
      }
    }
  }

  const target = payload.conversationState.currentMessage.userInputMessage;
  const context = target.userInputMessageContext;
  const sanitized = sanitizeKiroTools(context?.tools ?? []);
  if (context?.tools) context.tools = sanitized.tools;
  remapKiroToolHistory(payload.conversationState.history, sanitized.nameMap);
  if (sanitized.nameMap.size) {
    (payload as typeof payload & { _toolNameMap: Record<string, string> })._toolNameMap = Object.fromEntries(sanitized.nameMap);
  }
  if (body.tool_choice !== undefined) {
    (target.userInputMessageContext ??= {}).toolChoice = lowerKiroToolChoice(body.tool_choice, {
      sanitizedNames: new Set(sanitized.tools.map(tool => tool.toolSpecification.name)), nameMap: sanitized.nameMap,
    });
  }

  return payload;
}

