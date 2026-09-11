/**
 * OpenAI to Kiro Request Translator
 * Converts OpenAI Chat Completions format to Kiro/AWS CodeWhisperer format
 */
import { v4 as uuidv4, v5 as uuidv5 } from "uuid";
import {
  parseToolInput,
  normalizeKiroToolSchema,
  serializeToolResultContent,
} from "./messageHelpers.ts";
import {
  resolveKiroModelAlias,
  supportsKiroAdaptiveThinking,
  supportsKiroNativeReasoning,
} from "./adaptiveThinking.ts";
import { supportsKiroImages } from "../modelCapabilities.ts";
import { remapKiroToolHistory, sanitizeKiroTools } from "../sanitizer.ts";
import { mergeKiroReasoning, type KiroReasoningContent } from "../reasoning.ts";

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
 * `origin` tells Kiro which surface is calling. `AI_EDITOR` is the Kiro IDE, and
 * the service answers it with the full IDE treatment: a ~3700-token "you are
 * Kiro" persona prepended server-side, plus contextUsage and metering frames.
 * Other accepted surfaces (notably `CLI`) skip the persona entirely — the model
 * then answers as plain Amazon Q — but their streams carry only
 * assistantResponse/toolUse frames, with no usage and no terminal event.
 */
export const KIRO_DEFAULT_ORIGIN = "AI_EDITOR";

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
 * Kiro answers a raw upstream 400 for a description over
 * {@link KIRO_TOOL_DESC_MAX}, so the schema keeps a pointer and the full text is
 * handed back to be prepended to the current turn's content — the same
 * relocation kiro-gateway performs in
 * `converters_core.py::process_tools_with_long_descriptions`.
 *
 * The docs are *returned* rather than stashed on the message object, because the
 * tool-bearing user turn is moved into `history` on every multi-turn request
 * (see the currentMessage promotion below). Carrying them on the message lost
 * them there — the model then saw only the pointer and no documentation — and
 * also leaked an unknown `_toolDocs` field into the upstream payload, which Kiro
 * rejects.
 */
function buildKiroToolSpecs(tools: KiroToolInput[]): {
  specs: Array<Record<string, unknown>>;
  docs: string;
} {
  const docs: string[] = [];
  const specs = tools.map((t) => {
    const name = t.function?.name || t.name;
    let description = t.function?.description || t.description || "";

    if (!description.trim()) {
      description = `Tool: ${name}`;
    }

    if (description.length > KIRO_TOOL_DESC_MAX) {
      docs.push(`## Tool: ${name}\n\n${description}`);
      description = `[Full documentation in system prompt under '## Tool: ${name}']`;
    }

    return {
      toolSpecification: {
        name,
        description,
        inputSchema: {
          json: normalizeKiroToolSchema(
            t.function?.parameters || t.parameters || t.input_schema || {}
          ),
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
function carriesToolResults(msg): boolean {
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
function hasFollowingToolResult(messages, index: number): boolean {
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
function convertMessages(messages, tools, model, origin = KIRO_DEFAULT_ORIGIN) {
  let history = [];
  let currentMessage = null;

  let pendingUserContent = [];
  let pendingAssistantContent = [];
  let pendingReasoning: KiroReasoningContent | undefined;
  let pendingToolResults = [];
  // Text-only assistant turns that arrived in the middle of an open tool-result
  // batch. They are held back so the batch stays contiguous, then emitted as
  // their own assistant turn right after the batch flushes — see
  // `interruptsOpenToolBatch` below (issue #8903).
  let deferredAssistantContent: string[] = [];
  let pendingImages: Array<{ format: string; source: { bytes: string } }> = [];
  let currentRole = null;
  let toolsAttached = false;
  let toolDocs = "";

  // Attachments are accepted only for explicitly listed Kiro models. Unknown
  // and known text-only models remain image-disabled by default.
  const supportsImages = typeof model === "string" && supportsKiroImages(model);

  const flushPending = () => {
    if (currentRole === "user") {
      const content = pendingUserContent.join("\n\n").trim();
      const userMsg: {
        userInputMessage: {
          content: string;
          modelId: string;
          images?: Array<{ format: string; source: { bytes: string } }>;
          origin: string;
          userInputMessageContext?: {
            toolResults?: Array<Record<string, unknown>>;
            tools?: Array<Record<string, unknown>>;
          };
        };
      } = {
        userInputMessage: {
          content: content,
          modelId: "",
          origin,
        },
      };

      if (pendingToolResults.length > 0) {
        userMsg.userInputMessage.userInputMessageContext = {
          toolResults: pendingToolResults,
        };
      }

      // Attach images to userInputMessage (NOT userInputMessageContext)
      if (pendingImages.length > 0) {
        userMsg.userInputMessage.images = pendingImages;
      }

      // Add tools to the first emitted user turn. We track a flag instead of
      // relying on `history.length === 0` because the first few messages may
      // be assistant turns (e.g. when role=undefined collapses to a prior
      // assistant turn), in which case the first user flush would already see
      // a non-empty history and lose the tools schema.
      if (tools && tools.length > 0 && !toolsAttached) {
        if (!userMsg.userInputMessage.userInputMessageContext) {
          userMsg.userInputMessage.userInputMessageContext = {};
        }
        const built = buildKiroToolSpecs(tools);
        userMsg.userInputMessage.userInputMessageContext.tools = built.specs;
        if (built.docs) toolDocs = built.docs;
        toolsAttached = true;
      }

      history.push(userMsg);
      currentMessage = userMsg;
      pendingUserContent = [];
      pendingToolResults = [];
      pendingImages = [];

      // The tool batch is now closed, so any assistant text that was held back
      // to keep it contiguous can be emitted as its own turn (issue #8903).
      // Without this the deferred text would sit in a queue nothing drains and
      // be silently dropped from the transcript.
      if (deferredAssistantContent.length > 0) {
        history.push({
          assistantResponseMessage: {
            content: deferredAssistantContent.join("\n\n").trim(),
          },
        });
        deferredAssistantContent = [];
      }
    } else if (currentRole === "assistant") {
      const content = pendingAssistantContent.join("\n\n").trim();
      const assistantMsg = {
        assistantResponseMessage: {
          content: content,
          ...(pendingReasoning ? { reasoningContent: pendingReasoning } : {}),
        },
      };
      history.push(assistantMsg);
      pendingAssistantContent = [];
      pendingReasoning = undefined;
    }
  };

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    let role = msg.role;

    // Normalize: system/tool -> user
    if (role === "system" || role === "tool") {
      role = "user";
    }

    // If role changes, flush pending
    //
    // Exception: a text-only assistant message must not split a batch of tool
    // results that answers a single assistant turn. `tool` is normalized to
    // `user` above, so `tool -> assistant -> tool` looks like two role changes
    // and the interleaved flush would emit the first tool result and drop the
    // rest, leaving advertised `toolUses` without matching `toolResults`.
    // Bedrock rejects that transcript with 400 "Expected toolResult blocks"
    // (issue #8903). Defer the assistant text instead so the tool batch stays
    // contiguous; the text is re-emitted as its own assistant turn as soon as
    // the batch flushes.
    //
    // The lookahead matters: without it, an ordinary trailing assistant reply
    // (`tool -> assistant`, with no further tool message) would also be
    // deferred and its text lost. Only a genuine sandwich qualifies.
    const isTextOnlyAssistant =
      msg.role === "assistant" &&
      (!msg.tool_calls || msg.tool_calls.length === 0) &&
      !(Array.isArray(msg.content) && msg.content.some((c) => c.type === "tool_use"));
    const interruptsOpenToolBatch =
      isTextOnlyAssistant &&
      currentRole === "user" &&
      pendingToolResults.length > 0 &&
      hasFollowingToolResult(messages, i);

    if (interruptsOpenToolBatch) {
      const deferredText =
        typeof msg.content === "string"
          ? msg.content.trim()
          : Array.isArray(msg.content)
            ? msg.content
                .filter((c) => c.type === "text" || c.text)
                .map((c) => c.text || "")
                .join("\n")
                .trim()
            : "";
      if (deferredText) deferredAssistantContent.push(deferredText);
      continue;
    }

    // Once assistant text has been deferred, the tool batch is logically over
    // as soon as a message arrives that is not itself a tool result. Flush now
    // so the pending batch + deferred assistant turn are emitted before the new
    // user text, instead of that text merging into the tool-result turn and
    // leaving the deferred reply stranded after it (issue #8903).
    if (
      deferredAssistantContent.length > 0 &&
      currentRole === "user" &&
      msg.role !== "tool" &&
      !carriesToolResults(msg)
    ) {
      flushPending();
      currentRole = null;
    }

    if (role !== currentRole && currentRole !== null) {
      flushPending();
    }
    currentRole = role;
    if (role === "user") {
      // Extract content
      let content = "";
      if (typeof msg.content === "string") {
        content = msg.content;
      } else if (Array.isArray(msg.content)) {
        const textParts = msg.content
          .filter((c) => c.type === "text" || c.text)
          .map((c) => c.text || "");
        content = textParts.join("\n");

        // Extract images (OpenAI image_url and Anthropic image formats).
        // Skip entirely for models that do not support images — see supportsImages.
        for (const block of msg.content) {
          if (supportsImages && block.type === "image_url") {
            const url: string = block.image_url?.url || "";
            if (url.startsWith("data:")) {
              // data:image/jpeg;base64,<data>
              const [header, bytes] = url.split(",", 2);
              const mediaType = header.split(";")[0].replace("data:", ""); // e.g. "image/jpeg"
              const format = mediaType.split("/")[1] || "jpeg";
              if (bytes) pendingImages.push({ format, source: { bytes } });
            }
          } else if (supportsImages && block.type === "image" && block.source?.type === "base64") {
            const format = (block.source.media_type || "image/jpeg").split("/")[1] || "jpeg";
            if (block.source.data)
              pendingImages.push({ format, source: { bytes: block.source.data } });
          } else if (supportsImages && block.type === "image" && typeof block.image === "string") {
            // AI SDK-style image part: { type: "image", image: "data:...;base64,..." } (#1330)
            const url = block.image;
            if (url.startsWith("data:")) {
              const [header, bytes] = url.split(",", 2);
              const mediaType = header.split(";")[0].replace("data:", "");
              const format = mediaType.split("/")[1] || "jpeg";
              if (bytes) pendingImages.push({ format, source: { bytes } });
            }
          }
        }

        // Check for tool_result blocks. Kiro keeps tool observations in
        // userInputMessageContext, while image bytes belong on the surrounding
        // userInputMessage. Preserve both parts of an Anthropic Read result.
        const toolResultBlocks = msg.content.filter((c) => c.type === "tool_result");
        if (toolResultBlocks.length > 0) {
          toolResultBlocks.forEach((block) => {
            const text = serializeToolResultContent(block.content);
            pendingToolResults.push({
              toolUseId: block.tool_use_id,
              status: block.is_error ? "error" : "success",
              content: [{ text: text }],
            });
            if (supportsImages && Array.isArray(block.content)) {
              for (const item of block.content) {
                if (item?.type !== "image" || item.source?.type !== "base64") continue;
                const format = (item.source.media_type || "image/jpeg").split("/")[1] || "jpeg";
                if (item.source.data) {
                  pendingImages.push({ format, source: { bytes: item.source.data } });
                }
              }
            }
          });
        }
      }

      // Handle tool role (from normalized)
      if (msg.role === "tool") {
        // Reuse the shared serializer so non-string content (arrays, structured/JSON
        // blocks, images) is never collapsed to an empty string. CodeWhisperer rejects a
        // toolResult whose content is [{ text: "" }] with 400 "Improperly formed request"
        // — the same failure mode that hit the Anthropic tool_result path (issue #2446).
        const toolContent = serializeToolResultContent(msg.content);
        pendingToolResults.push({
          toolUseId: msg.tool_call_id,
          status: "success",
          content: [{ text: toolContent }],
        });
      } else if (content) {
        // #2306: Kiro/CodeWhisperer has no `system` role, so system messages are
        // normalized to `user`. Wrap their content in <system-reminder> tags so
        // the model can tell the system prompt apart from real user input instead
        // of treating the full Claude Code prompt as something the user typed.
        pendingUserContent.push(msg.role === "system" ? wrapSystemReminder(content) : content);
      }
    } else if (role === "assistant") {
      // Extract text content and tool uses
      let textContent = "";
      let toolUses = [];

      if (Array.isArray(msg.content)) {
        // 单个原生槽不能表达多个独立签名或隐藏块，不能静默覆盖。
        for (const block of msg.content) {
          if (block?.type === "thinking" && typeof block.thinking === "string" &&
              typeof block.signature === "string" && block.signature.length > 0) {
            pendingReasoning = mergeKiroReasoning(pendingReasoning, { reasoningText: { text: block.thinking, signature: block.signature } });
          } else if (block?.type === "redacted_thinking" && typeof block.data === "string" && block.data.length > 0) {
            pendingReasoning = mergeKiroReasoning(pendingReasoning, { redactedContent: block.data });
          }
        }
        const textBlocks = msg.content.filter((c) => c.type === "text");
        textContent = textBlocks
          .map((b) => b.text)
          .join("\n")
          .trim();

        const toolUseBlocks = msg.content.filter((c) => c.type === "tool_use");
        toolUses = toolUseBlocks;
      } else if (typeof msg.content === "string") {
        textContent = msg.content.trim();
      }

      if (msg.tool_calls && msg.tool_calls.length > 0) {
        toolUses = msg.tool_calls;
      }

      if (textContent) {
        pendingAssistantContent.push(textContent);
      }

      // Store tool uses in last assistant message
      if (toolUses.length > 0) {
        if (pendingAssistantContent.length === 0) {
          // pendingAssistantContent.push("Call tools");
        }

        // Flush to create assistant message with toolUses
        flushPending();

        const lastMsg = history[history.length - 1];
        if (lastMsg?.assistantResponseMessage) {
          const NAMESPACE_KIRO_TOOLUSE = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
          lastMsg.assistantResponseMessage.toolUses = toolUses.map((tc, idx) => {
            if (tc.function) {
              const stableId =
                tc.id || uuidv5(`${tc.function.name}:${idx}`, NAMESPACE_KIRO_TOOLUSE);
              return {
                toolUseId: stableId,
                name: tc.function.name,
                input: parseToolInput(tc.function.arguments),
              };
            } else {
              const stableId = tc.id || uuidv5(`${tc.name}:${idx}`, NAMESPACE_KIRO_TOOLUSE);
              return {
                toolUseId: stableId,
                name: tc.name,
                input: parseToolInput(tc.input),
              };
            }
          });
        }

        currentRole = null;
      }
    }
  }

  // Flush remaining
  if (currentRole !== null) {
    flushPending();
  }

  // Kiro requires currentMessage to be a user turn. If the request ends with a
  // user turn, move that final turn into currentMessage. If it ends with an
  // assistant/tool turn, synthesize a neutral filler ("...") instead of the
  // literal "Continue", which Kiro can read as a real instruction (#5231).
  if (history.length > 0 && history[history.length - 1].userInputMessage) {
    currentMessage = history.pop();
  } else {
    currentMessage = {
      userInputMessage: {
        content: "...",
        modelId: model,
      },
    };
  }

  // Promote the tools schema to currentMessage. Tools may have been attached
  // to any user turn in history (e.g. when the first message was assistant or
  // had an undefined role, the first user flush lands further down). Scan the
  // whole history so we never lose the schema.
  if (!currentMessage?.userInputMessage?.userInputMessageContext?.tools) {
    const carrier = history.find((item) => item?.userInputMessage?.userInputMessageContext?.tools);
    if (carrier?.userInputMessage?.userInputMessageContext?.tools) {
      if (!currentMessage.userInputMessage.userInputMessageContext) {
        currentMessage.userInputMessage.userInputMessageContext = {};
      }
      currentMessage.userInputMessage.userInputMessageContext.tools =
        carrier.userInputMessage.userInputMessageContext.tools;
    }
  }

  // Fallback: if the schema was never attached to any user turn (e.g. the
  // input contained no user messages and currentMessage is a synthesized
  // neutral-filler turn), attach the provided tools directly to currentMessage so
  // Kiro still sees the schema it needs to validate assistant.toolUses in
  // history.
  if (
    !toolsAttached &&
    tools &&
    tools.length > 0 &&
    !currentMessage?.userInputMessage?.userInputMessageContext?.tools
  ) {
    if (!currentMessage.userInputMessage.userInputMessageContext) {
      currentMessage.userInputMessage.userInputMessageContext = {};
    }
    const built = buildKiroToolSpecs(tools);
    currentMessage.userInputMessage.userInputMessageContext.tools = built.specs;
    if (built.docs) toolDocs = built.docs;
    toolsAttached = true;
  }

  // Clean up history for Kiro API compatibility
  history.forEach((item) => {
    if (item.userInputMessage?.userInputMessageContext?.tools) {
      delete item.userInputMessage.userInputMessageContext.tools;
    }

    if (
      item.userInputMessage?.userInputMessageContext &&
      Object.keys(item.userInputMessage.userInputMessageContext).length === 0
    ) {
      delete item.userInputMessage.userInputMessageContext;
    }

    if (item.userInputMessage && !item.userInputMessage.modelId) {
      item.userInputMessage.modelId = model;
    }

    // Kiro API requires `origin` on every userInputMessage
    if (item.userInputMessage && !item.userInputMessage.origin) {
      item.userInputMessage.origin = origin;
    }
  });

  // Kiro expects history to alternate between user and assistant turns. After
  // normalizing `system`/`tool` roles into `userInputMessage`, the history can
  // contain adjacent user turns, which Kiro can reject. Merge consecutive
  // `userInputMessage` entries by concatenating their content and preserving
  // any attached `userInputMessageContext` (e.g. accumulated toolResults).
  //
  // Why this is not redundant with the `flushPending` grouping in the main
  // loop: the assistant branch resets `currentRole = null` after emitting
  // `toolUses`. Any following `tool` role (normalized to user) and a
  // subsequent `user` role therefore each open their own flush, producing
  // two adjacent `userInputMessage` entries in history. This pass collapses
  // those.
  const mergedHistory: typeof history = [];
  for (const item of history) {
    const previous = mergedHistory[mergedHistory.length - 1];
    if (item.userInputMessage && previous?.userInputMessage) {
      const previousContent = previous.userInputMessage.content || "";
      const currentContent = item.userInputMessage.content || "";
      previous.userInputMessage.content = previousContent
        ? `${previousContent}\n\n${currentContent}`
        : currentContent;

      if (item.userInputMessage.userInputMessageContext) {
        const previousContext = previous.userInputMessage.userInputMessageContext || {};
        const nextContext = item.userInputMessage.userInputMessageContext;
        const mergedContext: Record<string, unknown> = { ...previousContext };

        for (const [key, value] of Object.entries(nextContext)) {
          const existing = (previousContext as Record<string, unknown>)[key];
          if (Array.isArray(existing) && Array.isArray(value)) {
            mergedContext[key] = [...existing, ...value];
          } else {
            mergedContext[key] = value;
          }
        }

        previous.userInputMessage.userInputMessageContext = mergedContext;
      }
    } else if (item.assistantResponseMessage && previous?.assistantResponseMessage) {
      // Kiro API also rejects consecutive assistant messages. Merge them.
      const previousContent = previous.assistantResponseMessage.content || "";
      const currentContent = item.assistantResponseMessage.content || "";
      previous.assistantResponseMessage.content = previousContent
        ? `${previousContent}\n\n${currentContent}`
        : currentContent;

      if (item.assistantResponseMessage.reasoningContent) {
        previous.assistantResponseMessage.reasoningContent = mergeKiroReasoning(previous.assistantResponseMessage.reasoningContent, item.assistantResponseMessage.reasoningContent);
      }
      if (item.assistantResponseMessage.toolUses) {
        const existingToolUses = previous.assistantResponseMessage.toolUses || [];
        previous.assistantResponseMessage.toolUses = [
          ...existingToolUses,
          ...item.assistantResponseMessage.toolUses,
        ];
      }
    } else {
      mergedHistory.push(item);
    }
  }

  // Ensure first message is user. Kiro API requires conversations to start
  // with a user message (fixes "Improperly formed request" for assistant-first).
  if (mergedHistory.length > 0 && mergedHistory[0].assistantResponseMessage) {
    const syntheticUserTurn = {
      userInputMessage: {
        content: "",
        modelId: model,
        origin,
      },
    };
    mergedHistory.unshift(syntheticUserTurn);
  }

  // Ensure assistant exists before toolResults. Kiro API validates that every
  // toolResults array has a preceding assistantResponseMessage with toolUses.
  // When the assistant message is missing (truncated conversation), we strip
  // the orphaned toolResults and convert them to text to preserve context.
  for (let i = 0; i < mergedHistory.length; i++) {
    const item = mergedHistory[i];
    if (!item.userInputMessage?.userInputMessageContext?.toolResults) continue;

    const prev = mergedHistory[i - 1];
    const hasPrecedingAssistant =
      prev?.assistantResponseMessage?.toolUses && prev.assistantResponseMessage.toolUses.length > 0;

    if (!hasPrecedingAssistant) {
      const toolResults = item.userInputMessage.userInputMessageContext.toolResults as Array<{
        toolUseId?: string;
        content?: Array<{ text?: string }>;
      }>;
      const toolResultTexts = toolResults
        .map((tr) => {
          const id = tr.toolUseId || "";
          const text = tr.content?.map((c) => c.text || "").join("\n") || "";
          return id ? `[Tool Result (${id})]\n${text}` : `[Tool Result]\n${text}`;
        })
        .join("\n\n");

      const originalContent = item.userInputMessage.content || "";
      item.userInputMessage.content = originalContent
        ? `${originalContent}\n\n${toolResultTexts}`
        : toolResultTexts;
      delete item.userInputMessage.userInputMessageContext.toolResults;

      if (Object.keys(item.userInputMessage.userInputMessageContext).length === 0) {
        delete item.userInputMessage.userInputMessageContext;
      }
    }
  }

  // Also check currentMessage for orphaned toolResults (not in history)
  if (currentMessage?.userInputMessage?.userInputMessageContext?.toolResults) {
    const lastHistory = mergedHistory[mergedHistory.length - 1];
    const hasPrecedingAssistant =
      lastHistory?.assistantResponseMessage?.toolUses &&
      lastHistory.assistantResponseMessage.toolUses.length > 0;

    if (!hasPrecedingAssistant) {
      const toolResults = currentMessage.userInputMessage.userInputMessageContext
        .toolResults as Array<{ toolUseId?: string; content?: Array<{ text?: string }> }>;
      const toolResultTexts = toolResults
        .map((tr) => {
          const id = tr.toolUseId || "";
          const text = tr.content?.map((c) => c.text || "").join("\n") || "";
          return id ? `[Tool Result (${id})]\n${text}` : `[Tool Result]\n${text}`;
        })
        .join("\n\n");

      const originalContent = currentMessage.userInputMessage.content || "";
      currentMessage.userInputMessage.content = originalContent
        ? `${originalContent}\n\n${toolResultTexts}`
        : toolResultTexts;
      delete currentMessage.userInputMessage.userInputMessageContext.toolResults;

      if (Object.keys(currentMessage.userInputMessage.userInputMessageContext).length === 0) {
        delete currentMessage.userInputMessage.userInputMessageContext;
      }
    }
  }

  // Ensure alternating roles by inserting synthetic assistant messages
  // between consecutive user turns that couldn't be merged.
  const alternatingHistory: typeof mergedHistory = [];
  for (const item of mergedHistory) {
    const last = alternatingHistory[alternatingHistory.length - 1];
    if (item.userInputMessage && last?.userInputMessage) {
      const syntheticAssistantTurn = {
        assistantResponseMessage: { content: "" },
      };
      alternatingHistory.push(syntheticAssistantTurn);
    }
    alternatingHistory.push(item);
  }

  return { history: alternatingHistory, currentMessage, toolsAttached, toolDocs };
}

/** Kiro's accepted reasoning-effort levels (`output_config.effort`). */
const KIRO_EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"];

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
 * Build Kiro payload from OpenAI format
 */
export function buildKiroPayload(model, body, stream, credentials, origin = KIRO_DEFAULT_ORIGIN) {
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
  let tools = body.tools || [];
  const maxTokens = body.max_tokens ?? body.max_completion_tokens ?? 32000;
  const temperature = body.temperature;
  const topP = body.top_p;

  // Kiro rejects history that references toolUses/toolResults without a tools
  // schema in userInputMessageContext. When callers omit body.tools but the
  // message history still contains assistant.tool_calls / role=tool turns,
  // synthesize a minimal tool schema from the tool names present in history
  // so Kiro accepts the request instead of returning `Improperly formed
  // request`. This preserves tool-call history and is a no-op when body.tools
  // is already populated.
  if (tools.length === 0) {
    const seen = new Set<string>();
    const synthesized: Array<Record<string, unknown>> = [];
    const pushName = (name: unknown) => {
      if (typeof name === "string" && name && !seen.has(name)) {
        seen.add(name);
        synthesized.push({
          type: "function",
          function: {
            name,
            description: `Tool: ${name}`,
            parameters: { type: "object", properties: {}, required: [] },
          },
        });
      }
    };
    for (const msg of messages) {
      if (msg?.role !== "assistant") continue;
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          pushName(tc?.function?.name || tc?.name);
        }
      }
      // Anthropic-style assistant blocks: content:[{type:"tool_use", name, ...}]
      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block?.type === "tool_use") {
            pushName(block.name);
          }
        }
      }
    }
    if (synthesized.length > 0) {
      tools = synthesized;
    }
  }

  const { history, currentMessage, toolsAttached, toolDocs } = convertMessages(
    messages,
    tools,
    normalizedModel,
    origin
  );

  const profileArn = credentials?.providerSpecificData?.profileArn || "";

  let finalContent = currentMessage?.userInputMessage?.content || "";
  const timestamp = new Date().toISOString();
  finalContent = `[Context: Current time is ${timestamp}]\n\n${finalContent}`;

  // Prepend documentation for tools whose description was relocated out of
  // `toolSpecification` (see buildKiroToolSpecs). Driven by convertMessages'
  // return value, not the message object, so the docs survive the tool-bearing
  // turn being moved into `history` on a multi-turn request.
  if (toolDocs) {
    finalContent = `# Tool Documentation\n\n${toolDocs}\n\n---\n\n${finalContent}`;
  }

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
          userInputMessageContext?: Record<string, unknown>;
        };
      };
      history: unknown[];
    };
    profileArn?: string;
    inferenceConfig?: {
      maxTokens?: number;
      temperature?: number;
      topP?: number;
    };
    additionalModelRequestFields?: {
      reasoning?: { effort: string };
      thinking?: { type: string };
      output_config?: { effort: string };
    };
  } = {
    conversationState: {
      chatTriggerType: "MANUAL",
      conversationId: uuidv4(),
      currentMessage: {
        userInputMessage: {
          content: finalContent,
          modelId: normalizedModel,
          origin,
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
  const requestedEffort = resolveKiroEffort(body) || (modelRequestsThinking ? "high" : "");
  const usesNativeReasoning = supportsKiroNativeReasoning(normalizedModel);
  const usesAdaptiveThinking = supportsKiroAdaptiveThinking(normalizedModel);
  const kiroEffort = usesNativeReasoning || usesAdaptiveThinking ? requestedEffort : "";
  const thinkingDisabled = usesAdaptiveThinking && body.thinking?.type === "disabled";
  if (kiroEffort || thinkingDisabled) {
    payload.additionalModelRequestFields = usesNativeReasoning
      ? { reasoning: { effort: kiroEffort } }
      : {
          ...(kiroEffort ? { output_config: { effort: kiroEffort } } : {}),
          ...(thinkingDisabled ? { thinking: { type: "disabled" } } : {}),
        };

    // Adaptive-only Claude models (Opus 4.7/4.8, Sonnet 5, Fable 5) reject a
    // non-default temperature / top_p with a 400 while thinking is active, so
    // strip both. Drop inferenceConfig entirely if nothing else remains.
    if (payload.inferenceConfig) {
      delete payload.inferenceConfig.temperature;
      delete payload.inferenceConfig.topP;
      if (Object.keys(payload.inferenceConfig).length === 0) {
        delete payload.inferenceConfig;
      }
    }
  }

  const context = payload.conversationState.currentMessage.userInputMessage.userInputMessageContext;
  if (Array.isArray(context?.tools)) {
    const { tools: sanitizedTools, nameMap } = sanitizeKiroTools(context.tools);
    context.tools = sanitizedTools;
    remapKiroToolHistory(payload.conversationState.history ?? [], nameMap);
    if (nameMap.size > 0) {
      (payload as typeof payload & { _toolNameMap: Record<string, string> })._toolNameMap =
        Object.fromEntries(nameMap);
    }

    // Lower Anthropic tool_choice only after sanitization, so forced long names
    // use the exact name declared in Kiro's tool schema.
    if (body.tool_choice !== undefined) {
      const choice = body.tool_choice;
      const choiceObject = choice && typeof choice === "object" && !Array.isArray(choice)
        ? choice as Record<string, unknown> : undefined;
      if (choiceObject?.disable_parallel_tool_use === true) {
        throw new Error("Unsupported tool_choice.disable_parallel_tool_use");
      }
      const choiceType = typeof choice === "string" ? choice : choiceObject?.type;
      if (choiceType === "auto") context.toolChoice = { auto: {} };
      else if (choiceType === "any") context.toolChoice = { any: {} };
      else if (choiceType === "none") context.toolChoice = { none: {} };
      else if (choiceType === "tool" && typeof choiceObject?.name === "string" && choiceObject.name) {
        const originalName = choiceObject.name;
        const sanitizedName = [...nameMap.entries()].find(([, original]) => original === originalName)?.[0] ?? originalName;
        const declared = (sanitizedTools as Array<Record<string, unknown>>).some((tool) =>
          (tool.toolSpecification as Record<string, unknown> | undefined)?.name === sanitizedName
        );
        if (!declared) throw new Error(`tool_choice references undeclared tool: ${originalName}`);
        context.toolChoice = { tool: { name: sanitizedName } };
      } else {
        throw new Error("Invalid tool_choice");
      }
    }
  } else if (body.tool_choice !== undefined) {
    const choice = body.tool_choice;
    const choiceObject = choice && typeof choice === "object" && !Array.isArray(choice)
      ? choice as Record<string, unknown> : undefined;
    if (choiceObject?.disable_parallel_tool_use === true) {
      throw new Error("Unsupported tool_choice.disable_parallel_tool_use");
    }
    const choiceType = typeof choice === "string" ? choice : choiceObject?.type;
    const target = payload.conversationState.currentMessage.userInputMessage;
    if (choiceType === "auto") (target.userInputMessageContext ??= {}).toolChoice = { auto: {} };
    else if (choiceType === "none") (target.userInputMessageContext ??= {}).toolChoice = { none: {} };
    else throw new Error("tool_choice requires declared tools");
  }

  return payload;
}

