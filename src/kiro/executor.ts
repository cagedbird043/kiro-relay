import {
  BaseExecutor,
  mergeUpstreamExtraHeaders,
  type ExecuteInput,
  type ExecutorLog,
  type ProviderCredentials,
} from "./base.ts";
import { KIRO_PROVIDER, contextLengthForModel } from "./config.ts";
import { v4 as uuidv4 } from "uuid";
import { refreshKiroToken, type KiroRefreshResult } from "./refresh.ts";
import {
  isExternalIdpAuthMethod,
  KIRO_EXTERNAL_IDP_TOKEN_TYPE_HEADER,
  KIRO_EXTERNAL_IDP_TOKEN_TYPE_VALUE,
} from "./externalIdp.ts";
import {
  splitInlineThinking,
  flushPendingThinking,
  type KiroThinkingState,
} from "./thinking.ts";
import { KiroStreamProtocolError, type KiroEvent } from "./events.ts";
import { ByteQueue, parseEventFrame, TEXT_ENCODER } from "./eventstream.ts";
import { kiroRuntimeHost, resolveKiroRuntimeRegion } from "./region.ts";
import {
  KIRO_TOOL_CALL_WRAPPER,
  appendBufferedKiroToolInput,
  encodeSse,
  getBufferedKiroToolInput,
  validateKiroToolCallWrapperInput,
  validateKiroToolName,
  validateKiroToolUse,
  type PendingKiroWrapperToolCall,
} from "./toolCallValidation.ts";

export { validateKiroToolUse } from "./toolCallValidation.ts";

type JsonRecord = Record<string, unknown>;

type UsageSummary = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
  estimated?: boolean;
};

type KiroStreamState = {
  endDetected: boolean;
  finishEmitted: boolean;
  startEmitted: boolean;
  stopSeen: boolean;
  stopReason?: unknown;
  stopDetails?: JsonRecord;
  hasToolCalls: boolean;
  toolCallIndex: number;
  toolCalls: Map<string, PendingKiroWrapperToolCall & { wrapper: boolean; toolIndex?: number; published?: boolean }>;
  generatedToolIdCounter: number;
  toolNameMap: Map<string, string>;
  invalidToolCall?: boolean;
  estimatedOutputChars: number;
  contextUsagePercentage?: number;
  hasContextUsage?: boolean;
  hasMeteringEvent?: boolean;
  usage?: Partial<UsageSummary>;
  hasReasoningContent?: boolean;
  reasoningChunkCount?: number;
  // Inline-thinking splitter state (populated only when thinkingExpected=true).
  thinking?: KiroThinkingState;
};

function normalizeKiroStopReason(raw: unknown, hasToolCalls: boolean): Extract<KiroEvent, { type: "stop" }>["reason"] {
  switch (typeof raw === "string" ? raw.toLowerCase() : raw) {
    case undefined: case "stop": case "end_turn": return hasToolCalls ? "tool_use" : "end_turn";
    case "tool_calls": case "tool_use": return "tool_use";
    case "length": case "max_tokens": case "max_output_tokens": return "max_tokens";
    case "stop_sequence": return "stop_sequence";
    case "refusal": case "content_filter": case "guardrail_intervened": return "refusal";
    case "pause_turn": return "pause_turn";
    default: throw new KiroStreamProtocolError("Unsupported Kiro stop reason", "missing_terminal");
  }
}

function buildKiroFinishChunk(
  state: KiroStreamState,
  responseId: string,
  created: number,
  model: string,
  includeUsage: boolean
): JsonRecord {
  const finishChunk: JsonRecord = {
    id: responseId,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: normalizeKiroStopReason(state.stopReason, state.hasToolCalls),
        ...(state.stopDetails ? { kiro_stop_details: state.stopDetails } : {}),
      },
    ],
  };

  if (includeUsage && state.usage) {
    finishChunk.usage = state.usage;
  }

  return finishChunk;
}


/**
 * Synthesize usage when Kiro reports only contextUsagePercentage. The
 * percentage is converted directly to Anthropic input tokens using the model's
 * context window; output tokens remain a text-length estimate.
 *
 * `contextLength` is the window reported by Kiro's live catalog. When it is
 * absent the static MODEL_CONTEXT_LENGTHS table decides, which fails closed on
 * unknown ids. It is resolved lazily so a response that never reported a
 * percentage never needs a context window at all.
 */
function ensureKiroUsage(state: KiroStreamState, model: string, contextLength?: number | null) {
  const usage = state.usage ?? {};
  const estimated = usage.prompt_tokens === undefined || usage.completion_tokens === undefined;
  const completionTokens = usage.completion_tokens ?? (state.estimatedOutputChars ? Math.max(1, Math.floor(state.estimatedOutputChars / 4)) : 0);
  const promptTokens = usage.prompt_tokens ?? (state.contextUsagePercentage
    ? Math.round(state.contextUsagePercentage / 100 *
        (typeof contextLength === "number" && contextLength > 0 ? contextLength : contextLengthForModel(model)))
    : 0);
  state.usage = { ...usage, prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens, ...(estimated ? { estimated: true } : {}) };
}

/**
 * Resolve the RUNTIME AWS region for a Kiro/CodeWhisperer connection.
 *
 * The runtime region is the region of the Amazon Q Developer profile (embedded in the
 * profileArn — always us-east-1 or eu-central-1), NOT the IAM Identity Center / OIDC token
 * region. An enterprise IdC instance may live in eu-north-1 (or any region), but the Q Developer
 * profile that serves generateAssistantResponse only exists in us-east-1 / eu-central-1, so a
 * runtime call must target the profileArn's region — routing to q.{idcRegion}.amazonaws.com
 * (a host that does not exist) is what caused "no limits + 502 on every request". Delegates to
 * the shared resolver (profileArn region → valid stored profile region → us-east-1). The IdC
 * token region is used only for oidc.{region} token mint/refresh, elsewhere.
 */
export function resolveKiroRegion(
  credentials: { providerSpecificData?: unknown } | null | undefined
): string {
  return resolveKiroRuntimeRegion(
    (credentials?.providerSpecificData || {}) as { region?: unknown; profileArn?: unknown }
  );
}

// Re-exported from the shared region module so existing importers (and tests) that pull
// kiroRuntimeHost from this executor keep working.
export { kiroRuntimeHost };

/**
 * Status codes for which trying the next candidate endpoint may succeed where the
 * current one failed (auth/profile mismatch, not a payload problem). Mirrors
 * 9router's KIRO_ENDPOINT_FALLBACK_STATUSES — a 400 (malformed body) is deliberately
 * excluded since resending the same body to another host cannot fix it.
 */
const KIRO_ENDPOINT_FALLBACK_STATUSES = new Set([401, 403, 404]);

/**
 * KiroExecutor - Executor for Kiro AI (AWS CodeWhisperer)
 * Uses AWS CodeWhisperer streaming API with AWS EventStream binary format
 */
export class KiroExecutor extends BaseExecutor {
  constructor(providerId = "kiro") {
    super(providerId, KIRO_PROVIDER);
  }

  buildHeaders(credentials: ProviderCredentials, stream = true): Record<string, string> {
    void stream;
    const headers: Record<string, string> = {
      ...this.config.headers,
      "Amz-Sdk-Request": "attempt=1; max=3",
      "Amz-Sdk-Invocation-Id": uuidv4(),
      "x-amzn-bedrock-cache-control": "enable",
      "anthropic-beta": "prompt-caching-2024-07-31",
    };

    const authMethod =
      typeof credentials.providerSpecificData?.authMethod === "string"
        ? credentials.providerSpecificData.authMethod
        : undefined;
    const isApiKey = authMethod === "api_key";
    const token = isApiKey
      ? credentials.apiKey || credentials.accessToken
      : credentials.accessToken;

    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
      // Long-lived Kiro/CodeWhisperer API keys authenticate with `tokentype: API_KEY`.
      if (isApiKey) headers["tokentype"] = "API_KEY";

      // Enterprise / Microsoft Entra "Your organization" (external_idp) logins send an
      // org-IdP-issued access token. CodeWhisperer only binds it to the Amazon Q Developer
      // profile when the request carries `TokenType: EXTERNAL_IDP`; without it every call
      // returns `ValidationException: Invalid ARN <clientId>` (the service falls back to the
      // token's client id as the resource ARN). AWS SSO (Builder ID / IDC) and social tokens
      // must NOT send this header, so it is gated on the persisted authMethod.
      if (isExternalIdpAuthMethod(authMethod)) {
        headers[KIRO_EXTERNAL_IDP_TOKEN_TYPE_HEADER] = KIRO_EXTERNAL_IDP_TOKEN_TYPE_VALUE;
      }
    }

    return headers;
  }

  transformRequest(model: string, body: unknown, stream: boolean, credentials: unknown): Record<string, unknown> {
    void stream;
    void credentials;
    const b = body as Record<string, unknown>;

    // Kiro API is strict and rejects any unknown top-level fields (like 'tools', 'stream', 'model', etc.)
    // We only preserve the fields specifically built by the openai-to-kiro translator.
    const kiroPayload: Record<string, unknown> = {};
    if (b.conversationState !== undefined) kiroPayload.conversationState = b.conversationState;
    if (b.profileArn !== undefined) kiroPayload.profileArn = b.profileArn;
    if (b.inferenceConfig !== undefined) kiroPayload.inferenceConfig = b.inferenceConfig;
    // 按模型目录声明的路径透传 effort，以及调用方显式要求的思考开关。
    if (b.additionalModelRequestFields !== undefined)
      kiroPayload.additionalModelRequestFields = b.additionalModelRequestFields;

    // Fallback: if somehow conversationState isn't there, return the rest without model
    // (for backward compatibility if something else bypasses the translator)
    if (!kiroPayload.conversationState) {
      const { model: _model, ...rest } = b;
      return rest;
    }

    return kiroPayload;
  }

  /**
   * Custom execute for Kiro - handles AWS EventStream binary response
   */
  async execute({
    model,
    body,
    stream,
    credentials,
    signal,
    log,
    upstreamExtraHeaders,
    contextLength,
  }: ExecuteInput) {
    // Route to the region-specific CodeWhisperer/Amazon Q endpoint. Enterprise IAM Identity
    // Center accounts (e.g. eu-central-1) are rejected by the default us-east-1 host; only the
    // regional endpoint accepts the region-bound token + profileArn.
    const region = resolveKiroRegion(credentials);
    const regionalUrl = `${kiroRuntimeHost(region)}/generateAssistantResponse`;

    // The Kiro IDE's own branded gateway (runtime.*.kiro.dev) only exists for
    // us-east-1 and only accepts Kiro OIDC/social tokens — it rejects
    // TokenType=API_KEY and external-IdP/IdC SSO tokens outright (403 "bearer
    // token invalid"), so those auth methods go straight to the region-resolved
    // CodeWhisperer/Amazon Q surface (mirrors 9router's getOrderedBaseUrls in
    // open-sse/executors/kiro.js). For everything else, try the branded gateway
    // first — it is the surface the native Kiro IDE itself talks to — and fall
    // back to the raw AWS host on an auth/profile-shaped failure.
    const authMethod =
      typeof credentials.providerSpecificData?.authMethod === "string"
        ? credentials.providerSpecificData.authMethod
        : undefined;
    const isCodeWhispererOnly =
      authMethod === "api_key" || authMethod === "idc" || isExternalIdpAuthMethod(authMethod);
    const candidateUrls =
      region === "us-east-1" && !isCodeWhispererOnly
        ? ["https://runtime.us-east-1.kiro.dev/generateAssistantResponse", regionalUrl]
        : [regionalUrl];

    const headers = this.buildHeaders(credentials, stream);
    mergeUpstreamExtraHeaders(headers, upstreamExtraHeaders);
    const transformedBody = await this.transformRequest(model, body, stream, credentials);
    const requestBody = JSON.stringify(transformedBody);

    let response!: Response;
    let url = candidateUrls[0];
    for (let i = 0; i < candidateUrls.length; i++) {
      url = candidateUrls[i];
      response = await fetch(url, {
        method: "POST",
        headers,
        body: requestBody,
        signal,
      });
      const hasFallback = i + 1 < candidateUrls.length;
      if (response.ok || !hasFallback || !KIRO_ENDPOINT_FALLBACK_STATUSES.has(response.status)) {
        break;
      }
      await response.body?.cancel().catch(() => undefined);
    }

    if (!response.ok) {
      return { response, url, headers, transformedBody };
    }

    // For Kiro, we need to transform the binary EventStream to SSE.
    // Create a TransformStream to convert binary to SSE text.
    //
    // When the user enabled thinking, Claude on Kiro streams its reasoning
    // **inline** as `<thinking>…</thinking>` blocks inside
    // `assistantResponseEvent.content` rather than as separate
    // `reasoningContentEvent` frames. We pass a hint so the transform stream
    // can split that inline reasoning into the OpenAI `delta.reasoning_content`
    // channel.
    const tb = transformedBody as Record<string, unknown>;
    const userInputMessage = (
      (tb?.conversationState as Record<string, unknown>)?.currentMessage as Record<string, unknown>
    )?.userInputMessage as Record<string, unknown>;
    const userContent = (userInputMessage?.content as string) || "";
    const declaredTools = (userInputMessage?.userInputMessageContext as Record<string, unknown>)?.tools;
    const thinkingExpected = userContent.includes("<thinking_mode>enabled</thinking_mode>");
    const rawToolNameMap =
      body && typeof body === "object" && "_toolNameMap" in body ? body._toolNameMap : undefined;
    const toolNameMap = new Map(
      rawToolNameMap && typeof rawToolNameMap === "object"
        ? Object.entries(rawToolNameMap).filter(
            (entry): entry is [string, string] => typeof entry[1] === "string"
          )
        : []
    );
    // A client tool that happens to be named `tool_call` (e.g. Hermes' deferred-tool
    // dispatcher) owns its own schema; only an undeclared `tool_call` is Kiro's wrapper.
    const clientDeclaresToolCall = Array.isArray(declaredTools) && declaredTools.some((tool) => {
      const kiroName = (tool as Record<string, Record<string, unknown>>)?.toolSpecification?.name;
      return typeof kiroName === "string" && (toolNameMap.get(kiroName) || kiroName) === KIRO_TOOL_CALL_WRAPPER;
    });
    const transformedResponse = this.transformEventStreamToSSE(response, model, {
      thinkingExpected,
      toolNameMap,
      contextLength,
      clientDeclaresToolCall,
    });

    return { response: transformedResponse, url, headers, transformedBody };
  }

  /**
   * Transform AWS EventStream binary response to SSE text stream.
   * Using TransformStream instead of ReadableStream.pull() to avoid Workers timeout.
   *
   * @param response        Upstream raw fetch response (binary EventStream).
   * @param model           Logical model id (kept in OpenAI chunks for clients).
   * @param opts
   * @param opts.thinkingExpected  When true, scan inbound
   *   `assistantResponseEvent.content` for inline `<thinking>…</thinking>`
   *   blocks and split them into the OpenAI `delta.reasoning_content` channel.
   *   Required for Claude on Kiro when `<thinking_mode>enabled</thinking_mode>`
   *   is in the system prompt, because Kiro streams reasoning inline rather
   *   than as separate `reasoningContentEvent` frames.
   * @param opts.contextLength  Context window from Kiro's live catalog, used to
   *   scale the synthesized usage estimate. Omitted -> the static
   *   MODEL_CONTEXT_LENGTHS table decides and fails closed on unknown ids.
   * @param opts.clientDeclaresToolCall  The request declared its own `tool_call` tool, so
   *   upstream calls to it are ordinary tool calls rather than Kiro's nested MCP wrapper.
   */
  transformEventStreamToSSE(
    response: Response,
    model: string,
    opts: {
      thinkingExpected?: boolean;
      toolNameMap?: Map<string, string>;
      contextLength?: number | null;
      clientDeclaresToolCall?: boolean;
    } = {}
  ) {
    const thinkingExpected = !!opts.thinkingExpected;
    const buffer = new ByteQueue();
    let chunkIndex = 0;
    const responseId = `chatcmpl-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);
    const state: KiroStreamState = {
      endDetected: false,
      finishEmitted: false,
      startEmitted: false,
      stopSeen: false,
      hasToolCalls: false,
      toolCallIndex: 0,
      toolCalls: new Map(),
      estimatedOutputChars: 0,
      generatedToolIdCounter: 0,
      toolNameMap: opts.toolNameMap || new Map(),
      hasReasoningContent: false,
      reasoningChunkCount: 0,
      thinking: thinkingExpected ? { thinkingMode: false, pendingTag: "" } : undefined,
    };

    const getToolCallId = (toolUse: JsonRecord): string => {
      if (typeof toolUse.toolUseId === "string" && toolUse.toolUseId) {
        return toolUse.toolUseId;
      }
      state.generatedToolIdCounter += 1;
      return `call_${created}_${state.generatedToolIdCounter}`;
    };

    const emitToolCallStart = (
      controller: TransformStreamDefaultController,
      toolCallId: string,
      toolName: string,
      toolIndex: number
    ) => {
      const startChunk: JsonRecord = {
        id: responseId,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {
              ...(chunkIndex === 0 ? { role: "assistant" } : {}),
              tool_calls: [
                {
                  index: toolIndex,
                  id: toolCallId,
                  type: "function",
                  function: { name: toolName, arguments: "" },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
      chunkIndex += 1;
      controller.enqueue(encodeSse(`data: ${JSON.stringify(startChunk)}\n\n`));
    };

    const emitToolCallArguments = (
      controller: TransformStreamDefaultController,
      toolIndex: number,
      argumentsStr: string
    ) => {
      const argsChunk: JsonRecord = {
        id: responseId,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [{ index: toolIndex, function: { arguments: argumentsStr } }],
            },
            finish_reason: null,
          },
        ],
      };
      chunkIndex += 1;
      controller.enqueue(encodeSse(`data: ${JSON.stringify(argsChunk)}\n\n`));
    };

    const failInvalidToolCall = (controller: TransformStreamDefaultController, message: string) => {
      const error = {
        error: {
          message,
          type: "invalid_request_error",
          code: "invalid_kiro_tool_call",
        },
      };
      state.invalidToolCall = true;
      state.finishEmitted = true;
      controller.enqueue(encodeSse(`data: ${JSON.stringify(error)}\n\n`));
      controller.enqueue(encodeSse("data: [DONE]\n\n"));
      controller.terminate();
    };

    const flushToolCalls = (controller: TransformStreamDefaultController): boolean => {
      for (const toolCall of state.toolCalls.values()) {
        if (toolCall.published) continue;
        const toolInput = getBufferedKiroToolInput(toolCall);
        if (toolCall.wrapper) {
          try {
            validateKiroToolCallWrapperInput(toolInput);
          } catch (error) {
            failInvalidToolCall(controller, error instanceof Error ? error.message : String(error));
            return false;
          }
          toolCall.toolIndex = state.toolCallIndex++;
          emitToolCallStart(controller, toolCall.toolCallId, toolCall.toolName, toolCall.toolIndex);
        }
        if (toolCall.wrapper || toolCall.inputKind === "object") {
          const args = typeof toolInput === "string" ? toolInput : JSON.stringify(toolInput ?? {});
          state.estimatedOutputChars += args.length;
          emitToolCallArguments(controller, toolCall.toolIndex!, args);
        }
        toolCall.published = true;
      }
      return true;
    };

    const transformStream = new TransformStream(
      {
        async transform(chunk, controller) {
          buffer.push(chunk);

          // Parse events from buffer
          while (buffer.length >= 16) {
            const totalLength = buffer.peekUint32BE(0);

            if (!totalLength || totalLength < 16 || totalLength > buffer.length) break;

            const eventData = buffer.read(totalLength);
            if (!eventData) break;

            const event = parseEventFrame(eventData);
            if (!event) continue;

            // Emit a role-only start chunk on the FIRST successfully-parsed AWS
            // EventStream frame. CodeWhisperer sends framing/metadata events before
            // the first content token, and on large/agentic contexts the gap before
            // that first `assistantResponseEvent` can be many seconds. The backend
            // stream-readiness gate (ensureStreamReadiness) holds the ENTIRE response
            // from the client until it observes a useful SSE frame, so without an
            // early frame the client sees a frozen connection for that whole window
            // (up to STREAM_READINESS_TIMEOUT_MS — 180s as configured by VibeProxy),
            // then a burst — the "minutes instead of seconds, not streaming" symptom.
            // A role-only `chat.completion.chunk` is a non-ping structured payload, so
            // it satisfies hasStreamReadinessSignal and hands the stream off
            // immediately. Mirrors the early lifecycle frame other executors already
            // emit (Claude message_start / OpenAI response.created). The downstream
            // idle timeout still guards genuine post-start stalls.
            if (!state.startEmitted) {
              state.startEmitted = true;
              const startChunk: JsonRecord = {
                id: responseId,
                object: "chat.completion.chunk",
                created,
                model,
                choices: [
                  {
                    index: 0,
                    delta: { role: "assistant" },
                    finish_reason: null,
                  },
                ],
              };
              chunkIndex++;
              controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(startChunk)}\n\n`));
            }

            const eventType = event.headers[":event-type"] || "";
            if (event.headers[":message-type"] === "exception" || event.headers[":message-type"] === "error") {
              const value = event.headers[":exception-type"] ?? event.headers[":error-code"] ?? eventType;
              const code = /^[A-Za-z][A-Za-z0-9_]{0,79}$/.test(value) ? value : "upstream_error";
              throw new Error(`Kiro upstream stream failed (${code})`);
            }

            // Output estimates count only emitted text and final tool snapshots.
            if (!state.contextUsagePercentage) state.contextUsagePercentage = 0;

            // Native reasoning frames. Verified against the live CodeWhisperer
            // stream (2026-07): with adaptive thinking enabled (via
            // additionalModelRequestFields), Kiro streams reasoning as a dedicated
            // `reasoningContentEvent` frame carrying `{ text, signature }` — NOT
            // inline `<thinking>` tags and NOT `assistantResponseEvent`. Some
            // models/variants instead use a `reasoningText` object or a flat
            // `{ text }` (cf. javargasm/pi-kiro `src/event-parser.ts`). OmniRoute
            // had no handler for this event, so the reasoning was silently dropped;
            // route it to the OpenAI `reasoning_content` channel.
            {
              const rp = event.payload as Record<string, unknown> | undefined;
              const rt = rp?.reasoningText;
              if (eventType === "reasoningContentEvent" || rt !== undefined) {
                let nativeReasoning = "";
                if (rt && typeof rt === "object") {
                  const rto = rt as { text?: unknown; Text?: unknown };
                  nativeReasoning =
                    typeof rto.text === "string"
                      ? rto.text
                      : typeof rto.Text === "string"
                        ? rto.Text
                        : "";
                } else if (typeof rt === "string") {
                  nativeReasoning = rt;
                } else if (typeof rp?.text === "string") {
                  nativeReasoning = rp.text as string;
                }
                const nestedSignature = rt && typeof rt === "object" && "signature" in rt ? rt.signature : undefined;
                const signature = typeof rp?.signature === "string" ? rp.signature
                  : typeof nestedSignature === "string" ? nestedSignature : undefined;
                const visibleNativeReasoning = nativeReasoning;
                const redacted = typeof rp?.redactedContent === "string" ? rp.redactedContent : undefined;
                if (visibleNativeReasoning || signature || redacted) {
                  if (visibleNativeReasoning) state.hasReasoningContent = true;
                  state.estimatedOutputChars += visibleNativeReasoning.length;
                  const reasoningDelta: JsonRecord = {
                    ...(visibleNativeReasoning
                      ? { reasoning_content: visibleNativeReasoning }
                      : {}),
                    ...(signature ? { reasoning_signature: signature } : {}),
                    ...(redacted ? { reasoning_redacted: redacted } : {}),
                  };
                  if ((state.reasoningChunkCount ?? 0) === 0 && chunkIndex === 0) {
                    reasoningDelta.role = "assistant";
                  }
                  const chunk: JsonRecord = {
                    id: responseId,
                    object: "chat.completion.chunk",
                    created,
                    model,
                    choices: [{ index: 0, delta: reasoningDelta, finish_reason: null }],
                  };
                  chunkIndex++;
                  if (visibleNativeReasoning) {
                    state.reasoningChunkCount = (state.reasoningChunkCount ?? 0) + 1;
                  }
                  controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
                }
                // Consume the reasoning frame (incl. signature-only) so it never
                // falls through to the content handlers below.
                continue;
              }
            }

            // Handle assistantResponseEvent
            if (eventType === "assistantResponseEvent") {
              const content =
                typeof event.payload?.content === "string" ? event.payload.content : "";
              if (!content) {
                continue;
              }
              if (!thinkingExpected) state.estimatedOutputChars += content.length;

              if (thinkingExpected && state.thinking) {
                // Claude on Kiro emits reasoning inline as `<thinking>…</thinking>`
                // when `<thinking_mode>enabled</thinking_mode>` is in the system prompt.
                // Split it into the OpenAI `reasoning_content` channel so downstream
                // consumers see the same shape they would get from a native reasoning model.
                const thinkingState = state.thinking;
                const onContent = (text: string) => {
                  if (!text) return;
                  state.estimatedOutputChars += text.length;
                  const chunk: JsonRecord = {
                    id: responseId,
                    object: "chat.completion.chunk",
                    created,
                    model,
                    choices: [
                      {
                        index: 0,
                        delta:
                          chunkIndex === 0
                            ? { role: "assistant", content: text }
                            : { content: text },
                        finish_reason: null,
                      },
                    ],
                  };
                  chunkIndex++;
                  controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
                };
                const onReasoning = (reasoning: string) => {
                  if (!reasoning) return;
                  state.estimatedOutputChars += reasoning.length;
                  state.hasReasoningContent = true;
                  const reasoningDelta: JsonRecord =
                    (state.reasoningChunkCount ?? 0) === 0 && chunkIndex === 0
                      ? { role: "assistant", reasoning_content: reasoning }
                      : { reasoning_content: reasoning };
                  const chunk: JsonRecord = {
                    id: responseId,
                    object: "chat.completion.chunk",
                    created,
                    model,
                    choices: [{ index: 0, delta: reasoningDelta, finish_reason: null }],
                  };
                  chunkIndex++;
                  state.reasoningChunkCount = (state.reasoningChunkCount ?? 0) + 1;
                  controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
                };
                splitInlineThinking(thinkingState, content, onContent, onReasoning);
              } else {
                const chunk: JsonRecord = {
                  id: responseId,
                  object: "chat.completion.chunk",
                  created,
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: chunkIndex === 0 ? { role: "assistant", content } : { content },
                      finish_reason: null,
                    },
                  ],
                };
                chunkIndex++;
                controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
              }
            }

            // Handle codeEvent
            if (eventType === "codeEvent" && typeof event.payload?.content === "string" && event.payload.content) {
              state.estimatedOutputChars += event.payload.content.length;
              const chunk: JsonRecord = {
                id: responseId,
                object: "chat.completion.chunk",
                created,
                model,
                choices: [
                  {
                    index: 0,
                    delta: { content: event.payload.content },
                    finish_reason: null,
                  },
                ],
              };
              chunkIndex++;
              controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            }

            // Handle toolUseEvent
            if (eventType === "toolUseEvent" && event.payload) {
              state.hasToolCalls = true;
              const toolUse = event.payload;
              const toolUses = Array.isArray(toolUse) ? toolUse : [toolUse];

              for (const rawToolUse of toolUses) {
                const singleToolUse = rawToolUse as JsonRecord;
                let toolName: string;
                try {
                  toolName = validateKiroToolName(singleToolUse);
                  toolName = state.toolNameMap.get(toolName) || toolName;
                } catch (error) {
                  failInvalidToolCall(
                    controller,
                    error instanceof Error ? error.message : String(error)
                  );
                  return;
                }

                const toolCallId = getToolCallId(singleToolUse);
                const toolInput = singleToolUse.input;

                const wrapper = toolName === KIRO_TOOL_CALL_WRAPPER && !opts.clientDeclaresToolCall;
                let toolCall = state.toolCalls.get(toolCallId);
                if (toolCall && (toolCall.wrapper !== wrapper || toolCall.toolName !== toolName || toolCall.published)) {
                  failInvalidToolCall(controller, "Invalid Kiro tool_call payload: inconsistent tool fragments");
                  return;
                }
                if (!toolCall) {
                  toolCall = { toolCallId, toolName, wrapper };
                  state.toolCalls.set(toolCallId, toolCall);
                  if (!wrapper) {
                    toolCall.toolIndex = state.toolCallIndex++;
                    emitToolCallStart(controller, toolCallId, toolName, toolCall.toolIndex);
                  }
                }
                try {
                  appendBufferedKiroToolInput(toolCall, toolInput);
                } catch (error) {
                  failInvalidToolCall(controller, error instanceof Error ? error.message : String(error));
                  return;
                }
                if (!wrapper && typeof toolInput === "string") {
                  state.estimatedOutputChars += toolInput.length;
                  emitToolCallArguments(controller, toolCall.toolIndex!, toolInput);
                }
              }
            }

            // Handle messageStopEvent
            if (eventType === "messageStopEvent") {
              if (!flushToolCalls(controller)) return;
              state.stopSeen = true;
              if (event.payload && Object.hasOwn(event.payload, "stopReason")) state.stopReason = event.payload.stopReason;
            }

            // Handle contextUsageEvent to extract contextUsagePercentage
            if (eventType === "contextUsageEvent") {
              const contextUsage =
                typeof event.payload?.contextUsagePercentage === "number"
                  ? event.payload.contextUsagePercentage
                  : 0;
              if (contextUsage <= 0) {
                continue;
              }
              state.contextUsagePercentage = contextUsage;
              // Mark that we received context usage event
              state.hasContextUsage = true;
            }

            // Handle meteringEvent - mark that we received it
            if (eventType === "meteringEvent") {
              state.hasMeteringEvent = true;
            }

            if (eventType === "metricsEvent" || eventType === "metadataEvent") {
              const payload = event.payload ?? {};
              const metadata = payload.metadataEvent && typeof payload.metadataEvent === "object" ? payload.metadataEvent as JsonRecord : payload;
              const rawMetrics = metadata.tokenUsage ?? metadata.usage ?? metadata.metricsEvent ?? metadata;
              if (rawMetrics && typeof rawMetrics === "object") {
                const metrics = rawMetrics as JsonRecord;
                const readNumber = (...values: unknown[]): number | undefined => values.find((value): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0);
                const input = readNumber(metrics.uncachedInputTokens, metrics.inputTokens, metrics.prompt_tokens);
                const output = readNumber(metrics.outputTokens, metrics.completion_tokens);
                const read = readNumber(metrics.cacheReadInputTokens, metrics.cacheReadTokens, metrics.cache_read_input_tokens);
                const write = readNumber(metrics.cacheWriteInputTokens, metrics.cacheCreationTokens, metrics.cache_creation_input_tokens);
                if (input !== undefined || output !== undefined || read !== undefined || write !== undefined) {
                  state.usage = { ...state.usage,
                    ...(input === undefined ? {} : { prompt_tokens: input }),
                    ...(output === undefined ? {} : { completion_tokens: output }),
                    ...(read === undefined ? {} : { cache_read_input_tokens: read }),
                    ...(write === undefined ? {} : { cache_creation_input_tokens: write }),
                  };
                }
              }
              if (Object.hasOwn(metadata, "stopReason")) {
                state.stopSeen = true;
                state.stopReason = metadata.stopReason;
                if (metadata.stopDetails && typeof metadata.stopDetails === "object") {
                  const refusal = (metadata.stopDetails as JsonRecord).refusal;
                  if (refusal && typeof refusal === "object") {
                    const fields = Object.entries(refusal).filter(([key, value]) => ["category", "explanation", "recommendedModel"].includes(key) && typeof value === "string");
                    state.stopDetails = { refusal: Object.fromEntries(fields) };
                  }
                }
              }
            }

          }

        },

        flush(controller) {
          // 物理结束时确认终止证据，保留计量帧之后到达的元数据。
          if (buffer.length > 0 || (!state.stopSeen && !state.hasMeteringEvent && !state.finishEmitted)) {
            throw new KiroStreamProtocolError();
          }
          if (!flushToolCalls(controller)) return;
          if (state.invalidToolCall) return;

          // Drain any pending inline-thinking tag fragment so we don't drop
          // trailing characters when the stream ends mid-tag (e.g. `<thi`).
          if (thinkingExpected && state.thinking) {
            const thinkingState = state.thinking;
            flushPendingThinking(
              thinkingState,
              (text) => {
                if (!text) return;
                state.estimatedOutputChars += text.length;
                const chunk: JsonRecord = {
                  id: responseId,
                  object: "chat.completion.chunk",
                  created,
                  model,
                  choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
                };
                chunkIndex++;
                controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
              },
              (reasoning) => {
                if (!reasoning) return;
                state.estimatedOutputChars += reasoning.length;
                const chunk: JsonRecord = {
                  id: responseId,
                  object: "chat.completion.chunk",
                  created,
                  model,
                  choices: [
                    { index: 0, delta: { reasoning_content: reasoning }, finish_reason: null },
                  ],
                };
                chunkIndex++;
                controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
              }
            );
          }

          // Emit finish chunk if not already sent
          if (!state.finishEmitted) {
            state.finishEmitted = true;
            ensureKiroUsage(state, model, opts.contextLength);
            const finishChunk = buildKiroFinishChunk(state, responseId, created, model, true);
            controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(finishChunk)}\n\n`));
          }

          // Send final done message
          controller.enqueue(TEXT_ENCODER.encode("data: [DONE]\n\n"));
        },
      },
      { highWaterMark: 16384 },
      { highWaterMark: 16384 }
    );

    const responseBody =
      response.body ??
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.close();
        },
      });
    const transformedStream = transformStream.readable;
    // pipeTo 将下游取消传给 HTTP 响应体；错误仍由 readable 交付消费者。
    void responseBody.pipeTo(transformStream.writable).catch(() => undefined);

    return new Response(transformedStream, {
      status: response.status,
      statusText: response.statusText,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }

  async refreshCredentials(credentials: ProviderCredentials, log?: ExecutorLog | null): Promise<KiroRefreshResult> {
    return refreshKiroToken(credentials.refreshToken, credentials.providerSpecificData, log);
  }
}

export default KiroExecutor;
