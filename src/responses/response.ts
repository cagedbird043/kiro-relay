import { randomUUID } from "node:crypto";
import type { Response as OpenAIResponse, ResponseStreamEvent, ResponseUsage } from "openai/resources/responses/responses";
import { readKiroEvents, type KiroEvent } from "../kiro/events.ts";
import { encodeReasoning } from "../kiro/reasoning.ts";

const id = (prefix: string) => `${prefix}_${randomUUID().replaceAll("-", "")}`;
type Usage = Omit<ResponseUsage, "output_tokens_details"> & {
  output_tokens_details: Partial<ResponseUsage["output_tokens_details"]>;
  estimated?: true;
};
export type RelayResponse = Omit<OpenAIResponse, "usage"> & { usage: Usage | null; stop_details?: Record<string, unknown> };
export type ResponseToolMetadata = Readonly<Record<string, { name: string; namespace?: string; custom: boolean }>>;
export type ResponsesOptions = Readonly<{ includeEncryptedContent?: boolean }>;
type Reasoning = { kind: "reasoning"; id: string; index: number; text: string; signature?: string; redacted?: string; summaryStarted: boolean };
type Message = { kind: "message"; id: string; index: number; text: string };
type Call = { kind: "tool"; id: string; index: number; callId: string; name: string; args: string; metadata?: { name: string; namespace?: string; custom: boolean } };
type Entry = Reasoning | Message | Call;
type Emit = (name: string, data: Record<string, unknown>) => void;
type State = {
  responseId: string; model: string; entries: Entry[]; calls: Map<number, Call>;
  activeReasoning?: Reasoning; lastReasoning?: Reasoning; activeMessage?: Message;
  responseTools: ResponseToolMetadata; includeEncryptedContent: boolean;
  usage: Usage | null; reason: string; status: "completed" | "incomplete" | "failed"; finalized: boolean;
  error: OpenAIResponse["error"]; stopDetails?: Record<string, unknown>;
};

function initial(model: string, responseTools: ResponseToolMetadata, options: ResponsesOptions): State {
  return { responseId: id("resp"), model, entries: [], calls: new Map(), responseTools, includeEncryptedContent: options.includeEncryptedContent === true, usage: null, reason: "end_turn", status: "completed", finalized: false, error: null };
}
function usageOf(event: Extract<KiroEvent, { type: "usage" }>): Usage {
  const cached = event.cacheReadInputTokens ?? 0;
  const written = event.cacheCreationInputTokens ?? 0;
  const input = event.inputTokens + cached + written;
  return { input_tokens: input, output_tokens: event.outputTokens, total_tokens: input + event.outputTokens, input_tokens_details: { cached_tokens: cached, cache_write_tokens: written }, output_tokens_details: {}, ...(event.estimated ? { estimated: true as const } : {}) };
}
function customInput(argumentsText: string): string {
  try {
    const parsed: unknown = JSON.parse(argumentsText);
    if (parsed && typeof parsed === "object" && "input" in parsed && typeof parsed.input === "string") return parsed.input;
  } catch {
    // 非 JSON 自定义工具输入按原文交付。
  }
  return argumentsText;
}
function reasoningSummary(entry: Reasoning): Array<{ type: "summary_text"; text: string }> {
  return entry.text ? [{ type: "summary_text", text: entry.text }] : [];
}
function outputItem(s: State, entry: Entry, pending = false): OpenAIResponse["output"][number] & { status?: "in_progress" | "completed" | "incomplete" } {
  const status = pending ? "in_progress" : s.status === "completed" ? "completed" : "incomplete";
  if (entry.kind === "reasoning") {
    const content = entry.redacted !== undefined ? { redactedContent: entry.redacted } : { reasoningText: { text: entry.text, ...(entry.signature !== undefined ? { signature: entry.signature } : {}) } };
    return { id: entry.id, type: "reasoning", content: [], summary: pending ? [] : reasoningSummary(entry), status,
      ...(s.includeEncryptedContent ? { encrypted_content: pending ? null : encodeReasoning(s.model, content) } : {}) };
  }
  if (entry.kind === "message") return { id: entry.id, type: "message", role: "assistant", status, content: pending ? [] : [{ type: "output_text", text: entry.text, annotations: [] }] };
  const metadata = entry.metadata;
  return metadata?.custom
    ? { id: entry.id, type: "custom_tool_call", call_id: entry.callId, name: metadata.name, ...(metadata.namespace ? { namespace: metadata.namespace } : {}), input: pending ? "" : customInput(entry.args), status }
    : { id: entry.id, type: "function_call", call_id: entry.callId, name: metadata?.name ?? entry.name, ...(metadata?.namespace ? { namespace: metadata.namespace } : {}), arguments: pending ? "" : entry.args, status };
}
function baseResponse(s: State): RelayResponse {
  return { id: s.responseId, object: "response", created_at: Math.floor(Date.now() / 1000), model: s.model as OpenAIResponse["model"],
    output_text: s.entries.filter((entry): entry is Message => entry.kind === "message").map(entry => entry.text).join(""),
    output: s.entries.map(entry => outputItem(s, entry)), error: s.error,
    incomplete_details: s.status === "incomplete" ? { reason: s.reason === "refusal" ? "content_filter" : "max_output_tokens" } : null,
    instructions: null, metadata: null, parallel_tool_calls: true, temperature: null, tool_choice: "auto", tools: [], top_p: null, status: s.status, usage: s.usage,
    ...(s.stopDetails ? { stop_details: s.stopDetails } : {}) };
}
function added(s: State, entry: Entry, emit?: Emit): void {
  s.entries.push(entry);
  emit?.("response.output_item.added", { type: "response.output_item.added", output_index: entry.index, item: outputItem(s, entry, true) });
}
function newReasoning(s: State, emit?: Emit): Reasoning {
  const entry: Reasoning = { kind: "reasoning", id: id("rs"), index: s.entries.length, text: "", summaryStarted: false };
  s.activeMessage = undefined;
  s.activeReasoning = s.lastReasoning = entry;
  added(s, entry, emit);
  return entry;
}
function finishEntry(s: State, entry: Entry, emit?: Emit): void {
  if (entry.kind === "reasoning" && entry.summaryStarted) {
    emit?.("response.reasoning_summary_text.done", { type: "response.reasoning_summary_text.done", item_id: entry.id, output_index: entry.index, summary_index: 0, text: entry.text });
    emit?.("response.reasoning_summary_part.done", { type: "response.reasoning_summary_part.done", item_id: entry.id, output_index: entry.index, summary_index: 0, part: { type: "summary_text", text: entry.text } });
  } else if (entry.kind === "message") {
    emit?.("response.output_text.done", { type: "response.output_text.done", item_id: entry.id, output_index: entry.index, content_index: 0, text: entry.text });
    emit?.("response.content_part.done", { type: "response.content_part.done", item_id: entry.id, output_index: entry.index, content_index: 0, part: { type: "output_text", text: entry.text, annotations: [] } });
  } else if (entry.kind === "tool") {
    const name = entry.metadata?.custom ? "response.custom_tool_call_input.done" : "response.function_call_arguments.done";
    emit?.(name, { type: name, item_id: entry.id, output_index: entry.index, call_id: entry.callId,
      ...(entry.metadata?.custom ? { input: customInput(entry.args) } : { arguments: entry.args }) });
  }
  emit?.("response.output_item.done", { type: "response.output_item.done", output_index: entry.index, item: outputItem(s, entry) });
}
function applyEvent(s: State, e: KiroEvent, emit?: Emit): void {
  if (e.type === "start") {
    if (e.responseId) s.responseId = e.responseId;
    const response = { ...baseResponse(s), status: "in_progress", output: [], output_text: "", usage: null };
    emit?.("response.created", { type: "response.created", response });
    emit?.("response.in_progress", { type: "response.in_progress", response });
  } else if (e.type === "thinking_delta") {
    const entry = s.activeReasoning && s.activeReasoning.redacted === undefined ? s.activeReasoning : newReasoning(s, emit);
    if (!entry.summaryStarted) {
      entry.summaryStarted = true;
      emit?.("response.reasoning_summary_part.added", { type: "response.reasoning_summary_part.added", item_id: entry.id, output_index: entry.index, summary_index: 0, part: { type: "summary_text", text: "" } });
    }
    entry.text += e.thinking;
    emit?.("response.reasoning_summary_text.delta", { type: "response.reasoning_summary_text.delta", item_id: entry.id, output_index: entry.index, summary_index: 0, delta: e.thinking });
  } else if (e.type === "thinking_signature") {
    const entry = s.lastReasoning && s.lastReasoning.redacted === undefined ? s.lastReasoning : newReasoning(s, emit);
    entry.signature = (entry.signature ?? "") + e.signature;
  } else if (e.type === "thinking_redacted") {
    newReasoning(s, emit).redacted = e.data;
  } else if (e.type === "text_delta") {
    s.activeReasoning = undefined;
    if (!s.activeMessage) {
      const entry: Message = { kind: "message", id: id("msg"), index: s.entries.length, text: "" };
      s.activeMessage = entry;
      added(s, entry, emit);
      emit?.("response.content_part.added", { type: "response.content_part.added", item_id: entry.id, output_index: entry.index, content_index: 0, part: { type: "output_text", text: "", annotations: [] } });
    }
    const entry = s.activeMessage;
    entry.text += e.text;
    emit?.("response.output_text.delta", { type: "response.output_text.delta", item_id: entry.id, output_index: entry.index, content_index: 0, delta: e.text });
  } else if (e.type === "tool_start" && !s.calls.has(e.index)) {
    s.activeReasoning = undefined; s.activeMessage = undefined;
    const metadata = s.responseTools[e.name];
    const entry: Call = { kind: "tool", id: id(metadata?.custom ? "ctc" : "fc"), index: s.entries.length, callId: e.id, name: e.name, args: "", metadata };
    s.calls.set(e.index, entry);
    added(s, entry, emit);
  } else if (e.type === "tool_input_delta") {
    const entry = s.calls.get(e.index);
    if (!entry) throw new Error(`Unknown tool index ${e.index}`);
    entry.args += e.delta;
    if (!entry.metadata?.custom) emit?.("response.function_call_arguments.delta", { type: "response.function_call_arguments.delta", item_id: entry.id, output_index: entry.index, call_id: entry.callId, delta: e.delta });
  } else if (e.type === "usage") s.usage = usageOf(e);
  else if (e.type === "stop" && !s.finalized) {
    s.finalized = true; s.reason = e.reason; s.stopDetails = e.details;
    s.status = e.reason === "max_tokens" || e.reason === "refusal" ? "incomplete" : e.reason === "pause_turn" ? "failed" : "completed";
    if (e.reason === "pause_turn") s.error = { code: "server_error", message: "Kiro paused before completing the response" };
    // 签名可晚于正文到达，终止时才发布不可再修改的完整推理项目。
    if (emit) for (const entry of s.entries) finishEntry(s, entry, emit);
    const name = `response.${s.status}`;
    emit?.(name, { type: name, response: baseResponse(s) });
  }
}

export async function collectResponses(response: globalThis.Response, model: string, responseTools: ResponseToolMetadata = {}, options: ResponsesOptions = {}): Promise<RelayResponse> {
  const s = initial(model, responseTools, options);
  for await (const event of readKiroEvents(response, model)) applyEvent(s, event);
  return baseResponse(s);
}
export function responsesStream(response: globalThis.Response, model: string, responseTools: ResponseToolMetadata = {}, options: ResponsesOptions = {}): globalThis.Response {
  const s = initial(model, responseTools, options);
  const encoder = new TextEncoder();
  const abort = new AbortController();
  let sequenceNumber = 0;
  const readable = new ReadableStream<Uint8Array>({ async start(controller) {
    const emit: Emit = (name, data) => controller.enqueue(encoder.encode(`event: ${name}\ndata: ${JSON.stringify({ ...data, sequence_number: sequenceNumber++ })}\n\n`));
    try {
      for await (const event of readKiroEvents(response, model, abort.signal)) {
        if (abort.signal.aborted) return;
        applyEvent(s, event, emit);
      }
      if (abort.signal.aborted) return;
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    } catch (error) {
      if (abort.signal.aborted) return;
      s.status = "failed";
      s.error = { code: "server_error", message: error instanceof Error ? error.message : "Response stream failed" };
      emit("response.failed", { type: "response.failed", response: baseResponse(s) });
      controller.close();
    }
  }, cancel(reason) { abort.abort(reason); } });
  return new globalThis.Response(readable, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" } });
}
export type { ResponseStreamEvent };
