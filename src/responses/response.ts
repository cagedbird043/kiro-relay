import { randomUUID } from "node:crypto";
import type { Response as OpenAIResponse, ResponseStreamEvent, ResponseUsage } from "openai/resources/responses/responses";
import { readKiroEvents, type KiroEvent } from "../kiro/events.ts";

const id = (prefix: string) => `${prefix}_${randomUUID().replaceAll("-", "")}`;
type Usage = ResponseUsage;
export type ResponseToolMetadata = Readonly<
  Record<string, { name: string; namespace?: string; custom: boolean }>
>;
type Call = { id: string; callId: string; name: string; args: string; kiroIndex: number; outputIndex: number; metadata?: { name: string; namespace?: string; custom: boolean } };
type State = {
  responseId: string; model: string; text: string; thinking: string; output: Array<Record<string, unknown>>;
  usage?: Usage; reason: string; status: "completed" | "incomplete" | "failed"; finalized: boolean; reasoningDone: boolean;
  messageId?: string; messageIndex?: number; reasoningId?: string; reasoningIndex?: number; reasoningSignature?: string;
  calls: Map<number, Call>; nextOutputIndex: number; outputOrder: Array<"reasoning" | "message" | number>; responseTools: ResponseToolMetadata;
};

function initial(model: string, responseTools: ResponseToolMetadata = {}): State {
  return { responseId: id("resp"), model, text: "", thinking: "", output: [], reason: "end_turn", status: "completed", finalized: false, reasoningDone: false, calls: new Map(), nextOutputIndex: 0, outputOrder: [], responseTools };
}

function usageOf(event: Extract<KiroEvent, { type: "usage" }>): Usage {
  const cached = event.cacheReadInputTokens ?? 0;
  return { input_tokens: event.inputTokens, output_tokens: event.outputTokens, total_tokens: event.inputTokens + event.outputTokens, input_tokens_details: { cached_tokens: cached, cache_write_tokens: event.cacheCreationInputTokens ?? 0 }, output_tokens_details: { reasoning_tokens: 0 } };
}

function baseResponse(s: State): OpenAIResponse {
  const now = Math.floor(Date.now() / 1000);
  return { id: s.responseId, object: "response", created_at: now, model: s.model as OpenAIResponse["model"], output_text: s.text, output: s.output as unknown as OpenAIResponse["output"], error: null, incomplete_details: s.status === "incomplete" ? { reason: "max_output_tokens" } : null, instructions: null, metadata: null, parallel_tool_calls: true, temperature: null, tool_choice: "auto", tools: [], top_p: null, status: s.status, usage: s.usage };
}
function jsonEvent(event: string, data: Record<string, unknown>): string { return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`; }
function responseCreated(s: State): Record<string, unknown> { return { type: "response.created", response: { ...baseResponse(s), status: "in_progress", output: [], output_text: "", usage: null } }; }
function allocate(s: State, kind: "reasoning" | "message" | number): number { const n = s.nextOutputIndex++; s.outputOrder.push(kind); return n; }
function ensureReasoning(s: State): void {
  if (s.reasoningId) return;
  s.reasoningId = id("rs");
  s.reasoningIndex = allocate(s, "reasoning");
}

function reasoningSummary(s: State): Array<{ type: "summary_text"; text: string }> {
  return s.thinking ? [{ type: "summary_text", text: s.thinking }] : [];
}


function customInput(argumentsText: string): string {
  try {
    const parsed: unknown = JSON.parse(argumentsText);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const input = (parsed as Record<string, unknown>).input;
      if (typeof input === "string") return input;
    }
  } catch {
    // Preserve malformed or already-freeform input verbatim.
  }
  return argumentsText;
}
function applyEvent(s: State, e: KiroEvent): void {
  if (e.type === "start" && e.responseId) s.responseId = e.responseId;
  else if (e.type === "thinking_delta") { ensureReasoning(s); s.thinking += e.thinking; }
  else if (e.type === "text_delta") { if (s.messageIndex === undefined) { s.messageId = id("msg"); s.messageIndex = allocate(s, "message"); } s.text += e.text; }
  else if (e.type === "thinking_signature") { ensureReasoning(s); s.reasoningSignature = e.signature; }
  else if (e.type === "stop") { if (s.finalized) return; s.finalized = true; s.reason = e.reason; s.status = e.reason === "max_tokens" ? "incomplete" : "completed"; }
  else if (e.type === "usage") s.usage = usageOf(e);
  else if (e.type === "tool_start" && !s.calls.has(e.index)) { const metadata = s.responseTools[e.name]; s.calls.set(e.index, { id: id(metadata?.custom ? "ctc" : "fc"), callId: e.id, name: metadata?.name ?? e.name, args: "", kiroIndex: e.index, outputIndex: allocate(s, e.index), metadata }); }
  else if (e.type === "tool_input_delta") { const c = s.calls.get(e.index); if (c) c.args += e.delta; }
}

function outputItems(s: State): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  for (const kind of s.outputOrder) {
    if (kind === "reasoning" && s.reasoningId) out.push({ id: s.reasoningId, type: "reasoning", content: [], encrypted_content: s.reasoningSignature ?? null, summary: reasoningSummary(s), status: "completed" });
    else if (kind === "message" && s.messageId) out.push({ id: s.messageId, type: "message", role: "assistant", status: "completed", content: [{ type: "output_text", text: s.text, annotations: [] }] });
    else if (typeof kind === "number") { const c = s.calls.get(kind); if (c) out.push(c.metadata?.custom ? { id: c.id, type: "custom_tool_call", call_id: c.callId, name: c.name, namespace: c.metadata.namespace, input: customInput(c.args), status: "completed" } : { id: c.id, type: "function_call", call_id: c.callId, name: c.name, ...(c.metadata?.namespace ? { namespace: c.metadata.namespace } : {}), arguments: c.args, status: "completed" }); }
  }
  return out;
}

export async function collectResponses(response: globalThis.Response, model: string, responseTools: ResponseToolMetadata = {}): Promise<OpenAIResponse> {
  const s = initial(model, responseTools); for await (const e of readKiroEvents(response, model)) applyEvent(s, e); s.output = outputItems(s); return baseResponse(s);
}

function finishReasoning(s: State): Array<[string, Record<string, unknown>]> {
  if (!s.reasoningId || s.reasoningDone) return [];
  s.reasoningDone = true;
  const item = {
    id: s.reasoningId,
    type: "reasoning",
    content: [],
    encrypted_content: s.reasoningSignature ?? null,
    summary: reasoningSummary(s),
    status: "completed",
  };
  const result: Array<[string, Record<string, unknown>]> = [];
  if (s.thinking) {
    result.push(
      ["response.reasoning_summary_text.done", { type: "response.reasoning_summary_text.done", item_id: s.reasoningId, output_index: s.reasoningIndex, summary_index: 0, text: s.thinking }],
      ["response.reasoning_summary_part.done", { type: "response.reasoning_summary_part.done", item_id: s.reasoningId, output_index: s.reasoningIndex, summary_index: 0, part: { type: "summary_text", text: s.thinking } }]
    );
  }
  result.push(["response.output_item.done", { type: "response.output_item.done", output_index: s.reasoningIndex, item }]);
  return result;
}

function streamEventObjects(s: State, e: KiroEvent): Array<[string, Record<string, unknown>]> {
  const result: Array<[string, Record<string, unknown>]> = [];
  if (e.type === "start") { if (e.responseId) s.responseId = e.responseId; result.push(["response.created", responseCreated(s)], ["response.in_progress", { type: "response.in_progress", response: { ...baseResponse(s), status: "in_progress", output: [], output_text: "", usage: null } }]); }
  else if (e.type === "thinking_delta") { if (!s.reasoningId) { s.reasoningId = id("rs"); s.reasoningIndex = allocate(s, "reasoning"); result.push(["response.output_item.added", { type: "response.output_item.added", output_index: s.reasoningIndex, item: { id: s.reasoningId, type: "reasoning", summary: [], status: "in_progress" } }], ["response.reasoning_summary_part.added", { type: "response.reasoning_summary_part.added", item_id: s.reasoningId, output_index: s.reasoningIndex, summary_index: 0, part: { type: "summary_text", text: "" } }]); } s.thinking += e.thinking; result.push(["response.reasoning_summary_text.delta", { type: "response.reasoning_summary_text.delta", item_id: s.reasoningId, output_index: s.reasoningIndex, summary_index: 0, delta: e.thinking }]); }
  else if (e.type === "thinking_signature") { const added = !s.reasoningId; ensureReasoning(s); s.reasoningSignature = e.signature; if (added) result.push(["response.output_item.added", { type: "response.output_item.added", output_index: s.reasoningIndex, item: { id: s.reasoningId, type: "reasoning", content: [], encrypted_content: null, summary: [], status: "in_progress" } }]); if (s.messageId || s.calls.size > 0) result.push(...finishReasoning(s)); }
  else if (e.type === "text_delta") { if (s.reasoningSignature) result.push(...finishReasoning(s)); if (!s.messageId) { s.messageId = id("msg"); s.messageIndex = allocate(s, "message"); result.push(["response.output_item.added", { type: "response.output_item.added", output_index: s.messageIndex, item: { id: s.messageId, type: "message", role: "assistant", status: "in_progress", content: [] } }], ["response.content_part.added", { type: "response.content_part.added", item_id: s.messageId, output_index: s.messageIndex, content_index: 0, part: { type: "output_text", text: "", annotations: [] } }]); } s.text += e.text; result.push(["response.output_text.delta", { type: "response.output_text.delta", item_id: s.messageId, output_index: s.messageIndex, content_index: 0, delta: e.text }]); }
  else if (e.type === "tool_start" && !s.calls.has(e.index)) { if (s.reasoningSignature) result.push(...finishReasoning(s)); const metadata = s.responseTools[e.name]; const c: Call = { id: id(metadata?.custom ? "ctc" : "fc"), callId: e.id, name: metadata?.name ?? e.name, args: "", kiroIndex: e.index, outputIndex: allocate(s, e.index), metadata }; s.calls.set(e.index, c); result.push(["response.output_item.added", { type: "response.output_item.added", output_index: c.outputIndex, item: metadata?.custom ? { id: c.id, type: "custom_tool_call", call_id: c.callId, name: c.name, namespace: metadata.namespace, input: "", status: "in_progress" } : { id: c.id, type: "function_call", call_id: c.callId, name: c.name, ...(metadata?.namespace ? { namespace: metadata.namespace } : {}), arguments: "", status: "in_progress" } }]); }
  else if (e.type === "tool_input_delta") { const c = s.calls.get(e.index); if (c) { c.args += e.delta; if (!c.metadata?.custom) result.push(["response.function_call_arguments.delta", { type: "response.function_call_arguments.delta", item_id: c.id, output_index: c.outputIndex, call_id: c.callId, delta: e.delta }]); } }
  else if (e.type === "usage") s.usage = usageOf(e);
  else if (e.type === "stop") {
    if (s.finalized) return result;
    s.finalized = true;
    s.reason = e.reason;
    s.status = e.reason === "max_tokens" ? "incomplete" : "completed";
    result.push(...finishReasoning(s));
    if (s.messageId) result.push(["response.output_text.done", { type: "response.output_text.done", item_id: s.messageId, output_index: s.messageIndex, content_index: 0, text: s.text }], ["response.content_part.done", { type: "response.content_part.done", item_id: s.messageId, output_index: s.messageIndex, content_index: 0, part: { type: "output_text", text: s.text, annotations: [] } }], ["response.output_item.done", { type: "response.output_item.done", output_index: s.messageIndex, item: { id: s.messageId, type: "message", role: "assistant", status: "completed", content: [{ type: "output_text", text: s.text, annotations: [] }] } }]);
    for (const c of s.calls.values()) { if (c.metadata?.custom) { const input = customInput(c.args); result.push(["response.custom_tool_call_input.done", { type: "response.custom_tool_call_input.done", item_id: c.id, call_id: c.callId, output_index: c.outputIndex, input }], ["response.output_item.done", { type: "response.output_item.done", output_index: c.outputIndex, item: { id: c.id, type: "custom_tool_call", call_id: c.callId, name: c.name, namespace: c.metadata.namespace, input, status: "completed" } }]); } else result.push(["response.function_call_arguments.done", { type: "response.function_call_arguments.done", item_id: c.id, output_index: c.outputIndex, call_id: c.callId, arguments: c.args }], ["response.output_item.done", { type: "response.output_item.done", output_index: c.outputIndex, item: { id: c.id, type: "function_call", call_id: c.callId, name: c.name, ...(c.metadata?.namespace ? { namespace: c.metadata.namespace } : {}), arguments: c.args, status: "completed" } }]); }
    s.output = outputItems(s);
    result.push(["response.completed", { type: "response.completed", response: baseResponse(s) }]);
  }
  return result;
}

export function responsesStream(response: globalThis.Response, model: string, responseTools: ResponseToolMetadata = {}): globalThis.Response {
  const s = initial(model, responseTools); const encoder = new TextEncoder(); let sequenceNumber = 0;
  const readable = new ReadableStream<Uint8Array>({ async start(controller) { try { for await (const e of readKiroEvents(response, model)) { for (const [name, data] of streamEventObjects(s, e)) controller.enqueue(encoder.encode(jsonEvent(name, { ...data, sequence_number: sequenceNumber++ }))); } controller.enqueue(encoder.encode("data: [DONE]\n\n")); controller.close(); } catch (error) { controller.error(error); } } });
  return new globalThis.Response(readable, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" } });
}
export type { ResponseStreamEvent };
