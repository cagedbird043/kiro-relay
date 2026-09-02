import { randomUUID } from "node:crypto";
import type { Response as OpenAIResponse, ResponseStreamEvent, ResponseUsage } from "openai/resources/responses/responses";
import { readKiroEvents, type KiroEvent } from "../kiro/events.ts";

const id = (prefix: string) => `${prefix}_${randomUUID().replaceAll("-", "")}`;
type Usage = ResponseUsage;
type Call = { id: string; callId: string; name: string; args: string; kiroIndex: number; outputIndex: number };
type State = {
  responseId: string; model: string; text: string; thinking: string; output: Array<Record<string, unknown>>;
  usage?: Usage; reason: string; status: "completed" | "incomplete" | "failed"; finalized: boolean;
  messageId?: string; messageIndex?: number; reasoningId?: string; reasoningIndex?: number;
  calls: Map<number, Call>; nextOutputIndex: number; outputOrder: Array<"reasoning" | "message" | number>;
};

function initial(model: string): State {
  return { responseId: id("resp"), model, text: "", thinking: "", output: [], reason: "end_turn", status: "completed", finalized: false, calls: new Map(), nextOutputIndex: 0, outputOrder: [] };
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

function applyEvent(s: State, e: KiroEvent): void {
  if (e.type === "start" && e.responseId) s.responseId = e.responseId;
  else if (e.type === "thinking_delta") { if (s.reasoningIndex === undefined) { s.reasoningId = id("rs"); s.reasoningIndex = allocate(s, "reasoning"); } s.thinking += e.thinking; }
  else if (e.type === "text_delta") { if (s.messageIndex === undefined) { s.messageId = id("msg"); s.messageIndex = allocate(s, "message"); } s.text += e.text; }
  else if (e.type === "stop") { if (s.finalized) return; s.finalized = true; s.reason = e.reason; s.status = e.reason === "max_tokens" ? "incomplete" : "completed"; }
  else if (e.type === "usage") s.usage = usageOf(e);
  else if (e.type === "tool_start" && !s.calls.has(e.index)) s.calls.set(e.index, { id: id("fc"), callId: e.id, name: e.name, args: "", kiroIndex: e.index, outputIndex: allocate(s, e.index) });
  else if (e.type === "tool_input_delta") { const c = s.calls.get(e.index); if (c) c.args += e.delta; }
}

function outputItems(s: State): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  for (const kind of s.outputOrder) {
    if (kind === "reasoning" && s.reasoningId) out.push({ id: s.reasoningId, type: "reasoning", summary: [{ type: "summary_text", text: s.thinking }], status: "completed" });
    else if (kind === "message" && s.messageId) out.push({ id: s.messageId, type: "message", role: "assistant", status: "completed", content: [{ type: "output_text", text: s.text, annotations: [] }] });
    else if (typeof kind === "number") { const c = s.calls.get(kind); if (c) out.push({ id: c.id, type: "function_call", call_id: c.callId, name: c.name, arguments: c.args, status: "completed" }); }
  }
  return out;
}

export async function collectResponses(response: globalThis.Response, model: string): Promise<OpenAIResponse> {
  const s = initial(model); for await (const e of readKiroEvents(response, model)) applyEvent(s, e); s.output = outputItems(s); return baseResponse(s);
}

function streamEventObjects(s: State, e: KiroEvent): Array<[string, Record<string, unknown>]> {
  const result: Array<[string, Record<string, unknown>]> = [];
  if (e.type === "start") { if (e.responseId) s.responseId = e.responseId; result.push(["response.created", responseCreated(s)], ["response.in_progress", { type: "response.in_progress", response: { ...baseResponse(s), status: "in_progress", output: [], output_text: "", usage: null } }]); }
  else if (e.type === "thinking_delta") { if (!s.reasoningId) { s.reasoningId = id("rs"); s.reasoningIndex = allocate(s, "reasoning"); result.push(["response.output_item.added", { type: "response.output_item.added", output_index: s.reasoningIndex, item: { id: s.reasoningId, type: "reasoning", summary: [], status: "in_progress" } }], ["response.reasoning_summary_part.added", { type: "response.reasoning_summary_part.added", item_id: s.reasoningId, output_index: s.reasoningIndex, summary_index: 0, part: { type: "summary_text", text: "" } }]); } s.thinking += e.thinking; result.push(["response.reasoning_summary_text.delta", { type: "response.reasoning_summary_text.delta", item_id: s.reasoningId, output_index: s.reasoningIndex, summary_index: 0, delta: e.thinking }]); }
  else if (e.type === "text_delta") { if (!s.messageId) { s.messageId = id("msg"); s.messageIndex = allocate(s, "message"); result.push(["response.output_item.added", { type: "response.output_item.added", output_index: s.messageIndex, item: { id: s.messageId, type: "message", role: "assistant", status: "in_progress", content: [] } }], ["response.content_part.added", { type: "response.content_part.added", item_id: s.messageId, output_index: s.messageIndex, content_index: 0, part: { type: "output_text", text: "", annotations: [] } }]); } s.text += e.text; result.push(["response.output_text.delta", { type: "response.output_text.delta", item_id: s.messageId, output_index: s.messageIndex, content_index: 0, delta: e.text }]); }
  else if (e.type === "tool_start" && !s.calls.has(e.index)) { const c: Call = { id: id("fc"), callId: e.id, name: e.name, args: "", kiroIndex: e.index, outputIndex: allocate(s, e.index) }; s.calls.set(e.index, c); result.push(["response.output_item.added", { type: "response.output_item.added", output_index: c.outputIndex, item: { id: c.id, type: "function_call", call_id: c.callId, name: c.name, arguments: "", status: "in_progress" } }]); }
  else if (e.type === "tool_input_delta") { const c = s.calls.get(e.index); if (c) { c.args += e.delta; result.push(["response.function_call_arguments.delta", { type: "response.function_call_arguments.delta", item_id: c.id, output_index: c.outputIndex, call_id: c.callId, delta: e.delta }]); } }
  else if (e.type === "usage") s.usage = usageOf(e);
  else if (e.type === "stop") { if (s.finalized) return result; s.finalized = true; s.reason = e.reason; s.status = e.reason === "max_tokens" ? "incomplete" : "completed"; if (s.reasoningId) result.push(["response.reasoning_summary_text.done", { type: "response.reasoning_summary_text.done", item_id: s.reasoningId, output_index: s.reasoningIndex, summary_index: 0, text: s.thinking }], ["response.reasoning_summary_part.done", { type: "response.reasoning_summary_part.done", item_id: s.reasoningId, output_index: s.reasoningIndex, summary_index: 0, part: { type: "summary_text", text: s.thinking } }], ["response.output_item.done", { type: "response.output_item.done", output_index: s.reasoningIndex, item: { id: s.reasoningId, type: "reasoning", summary: [{ type: "summary_text", text: s.thinking }], status: "completed" } }]); if (s.messageId) result.push(["response.output_text.done", { type: "response.output_text.done", item_id: s.messageId, output_index: s.messageIndex, content_index: 0, text: s.text }], ["response.content_part.done", { type: "response.content_part.done", item_id: s.messageId, output_index: s.messageIndex, content_index: 0, part: { type: "output_text", text: s.text, annotations: [] } }], ["response.output_item.done", { type: "response.output_item.done", output_index: s.messageIndex, item: { id: s.messageId, type: "message", role: "assistant", status: "completed", content: [{ type: "output_text", text: s.text, annotations: [] }] } }]); for (const c of s.calls.values()) result.push(["response.function_call_arguments.done", { type: "response.function_call_arguments.done", item_id: c.id, output_index: c.outputIndex, call_id: c.callId, arguments: c.args }], ["response.output_item.done", { type: "response.output_item.done", output_index: c.outputIndex, item: { id: c.id, type: "function_call", call_id: c.callId, name: c.name, arguments: c.args, status: "completed" } }]); s.output = outputItems(s); result.push(["response.completed", { type: "response.completed", response: baseResponse(s) }]); }
  return result;
}

export function responsesStream(response: globalThis.Response, model: string): globalThis.Response {
  const encoder = new TextEncoder(); const body = new ReadableStream<Uint8Array>({ async start(controller) { const s = initial(model); let sequence = 0; try { for await (const e of readKiroEvents(response, model)) for (const [name, data] of streamEventObjects(s, e)) controller.enqueue(encoder.encode(jsonEvent(name, { sequence_number: sequence++, ...data }))); controller.close(); } catch (error) { controller.error(error); } } });
  return new Response(body, { headers: { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" } });
}
export type { ResponseStreamEvent };
