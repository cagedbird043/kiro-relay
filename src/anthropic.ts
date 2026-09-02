/* Anthropic <-> Kiro wire adapters. */
import { readKiroEvents, type KiroEvent } from "./kiro/events.ts";
type Json = Record<string, unknown>;

export function anthropicToOpenAI(body: Json): Json {
  const out: Json = { ...body };
  if (Array.isArray(body.messages)) out.messages = body.messages.map((message) => {
    if (!message || typeof message !== "object") return message;
    const m = { ...(message as Json) }; const content = m.content;
    if (Array.isArray(content)) m.content = content.map((block) => {
      if (!block || typeof block !== "object") return block;
      const b = block as Json;
      if (b.type === "image" && b.source && typeof b.source === "object") { const s = b.source as Json; const url = s.type === "base64" ? `data:${typeof s.media_type === "string" ? s.media_type : "image/jpeg"};base64,${typeof s.data === "string" ? s.data : ""}` : s.url; return { type: "image_url", image_url: { url } }; }
      if (b.type === "tool_use") return { ...b, input: b.input ?? {} };
      if (b.type === "tool_result") return { ...b, content: b.content ?? "" };
      return { ...b };
    });
    if (m.role === "assistant" && Array.isArray(m.content)) { const calls = m.content.filter((b): b is Json => !!b && typeof b === "object" && (b as Json).type === "tool_use"); if (calls.length) m.tool_calls = calls.map((b) => ({ id: b.id, type: "function", function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) } })); }
    return m;
  });
  if (Array.isArray(body.system)) out.system = body.system.map((b) => ({ ...(b as Json) }));
  return out;
}
function makeId(prefix: string): string { try { return `${prefix}_${crypto.randomUUID()}`; } catch { return `${prefix}_${Date.now()}`; } }
function sse(event: string, data: Json): string { return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`; }
type Block = { type: string; id?: string; name?: string; text?: string; thinking?: string; json?: string; input?: unknown };

export function openAIStreamToAnthropic(response: Response, model = "unknown"): Response {
  const encoder = new TextEncoder(); const messageId = makeId("msg");
  const stream = new ReadableStream<Uint8Array>({ start(controller) {
    void (async () => {
      const blocks: Block[] = []; const toolBlocks = new Map<number, number>(); const closed = new Set<number>(); let active: number | undefined; let thinkingIndex: number | undefined; let textIndex: number | undefined;
      let input = 0; let output = 0; let cacheRead: number | undefined; let cacheCreation: number | undefined; let reason = "end_turn"; let started = false;
      const emit = (event: string, data: Json) => controller.enqueue(encoder.encode(sse(event, data)));
      const close = (index: number) => { if (!closed.has(index)) { closed.add(index); emit("content_block_stop", { type: "content_block_stop", index }); } };
      const ensure = (index: number, type: string, extra: Json = {}) => { if (blocks[index]) return; if (active !== undefined) close(active); active = index; blocks[index] = { type, ...extra }; emit("content_block_start", { type: "content_block_start", index, content_block: { type, ...extra, ...(type === "text" ? { text: "" } : {}), ...(type === "thinking" ? { thinking: "" } : {}), ...(type === "tool_use" ? { input: {} } : {}) } }); };
      const finish = () => { if (!started) { started = true; emit("message_start", { type: "message_start", message: { id: messageId, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } }); } blocks.forEach((_b, i) => close(i)); const usage: Json = { input_tokens: input, output_tokens: output }; if (cacheRead !== undefined) usage.cache_read_input_tokens = cacheRead; if (cacheCreation !== undefined) usage.cache_creation_input_tokens = cacheCreation; emit("message_delta", { type: "message_delta", delta: { stop_reason: reason, stop_sequence: null }, usage }); emit("message_stop", { type: "message_stop" }); controller.close(); };
      try { for await (const event of readKiroEvents(response, model)) {
        if (event.type === "start") { if (!started) { started = true; emit("message_start", { type: "message_start", message: { id: event.responseId || messageId, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } }); } }
        else if (event.type === "thinking_delta") { const index = thinkingIndex ?? (thinkingIndex = blocks.length); ensure(index, "thinking"); emit("content_block_delta", { type: "content_block_delta", index, delta: { type: "thinking_delta", thinking: event.thinking } }); }
        else if (event.type === "text_delta") { const index = textIndex ?? (textIndex = blocks.length); ensure(index, "text"); emit("content_block_delta", { type: "content_block_delta", index, delta: { type: "text_delta", text: event.text } }); }
        else if (event.type === "tool_start") { const index = toolBlocks.get(event.index) ?? blocks.length; toolBlocks.set(event.index, index); ensure(index, "tool_use", { id: event.id, name: event.name }); }
        else if (event.type === "tool_input_delta") { const index = toolBlocks.get(event.index) ?? blocks.length; toolBlocks.set(event.index, index); ensure(index, "tool_use"); emit("content_block_delta", { type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: event.delta } }); }
        else if (event.type === "usage") { input = event.inputTokens; output = event.outputTokens; cacheRead = event.cacheReadInputTokens; cacheCreation = event.cacheCreationInputTokens; }
        else if (event.type === "stop") reason = event.reason;
      } finish(); } catch { finish(); }
    })();
  }, cancel() {} });
  return new Response(stream, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" } });
}
export async function collectAnthropicResponse(response: Response, model = "unknown"): Promise<Response> {
  const events: KiroEvent[] = []; for await (const event of readKiroEvents(response, model)) events.push(event);
  const start = events.find((e): e is Extract<KiroEvent, {type:"start"}> => e.type === "start"); const content: Block[] = []; const args = new Map<number, string>(); const toolBlocks = new Map<number, number>();
  for (const event of events) { if (event.type === "text_delta") { let b = content.find((x) => x.type === "text"); if (!b) { b = { type: "text", text: "" }; content.push(b); } b.text = (b.text || "") + event.text; } else if (event.type === "thinking_delta") { let b = content.find((x) => x.type === "thinking"); if (!b) { b = { type: "thinking", thinking: "" }; content.push(b); } b.thinking = (b.thinking || "") + event.thinking; } else if (event.type === "tool_start") { const blockIndex = toolBlocks.get(event.index) ?? content.length; toolBlocks.set(event.index, blockIndex); content[blockIndex] = { type: "tool_use", id: event.id, name: event.name, input: {} }; } else if (event.type === "tool_input_delta") args.set(event.index, (args.get(event.index) || "") + event.delta); }
  for (const [index, raw] of args) { const blockIndex = toolBlocks.get(index); const b = blockIndex === undefined ? undefined : content[blockIndex]; if (b) { try { b.input = JSON.parse(raw); } catch { b.input = raw; } } }
  const usageEvent = events.find((e): e is Extract<KiroEvent, {type:"usage"}> => e.type === "usage"); const stop = [...events].reverse().find((e): e is Extract<KiroEvent, {type:"stop"}> => e.type === "stop"); const usage: Json = { input_tokens: usageEvent?.inputTokens || 0, output_tokens: usageEvent?.outputTokens || 0 }; if (usageEvent?.cacheReadInputTokens !== undefined) usage.cache_read_input_tokens = usageEvent.cacheReadInputTokens; if (usageEvent?.cacheCreationInputTokens !== undefined) usage.cache_creation_input_tokens = usageEvent.cacheCreationInputTokens;
  return Response.json({ id: start?.responseId || makeId("msg"), type: "message", role: "assistant", model, content: content.filter(Boolean), stop_reason: stop?.reason || "end_turn", stop_sequence: null, usage });
}
export function countAnthropicTokens(body: Json): number { let text = JSON.stringify(body.messages || []); if (body.system) text += JSON.stringify(body.system); if (body.tools) text += JSON.stringify(body.tools); return Math.max(1, Math.ceil(text.length / 4)); }
