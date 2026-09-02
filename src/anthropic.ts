/* Anthropic <-> OpenAI wire adapters for the Kiro relay. */

type AnyRecord = Record<string, any>;

/** Convert an Anthropic request to the hybrid message shape understood by buildKiroPayload. */
export function anthropicToOpenAI(body: AnyRecord): AnyRecord {
  const out: AnyRecord = { ...body };
  if (Array.isArray(body.messages)) {
    out.messages = body.messages.map((message: AnyRecord) => {
      const m: AnyRecord = { ...message };
      if (Array.isArray(message.content)) {
        m.content = message.content.map((block: AnyRecord) => {
          if (!block || typeof block !== "object") return block;
          if (block.type === "image" && block.source) {
            const source = block.source;
            const url = source.type === "base64" ? `data:${source.media_type || "image/jpeg"};base64,${source.data || ""}` : source.url;
            return { type: "image_url", image_url: { url } };
          }
          if (block.type === "tool_use") return { ...block, input: block.input ?? {} };
          if (block.type === "tool_result") return { ...block, content: block.content ?? "" };
          return { ...block };
        });
      }
      if (m.role === "assistant" && Array.isArray(m.content)) {
        const calls = m.content.filter((b: AnyRecord) => b?.type === "tool_use");
        if (calls.length) m.tool_calls = calls.map((b: AnyRecord) => ({ id: b.id, type: "function", function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) } }));
      }
      return m;
    });
  }
  if (Array.isArray(body.system)) out.system = body.system.map((b: AnyRecord) => ({ ...b }));
  return out;
}

function id(prefix: string): string {
  try { return `${prefix}_${crypto.randomUUID()}`; } catch { return `${prefix}_${Date.now()}`; }
}
function sse(event: string, data: AnyRecord): string { return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`; }
function textOfDelta(delta: any): string { return typeof delta === "string" ? delta : typeof delta?.text === "string" ? delta.text : ""; }

/** Convert OpenAI SSE chunks into Anthropic's ordered SSE event stream. */
export function openAIStreamToAnthropic(response: Response, model = "unknown"): Response {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const reader = response.body?.getReader();
  const state: { started: boolean; blocks: Array<{ type: string; id?: string; name?: string }>; keys: Map<string, number>; closed: Set<number>; stopped: boolean; messageId: string; input?: number; output?: number; cacheRead?: number; cacheCreation?: number; stopReason: string } = { started: false, blocks: [], keys: new Map(), closed: new Set(), stopped: false, messageId: id("msg"), stopReason: "end_turn" };
  let pending = "";
  let activeIndex: number | undefined;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      void (async () => {
        if (!reader) {
          finish(controller);
          return;
        }
        try {
          while (!state.stopped) {
            const { value, done } = await reader.read();
            if (done) {
              pending += decoder.decode();
              if (pending) processLine(pending, controller);
              finish(controller);
              return;
            }
            pending += decoder.decode(value, { stream: true });
            const lines = pending.split(/\r?\n/);
            pending = lines.pop() || "";
            for (const line of lines) processLine(line, controller);
          }
        } catch {
          finish(controller);
        }
      })();
    },
    cancel() {
      return reader?.cancel().catch(() => undefined);
    },
  });
  function emit(controller: ReadableStreamDefaultController<Uint8Array>, event: string, data: AnyRecord) { controller.enqueue(encoder.encode(sse(event, data))); }
  function start(controller: ReadableStreamDefaultController<Uint8Array>) {
    if (state.started) return;
    state.started = true;
    emit(controller, "message_start", { type: "message_start", message: { id: state.messageId, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: state.input ?? 0, output_tokens: state.output ?? 0 } } });
  }
  function mapped(key: string): number { const old = state.keys.get(key); if (old !== undefined) return old; const index = state.blocks.length; state.keys.set(key, index); return index; }
  function stopBlock(controller: ReadableStreamDefaultController<Uint8Array>, index: number) {
    if (state.closed.has(index)) return;
    state.closed.add(index);
    emit(controller, "content_block_stop", { type: "content_block_stop", index });
  }
  function block(controller: ReadableStreamDefaultController<Uint8Array>, index: number, type: string, extra: AnyRecord = {}) {
    if (state.blocks[index]) return;
    if (activeIndex !== undefined) stopBlock(controller, activeIndex);
    activeIndex = index;
    state.blocks[index] = { type, ...extra };
    emit(controller, "content_block_start", { type: "content_block_start", index, content_block: { type, ...extra, ...(type === "text" ? { text: "" } : {}), ...(type === "thinking" ? { thinking: "" } : {}), ...(type === "tool_use" ? { input: {} } : {}) } });
  }
  function processLine(line: string, controller: ReadableStreamDefaultController<Uint8Array>) {
    if (!line.startsWith("data:")) return;
    const raw = line.slice(5).trim();
    if (!raw) return;
    if (raw === "[DONE]") {
      finish(controller);
      void reader?.cancel().catch(() => undefined);
      return;
    }
    let chunk: AnyRecord; try { chunk = JSON.parse(raw); } catch { return; }
    start(controller);
    const choice = chunk.choices?.[0] || {};
    const delta = choice.delta || choice.message || {};
    const choiceIndex = Number.isInteger(choice.index) ? choice.index : 0;
    if (delta.reasoning_content !== undefined) { const index = mapped(`thinking:${choiceIndex}`); block(controller, index, "thinking"); const value = textOfDelta(delta.reasoning_content); if (value) emit(controller, "content_block_delta", { type: "content_block_delta", index, delta: { type: "thinking_delta", thinking: value } }); }
    if (delta.content !== undefined) { const value = textOfDelta(delta.content); if (value) { const index = mapped(`text:${choiceIndex}`); block(controller, index, "text"); emit(controller, "content_block_delta", { type: "content_block_delta", index, delta: { type: "text_delta", text: value } }); } }
    for (const tc of (Array.isArray(delta.tool_calls) ? delta.tool_calls : [])) { const ti = Number.isInteger(tc.index) ? tc.index : 0; const index = mapped(`tool:${choiceIndex}:${ti}`); block(controller, index, "tool_use", { id: tc.id || id("toolu"), name: tc.function?.name || tc.name || "" }); const args = tc.function?.arguments ?? tc.arguments; if (typeof args === "string" && args) emit(controller, "content_block_delta", { type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: args } }); }
    if (choice.finish_reason) state.stopReason = choice.finish_reason === "tool_calls" ? "tool_use" : choice.finish_reason === "length" ? "max_tokens" : "end_turn";
    const usage = chunk.usage;
    if (usage) { if (usage.prompt_tokens !== undefined || usage.input_tokens !== undefined) state.input = usage.prompt_tokens ?? usage.input_tokens; if (usage.completion_tokens !== undefined || usage.output_tokens !== undefined) state.output = usage.completion_tokens ?? usage.output_tokens; if (usage.cache_read_input_tokens !== undefined || usage.cache_read_tokens !== undefined) state.cacheRead = usage.cache_read_input_tokens ?? usage.cache_read_tokens; if (usage.cache_creation_input_tokens !== undefined || usage.cache_creation_tokens !== undefined) state.cacheCreation = usage.cache_creation_input_tokens ?? usage.cache_creation_tokens; }
  }
  function finish(controller: ReadableStreamDefaultController<Uint8Array>) {
    if (state.stopped) return; state.stopped = true; start(controller);
    state.blocks.forEach((_block, index) => stopBlock(controller, index));
    const usage: AnyRecord = { input_tokens: state.input ?? 0, output_tokens: state.output ?? 0 };
    if (state.cacheRead !== undefined) usage.cache_read_input_tokens = state.cacheRead;
    if (state.cacheCreation !== undefined) usage.cache_creation_input_tokens = state.cacheCreation;
    emit(controller, "message_delta", { type: "message_delta", delta: { stop_reason: state.stopReason, stop_sequence: null }, usage });
    emit(controller, "message_stop", { type: "message_stop" }); controller.close();
  }
  return new Response(stream, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" } });
}

/** Collect an OpenAI response and return an ordinary Anthropic message response. */
export async function collectAnthropicResponse(response: Response, model = "unknown"): Promise<Response> {
  let source = response;
  if (response.headers.get("content-type")?.includes("application/json")) {
    const raw = await response.text();
    source = new Response(`data: ${raw.trim()}\n\ndata: [DONE]\n\n`, {
      headers: { "content-type": "text/event-stream" },
    });
  }
  const converted = await openAIStreamToAnthropic(source, model).text();
  const events: Array<{ event: string; data: AnyRecord }> = [];
  for (const match of converted.matchAll(/event:\s*([^\r\n]+)\r?\ndata:\s*([^\r\n]+)/g)) { try { events.push({ event: match[1], data: JSON.parse(match[2]) }); } catch {} }
  const start = events.find((e) => e.event === "message_start")?.data?.message || { id: id("msg"), type: "message", role: "assistant", model };
  const content: AnyRecord[] = [];
  for (const event of events) {
    const data = event.data;
    if (event.event === "content_block_start") content[data.index] = { ...data.content_block };
    else if (event.event === "content_block_delta") { const block = content[data.index] || (content[data.index] = { type: "text", text: "" }); const delta = data.delta || {}; if (delta.text !== undefined) block.text = (block.text || "") + delta.text; if (delta.thinking !== undefined) block.thinking = (block.thinking || "") + delta.thinking; if (delta.partial_json !== undefined) block.__json = (block.__json || "") + delta.partial_json; }
  }
  for (const block of content) if (block && block.__json !== undefined) { try { block.input = JSON.parse(block.__json); } catch { block.input = {}; } delete block.__json; }
  const md = events.find((e) => e.event === "message_delta")?.data;
  const streamUsage = md?.usage || {};
  const usage: AnyRecord = { input_tokens: streamUsage.input_tokens ?? start.usage?.input_tokens ?? 0, output_tokens: streamUsage.output_tokens ?? start.usage?.output_tokens ?? 0 };
  if (streamUsage.cache_read_input_tokens !== undefined) usage.cache_read_input_tokens = streamUsage.cache_read_input_tokens;
  if (streamUsage.cache_creation_input_tokens !== undefined) usage.cache_creation_input_tokens = streamUsage.cache_creation_input_tokens;
  return Response.json({ ...start, content: content.filter(Boolean), stop_reason: md?.delta?.stop_reason || "end_turn", stop_sequence: null, usage });
}

/** Deterministic local estimate; deliberately does not claim cache usage. */
export function countAnthropicTokens(body: AnyRecord): number {
  let text = JSON.stringify(body?.messages || []); if (body?.system) text += JSON.stringify(body.system); if (body?.tools) text += JSON.stringify(body.tools); return Math.max(1, Math.ceil(text.length / 4));
}
