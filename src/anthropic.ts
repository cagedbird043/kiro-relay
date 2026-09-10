/* Anthropic <-> Kiro wire adapters. */
import { readKiroEvents, type KiroEvent } from "./kiro/events.ts";
type Json = Record<string, unknown>;

export function anthropicToOpenAI(body: Json): Json {
  const reject = (message: string): never => { throw new Error(message); };
  const outputConfig = body.output_config;
  if (outputConfig && typeof outputConfig === "object" && !Array.isArray(outputConfig)) {
    const unsupported = ["format", "task_budget"].find((key) => key in (outputConfig as Json));
    if (unsupported) reject(`Unsupported output_config.${unsupported}`);
  }
  if ("speed" in body) reject("Unsupported speed");
  if (body.stop_sequences !== undefined) {
    if (!Array.isArray(body.stop_sequences) || body.stop_sequences.some(value => typeof value !== "string")) reject("stop_sequences must be an array of strings");
    if ((body.stop_sequences as unknown[]).length) reject("Unsupported stop_sequences on this relay");
  }
  const validateBlock = (block: unknown, location: string) => {
    if (!block || typeof block !== "object") return;
    const b = block as Json;
    if (b.type === "document") reject(`Unsupported ${location} document content`);
    const source = b.source && typeof b.source === "object" ? b.source as Json : undefined;
    const imageUrl = b.image_url && typeof b.image_url === "object" ? b.image_url as Json : undefined;
    const remoteImage =
      (b.type === "image" && source?.type === "url") ||
      (b.type === "image_url" && typeof imageUrl?.url === "string" && !imageUrl.url.startsWith("data:"));
    if (remoteImage) reject(`Unsupported ${location} remote image URL`);
    if (b.type === "tool_result" && Array.isArray(b.content)) {
      for (const item of b.content) {
        if (!item || typeof item !== "object" || Array.isArray(item)) {
          reject(`Unsupported ${location} structured tool_result content`);
        }
        const structured = item as Json;
        const validText = structured.type === "text" && typeof structured.text === "string";
        const imageSource = structured.source && typeof structured.source === "object" &&
          !Array.isArray(structured.source) ? structured.source as Json : undefined;
        const validBase64Image = structured.type === "image" && imageSource?.type === "base64" &&
          typeof imageSource.media_type === "string" && imageSource.media_type.startsWith("image/") &&
          typeof imageSource.data === "string" && imageSource.data.length > 0;
        if (!validText && !validBase64Image) {
          reject(`Unsupported ${location} structured tool_result content`);
        }
      }
    }
  };
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!message || typeof message !== "object") continue;
      const content = (message as Json).content;
      if (Array.isArray(content)) content.forEach((block) => validateBlock(block, "message"));
    }
  }
  if (Array.isArray(body.system)) {
    for (const block of body.system) validateBlock(block, "system");
  }
  const { system, ...out }: Json = body;
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
  // Anthropic 将系统提示放在顶层，Kiro 转换器只读取消息序列。
  if (typeof system === "string" || Array.isArray(system)) {
    const content = Array.isArray(system) ? system.map((block) => ({ ...(block as Json) })) : system;
    out.messages = [{ role: "system", content }, ...(Array.isArray(out.messages) ? out.messages : [])];
  }
  return out;
}
function makeId(prefix: string): string { try { return `${prefix}_${crypto.randomUUID()}`; } catch { return `${prefix}_${Date.now()}`; } }
function sse(event: string, data: Json): string { return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`; }
type Block = { type: string; id?: string; name?: string; text?: string; thinking?: string; signature?: string; data?: string; input?: unknown };

export function openAIStreamToAnthropic(response: Response, model = "unknown"): Response {
  const encoder = new TextEncoder(); const messageId = makeId("msg");
  const abort = new AbortController();
  let ended = false;
  let timer: NodeJS.Timeout | undefined;
  const stream = new ReadableStream<Uint8Array>({ start(controller) {
    void (async () => {
      const blocks: Block[] = []; const toolBlocks = new Map<number, number>(); const closed = new Set<number>();
      let active: number | undefined; let activeType: string | undefined;
      let input = 0; let output = 0; let cacheRead: number | undefined; let cacheCreation: number | undefined; let reason = "end_turn"; let started = false; let terminal = false;
      let estimated = false;
      let stopDetails: Record<string, unknown> | undefined;
      const emit = (event: string, data: Json) => { if (!ended) controller.enqueue(encoder.encode(sse(event, data))); };
      const close = (index: number) => { if (!closed.has(index)) { closed.add(index); emit("content_block_stop", { type: "content_block_stop", index }); if (active === index) { active = undefined; activeType = undefined; } } };
      const ensure = (type: string, extra: Json = {}, forceFresh = false): number => {
        if (!forceFresh && active !== undefined && activeType === type && !closed.has(active)) return active;
        if (active !== undefined && !closed.has(active)) close(active);
        const index = blocks.length; active = index; activeType = type; blocks[index] = { type, ...extra };
        const contentBlock: Json = { type, ...extra, ...(type === "text" ? { text: "" } : {}), ...(type === "thinking" ? { thinking: "" } : {}), ...(type === "tool_use" ? { input: {} } : {}) };
        emit("content_block_start", { type: "content_block_start", index, content_block: contentBlock }); return index;
      };
      const ensureTool = (eventIndex: number, extra: Json): number => {
        const mapped = toolBlocks.get(eventIndex);
        if (mapped !== undefined) {
          if (closed.has(mapped)) throw new Error(`tool index ${eventIndex} is already closed`);
          if (active !== mapped) { if (active !== undefined && !closed.has(active)) close(active); active = mapped; activeType = "tool_use"; }
          return mapped;
        }
        return ensure("tool_use", extra);
      };
      const finish = () => {
        if (ended) return;
        if (!started) emit("message_start", { type: "message_start", message: { id: messageId, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } });
        blocks.forEach((_block, index) => close(index));
        const usage: Json = { input_tokens: input, output_tokens: output, ...(estimated ? { estimated: true } : {}) };
        if (cacheRead !== undefined) usage.cache_read_input_tokens = cacheRead;
        if (cacheCreation !== undefined) usage.cache_creation_input_tokens = cacheCreation;
        emit("message_delta", { type: "message_delta", delta: { stop_reason: reason, stop_sequence: null, ...(stopDetails ? { stop_details: stopDetails } : {}) }, usage });
        emit("message_stop", { type: "message_stop" });
        ended = true;
        controller.close();
      };
      const fail = (error: unknown) => { if (ended) return; ended = true; if (!started) { started = true; controller.enqueue(encoder.encode(sse("message_start", { type: "message_start", message: { id: messageId, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } }))); } const message = error instanceof Error ? error.message : "upstream stream failed"; controller.enqueue(encoder.encode(sse("error", { type: "error", error: { type: "api_error", message } }))); try { controller.close(); } catch {} };
      timer = setInterval(() => { if (!ended) emit("ping", { type: "ping" }); }, 15000);
      try { for await (const event of readKiroEvents(response, model, abort.signal)) {
        if (ended) break;
        if (event.type === "start") { if (!started) { started = true; emit("message_start", { type: "message_start", message: { id: event.responseId || messageId, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } }); } }
        else if (event.type === "thinking_delta") { const index = ensure("thinking"); emit("content_block_delta", { type: "content_block_delta", index, delta: { type: "thinking_delta", thinking: event.thinking } }); }
        else if (event.type === "thinking_signature") { const index = (activeType === "thinking" && active !== undefined && !closed.has(active)) ? active : [...blocks].map((b, i) => [b, i] as const).reverse().find(([b, i]) => b.type === "thinking" && !closed.has(i))?.[1] ?? ensure("thinking"); blocks[index].signature = (blocks[index].signature || "") + event.signature; emit("content_block_delta", { type: "content_block_delta", index, delta: { type: "signature_delta", signature: event.signature } }); }
        else if (event.type === "thinking_redacted") { const index = ensure("redacted_thinking", { data: event.data }, true); close(index); }
        else if (event.type === "text_delta") { const index = ensure("text"); emit("content_block_delta", { type: "content_block_delta", index, delta: { type: "text_delta", text: event.text } }); }
        else if (event.type === "tool_start") { const mapped = toolBlocks.get(event.index); const index = mapped === undefined ? ensure("tool_use", { id: event.id, name: event.name }, true) : ensureTool(event.index, { id: event.id, name: event.name }); toolBlocks.set(event.index, index); }
        else if (event.type === "tool_input_delta") { const mapped = toolBlocks.get(event.index); if (mapped === undefined || closed.has(mapped)) throw new Error(`unknown or closed tool index ${event.index}`); const index = ensureTool(event.index, {}); emit("content_block_delta", { type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: event.delta } }); }
        else if (event.type === "usage") { input = event.inputTokens; output = event.outputTokens; cacheRead = event.cacheReadInputTokens; cacheCreation = event.cacheCreationInputTokens; estimated = event.estimated === true; }
        else if (event.type === "stop") { reason = event.reason; stopDetails = event.details; terminal = true; }
      } if (terminal) finish(); else throw new Error("Kiro stream ended before terminal event"); } catch (error) { fail(error); } finally { clearInterval(timer); }
    })();
  }, cancel(reason) { ended = true; clearInterval(timer); abort.abort(reason); } });
  return new Response(stream, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" } });
}

export async function collectAnthropicResponse(response: Response, model = "unknown"): Promise<Response> {
  const events: KiroEvent[] = []; for await (const event of readKiroEvents(response, model)) events.push(event);
  const start = events.find((e): e is Extract<KiroEvent, {type:"start"}> => e.type === "start"); const content: Block[] = []; const args = new Map<number, string>(); const toolBlocks = new Map<number, number>(); let current: number | undefined;
  for (const event of events) {
    if (event.type === "text_delta") {
      if (current === undefined || content[current]?.type !== "text") { current = content.length; content.push({ type: "text", text: "" }); }
      content[current].text = (content[current].text || "") + event.text;
    } else if (event.type === "thinking_delta") {
      if (current === undefined || content[current]?.type !== "thinking") { current = content.length; content.push({ type: "thinking", thinking: "" }); }
      content[current].thinking = (content[current].thinking || "") + event.thinking;
    } else if (event.type === "thinking_signature") {
      let index = [...content].map((block, i) => [block, i] as const).reverse().find(([block]) => block.type === "thinking")?.[1];
      if (index === undefined || (current !== undefined && content[current]?.type === "tool_use")) {
        index = content.length;
        content.push({ type: "thinking", thinking: "" });
        current = index;
      }
      content[index].signature = (content[index].signature || "") + event.signature;
    } else if (event.type === "thinking_redacted") {
      current = content.length;
      content.push({ type: "redacted_thinking", data: event.data });
    } else if (event.type === "tool_start") {
      const blockIndex = toolBlocks.get(event.index) ?? content.length;
      toolBlocks.set(event.index, blockIndex);
      current = blockIndex;
      content[blockIndex] = { type: "tool_use", id: event.id, name: event.name, input: {} };
    } else if (event.type === "tool_input_delta") {
      args.set(event.index, (args.get(event.index) || "") + event.delta);
    }
  }
  for (const [index, raw] of args) { const blockIndex = toolBlocks.get(index); const b = blockIndex === undefined ? undefined : content[blockIndex]; if (b) { try { b.input = JSON.parse(raw); } catch { b.input = raw; } } }
  const usageEvent = events.find((e): e is Extract<KiroEvent, {type:"usage"}> => e.type === "usage"); const stop = [...events].reverse().find((e): e is Extract<KiroEvent, {type:"stop"}> => e.type === "stop"); const usage: Json = { input_tokens: usageEvent?.inputTokens || 0, output_tokens: usageEvent?.outputTokens || 0 }; if (usageEvent?.cacheReadInputTokens !== undefined) usage.cache_read_input_tokens = usageEvent.cacheReadInputTokens; if (usageEvent?.cacheCreationInputTokens !== undefined) usage.cache_creation_input_tokens = usageEvent.cacheCreationInputTokens;
  if (usageEvent?.estimated) usage.estimated = true;
  return Response.json({ id: start?.responseId || makeId("msg"), type: "message", role: "assistant", model, content: content.filter(Boolean), stop_reason: stop?.reason || "end_turn", stop_sequence: null, ...(stop?.details ? { stop_details: stop.details } : {}), usage });
}

export function countAnthropicTokens(body: Json): number {
  const encoder = new TextEncoder(); let bytes = 0; const walk = (value: unknown) => { if (typeof value === "string") { bytes += encoder.encode(value).byteLength; return; } if (Array.isArray(value)) { bytes += 2; value.forEach(walk); return; } if (value && typeof value === "object") { bytes += 2; for (const [key, item] of Object.entries(value)) { bytes += encoder.encode(key).byteLength + 2; walk(item); } } else bytes += 4; };
  walk(body.messages || []); walk(body.system || []); walk(body.tools || []); return Math.max(1, Math.ceil(bytes / 3));
}
