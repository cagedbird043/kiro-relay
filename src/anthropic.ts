/* Anthropic <-> Kiro wire adapters. */
import { readKiroEvents, type KiroEvent } from "./kiro/events.ts";
type Json = Record<string, unknown>;

/** Kiro enforces Anthropic's per-image cap on the base64 payload; larger images fail upstream with a bare 400. */
const MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024;

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
  // Mirrors Anthropic's own rejection wording so clients run their image-shrink recovery instead of
  // treating Kiro's reasonless 400 as a fatal request error.
  const checkImageSize = (data: unknown, path: string) => {
    if (typeof data === "string" && data.length > MAX_IMAGE_BASE64_BYTES) {
      reject(`${path}: image exceeds 5 MB maximum: ${data.length} bytes > ${MAX_IMAGE_BASE64_BYTES} bytes`);
    }
  };
  const validateBlock = (block: unknown, location: string, path: string) => {
    if (!block || typeof block !== "object") return;
    const b = block as Json;
    if (b.type === "document") reject(`Unsupported ${location} document content`);
    const source = b.source && typeof b.source === "object" ? b.source as Json : undefined;
    const imageUrl = b.image_url && typeof b.image_url === "object" ? b.image_url as Json : undefined;
    const remoteImage =
      (b.type === "image" && source?.type === "url") ||
      (b.type === "image_url" && typeof imageUrl?.url === "string" && !imageUrl.url.startsWith("data:"));
    if (remoteImage) reject(`Unsupported ${location} remote image URL`);
    if (b.type === "image" && source?.type === "base64") checkImageSize(source.data, `${path}.image.source.base64`);
    if (b.type === "image_url" && typeof imageUrl?.url === "string" && imageUrl.url.startsWith("data:")) {
      checkImageSize(imageUrl.url.slice(imageUrl.url.indexOf(",") + 1), `${path}.image_url.url`);
    }
    if (b.type === "tool_result" && Array.isArray(b.content)) {
      b.content.forEach((item: unknown, index: number) => {
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
        if (validBase64Image) checkImageSize(imageSource.data, `${path}.content.${index}.image.source.base64`);
      });
    }
  };
  if (Array.isArray(body.messages)) {
    body.messages.forEach((message: unknown, messageIndex: number) => {
      if (!message || typeof message !== "object") return;
      const content = (message as Json).content;
      if (Array.isArray(content)) {
        content.forEach((block, blockIndex) => validateBlock(block, "message", `messages.${messageIndex}.content.${blockIndex}`));
      }
    });
  }
  if (Array.isArray(body.system)) {
    body.system.forEach((block, index) => validateBlock(block, "system", `system.${index}`));
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

class AnthropicContent {
  readonly blocks: Block[] = [];
  readonly toolInputs = new Map<number, string>();
  private readonly tools = new Map<number, number>();
  private current?: number;
  private lastReasoning?: number;

  private add(block: Block): number {
    const index = this.blocks.length;
    this.blocks.push(block);
    this.current = index;
    if (block.type === "thinking" || block.type === "redacted_thinking") this.lastReasoning = index;
    return index;
  }

  accept(event: KiroEvent): void {
    if (event.type === "text_delta" || event.type === "thinking_delta") {
      const type = event.type === "text_delta" ? "text" : "thinking";
      const index = this.current !== undefined && this.blocks[this.current].type === type ? this.current : this.add({ type });
      const block = this.blocks[index];
      block[type] = (block[type] ?? "") + (event.type === "text_delta" ? event.text : event.thinking);
    } else if (event.type === "thinking_signature") {
      const index = this.lastReasoning !== undefined && this.blocks[this.lastReasoning].type === "thinking"
        ? this.lastReasoning : this.add({ type: "thinking", thinking: "" });
      const block = this.blocks[index];
      block.signature = (block.signature ?? "") + event.signature;
    } else if (event.type === "thinking_redacted") {
      this.add({ type: "redacted_thinking", data: event.data });
    } else if (event.type === "tool_start") {
      if (!this.tools.has(event.index)) this.tools.set(event.index, this.add({ type: "tool_use", id: event.id, name: event.name, input: {} }));
    } else if (event.type === "tool_input_delta") {
      const index = this.tools.get(event.index);
      if (index === undefined) throw new Error(`Unknown tool index ${event.index}`);
      this.toolInputs.set(index, (this.toolInputs.get(index) ?? "") + event.delta);
    }
  }

  finish(): Block[] {
    for (const [index, raw] of this.toolInputs) {
      try { this.blocks[index].input = JSON.parse(raw); } catch { this.blocks[index].input = raw; }
    }
    return this.blocks;
  }
}

function anthropicUsage(event?: Extract<KiroEvent, { type: "usage" }>): Json {
  return {
    input_tokens: event?.inputTokens ?? 0,
    output_tokens: event?.outputTokens ?? 0,
    ...(event?.cacheReadInputTokens === undefined ? {} : { cache_read_input_tokens: event.cacheReadInputTokens }),
    ...(event?.cacheCreationInputTokens === undefined ? {} : { cache_creation_input_tokens: event.cacheCreationInputTokens }),
    ...(event?.estimated ? { estimated: true } : {}),
  };
}

export function openAIStreamToAnthropic(response: Response, model = "unknown"): Response {
  const encoder = new TextEncoder(); const messageId = makeId("msg");
  const abort = new AbortController();
  let ended = false;
  let timer: NodeJS.Timeout | undefined;
  const stream = new ReadableStream<Uint8Array>({ async start(controller) {
    const content = new AnthropicContent();
    let started = false;
    let cursor = 0; let blockStarted = false; let sentText = 0; let sentSignature = 0;
    let usage: Extract<KiroEvent, { type: "usage" }> | undefined;
    let stop: Extract<KiroEvent, { type: "stop" }> | undefined;
    const emit = (event: string, data: Json) => { if (!ended) controller.enqueue(encoder.encode(sse(event, data))); };
    const start = (id = messageId) => {
      if (started) return;
      started = true;
      emit("message_start", { type: "message_start", message: { id, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } });
    };
    const pump = (final = false) => {
      while (cursor < content.blocks.length) {
        const block = content.blocks[cursor];
        if (!blockStarted) {
          const initial: Block = block.type === "tool_use" ? { type: block.type, id: block.id, name: block.name, input: {} }
            : block.type === "thinking" ? { type: block.type, thinking: "" }
            : block.type === "text" ? { type: block.type, text: "" } : block;
          emit("content_block_start", { type: "content_block_start", index: cursor, content_block: initial });
          blockStarted = true;
        }
        const text = block.type === "tool_use" ? content.toolInputs.get(cursor) ?? "" : block.type === "thinking" ? block.thinking ?? "" : block.text ?? "";
        if (text.length > sentText) {
          const part = text.slice(sentText);
          const delta = block.type === "tool_use" ? { type: "input_json_delta", partial_json: part }
            : block.type === "thinking" ? { type: "thinking_delta", thinking: part } : { type: "text_delta", text: part };
          emit("content_block_delta", { type: "content_block_delta", index: cursor, delta });
          sentText = text.length;
        }
        if (block.signature && block.signature.length > sentSignature) {
          emit("content_block_delta", { type: "content_block_delta", index: cursor, delta: { type: "signature_delta", signature: block.signature.slice(sentSignature) } });
          sentSignature = block.signature.length;
        }
        // 工具参数和签名仍可能晚到；保持当前块开放，后续块按原顺序缓冲。
        if (!final && block.type !== "redacted_thinking" && !(block.type === "text" && cursor + 1 < content.blocks.length)) break;
        emit("content_block_stop", { type: "content_block_stop", index: cursor });
        cursor++; blockStarted = false; sentText = 0; sentSignature = 0;
      }
    };
    timer = setInterval(() => { if (!ended) emit("ping", { type: "ping" }); }, 15000);
    try {
      for await (const event of readKiroEvents(response, model, abort.signal)) {
        if (ended) return;
        if (event.type === "start") start(event.responseId);
        content.accept(event);
        if (event.type === "usage") usage = event;
        else if (event.type === "stop") stop = event;
        pump();
      }
      if (ended) return;
      if (!stop) throw new Error("Kiro stream ended before terminal event");
      start();
      pump(true);
      emit("message_delta", { type: "message_delta", delta: { stop_reason: stop.reason, stop_sequence: null, ...(stop.details ? { stop_details: stop.details } : {}) }, usage: anthropicUsage(usage) });
      emit("message_stop", { type: "message_stop" });
      ended = true;
      controller.close();
    } catch (error) {
      if (ended) return;
      start();
      emit("error", { type: "error", error: { type: "api_error", message: error instanceof Error ? error.message : "upstream stream failed" } });
      ended = true;
      controller.close();
    } finally { clearInterval(timer); }
  }, cancel(reason) { ended = true; clearInterval(timer); abort.abort(reason); } });
  return new Response(stream, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" } });
}

export async function collectAnthropicResponse(response: Response, model = "unknown"): Promise<Response> {
  const content = new AnthropicContent();
  let responseId = makeId("msg");
  let usage: Extract<KiroEvent, { type: "usage" }> | undefined;
  let stop: Extract<KiroEvent, { type: "stop" }> | undefined;
  for await (const event of readKiroEvents(response, model)) {
    content.accept(event);
    if (event.type === "start" && event.responseId) responseId = event.responseId;
    else if (event.type === "usage") usage = event;
    else if (event.type === "stop") stop = event;
  }
  if (!stop) throw new Error("Kiro stream ended before terminal event");
  return Response.json({ id: responseId, type: "message", role: "assistant", model, content: content.finish(), stop_reason: stop.reason, stop_sequence: null, ...(stop.details ? { stop_details: stop.details } : {}), usage: anthropicUsage(usage) });
}

export function countAnthropicTokens(body: Json): number {
  const encoder = new TextEncoder(); let bytes = 0; const walk = (value: unknown) => { if (typeof value === "string") { bytes += encoder.encode(value).byteLength; return; } if (Array.isArray(value)) { bytes += 2; value.forEach(walk); return; } if (value && typeof value === "object") { bytes += 2; for (const [key, item] of Object.entries(value)) { bytes += encoder.encode(key).byteLength + 2; walk(item); } } else bytes += 4; };
  walk(body.messages || []); walk(body.system || []); walk(body.tools || []); return Math.max(1, Math.ceil(bytes / 3));
}
