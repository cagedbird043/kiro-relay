type StopReason = "end_turn" | "tool_use" | "max_tokens" | "stop_sequence";
export type KiroEvent =
  | { type: "start"; model: string; responseId?: string }
  | { type: "text_delta"; text: string }
  | { type: "thinking_delta"; thinking: string }
  | { type: "thinking_signature"; signature: string }
  | { type: "tool_start"; index: number; id: string; name: string }
  | { type: "tool_input_delta"; index: number; delta: string }
  | { type: "usage"; inputTokens: number; outputTokens: number; cacheReadInputTokens?: number; cacheCreationInputTokens?: number }
  | { type: "stop"; reason: StopReason };

type RecordValue = Record<string, unknown>;
const isRecord = (v: unknown): v is RecordValue => !!v && typeof v === "object";
const str = (v: unknown): string | undefined => typeof v === "string" ? v : undefined;
const num = (v: unknown): number | undefined => typeof v === "number" && Number.isFinite(v) ? v : undefined;

function stopReason(reason: unknown): StopReason {
  if (reason === "tool_calls" || reason === "tool_use") return "tool_use";
  if (reason === "length" || reason === "max_tokens") return "max_tokens";
  if (reason === "stop_sequence") return "stop_sequence";
  return "end_turn";
}

/** Parse the executor's OpenAI-compatible SSE, tolerating fragmented and multiline data frames. */
export async function* readKiroEvents(response: Response, model: string): AsyncGenerator<KiroEvent> {
  if (response.headers.get("content-type")?.includes("application/json")) {
    const raw = await response.text();
    const synthetic = new Response(`data: ${raw}\n\ndata: [DONE]\n\n`, {
      headers: { "content-type": "text/event-stream" },
    });
    yield* readKiroEvents(synthetic, model);
    return;
  }
  const reader = response.body?.getReader();
  if (!reader) { yield { type: "start", model }; yield { type: "stop", reason: "end_turn" }; return; }
  const decoder = new TextDecoder();
  let pending = "";
  let started = false;
  const tools = new Map<number, { id: string; name: string; started: boolean; pending: string; idSeen: boolean; nameSeen: boolean }>();
  let stopped = false;
  const start = (chunk?: RecordValue): KiroEvent | undefined => {
    if (started) return;
    started = true;
    const id = chunk && str(chunk.id);
    return { type: "start", model, ...(id ? { responseId: id } : {}) };
  };
  const parse = (raw: string): KiroEvent[] => {
    const out: KiroEvent[] = [];
    if (raw === "[DONE]") { for (const [index, tool] of tools) if (!tool.started) { tool.started = true; out.push({ type: "tool_start", index, id: tool.id, name: tool.name }); if (tool.pending) { out.push({ type: "tool_input_delta", index, delta: tool.pending }); tool.pending = ""; } } out.push({ type: "stop", reason: "end_turn" }); stopped = true; return out; }
    let value: unknown; try { value = JSON.parse(raw); } catch { return out; }
    if (!isRecord(value)) return out;
    const initial = start(value); if (initial) out.push(initial);
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const delta = isRecord(choice.delta) ? choice.delta : isRecord(choice.message) ? choice.message : {};
    const reasoning = str(delta.reasoning_content) ?? str(delta.thinking);
    if (reasoning) out.push({ type: "thinking_delta", thinking: reasoning });
    const signature = str(delta.reasoning_signature);
    if (signature) out.push({ type: "thinking_signature", signature });
    const text = typeof delta.content === "string" ? delta.content : isRecord(delta.content) ? str(delta.content.text) : undefined;
    if (text) out.push({ type: "text_delta", text });
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const item of calls) {
      if (!isRecord(item)) continue;
      const index = num(item.index) ?? 0;
      const fn = isRecord(item.function) ? item.function : {};
      const prior = tools.get(index) ?? { id: `tool_${index}`, name: "", started: false, pending: "", idSeen: false, nameSeen: false };
      const id = str(item.id); const name = str(fn.name) ?? str(item.name);
      if (id !== undefined) { prior.id = id; prior.idSeen = true; }
      if (name !== undefined) { prior.name = name; prior.nameSeen = true; }
      tools.set(index, prior);
      const args = str(fn.arguments) ?? str(item.arguments);
      if (!prior.started && prior.idSeen && prior.nameSeen) {
        prior.started = true;
        out.push({ type: "tool_start", index, id: prior.id, name: prior.name });
        if (prior.pending) { out.push({ type: "tool_input_delta", index, delta: prior.pending }); prior.pending = ""; }
      }
      if (args) { if (prior.started) out.push({ type: "tool_input_delta", index, delta: args }); else prior.pending += args; }
    }
    const usage = isRecord(value.usage) ? value.usage : undefined;
    if (usage) {
      const inputTokens = num(usage.prompt_tokens) ?? num(usage.input_tokens) ?? num(usage.inputTokens) ?? 0;
      const outputTokens = num(usage.completion_tokens) ?? num(usage.output_tokens) ?? num(usage.outputTokens) ?? 0;
      const cacheReadInputTokens = num(usage.cache_read_input_tokens) ?? num(usage.cacheReadInputTokens) ?? num(usage.cache_read_tokens);
      const cacheCreationInputTokens = num(usage.cache_creation_input_tokens) ?? num(usage.cacheCreationInputTokens) ?? num(usage.cache_write_input_tokens);
      out.push({ type: "usage", inputTokens, outputTokens, ...(cacheReadInputTokens === undefined ? {} : { cacheReadInputTokens }), ...(cacheCreationInputTokens === undefined ? {} : { cacheCreationInputTokens }) });
    }
    const reason = choice.finish_reason; if (reason !== undefined && reason !== null) {
      for (const [index, tool] of tools) if (!tool.started) { tool.started = true; out.push({ type: "tool_start", index, id: tool.id, name: tool.name }); if (tool.pending) { out.push({ type: "tool_input_delta", index, delta: tool.pending }); tool.pending = ""; } }
      out.push({ type: "stop", reason: stopReason(reason) });
    }
    return out;
  };
  try {
    while (!stopped) {
      const { value, done } = await reader.read();
      if (done) { pending += decoder.decode(); break; }
      pending += decoder.decode(value, { stream: true });
      const lines = pending.split(/\r?\n/); pending = lines.pop() ?? "";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("data:")) data += (data ? "\n" : "") + line.slice(5).trim();
        else if (!line.trim() && data) { for (const event of parse(data)) yield event; data = ""; if (stopped) break; }
      }
      if (data) pending = `data: ${data}\n\n${pending}`;
    }
    if (!stopped && pending.trim()) {
      const match = pending.match(/^data:\s*([\s\S]*?)(?:\r?\n|$)/); if (match) for (const event of parse(match[1].trim())) yield event;
    }
    if (!started) { const initial = start(); if (initial) yield initial; }
    if (!stopped) yield { type: "stop", reason: "end_turn" };
  } finally { reader.releaseLock(); }
}
