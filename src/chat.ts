/** OpenAI Chat Completions encoding over the executor's internal event stream. */

import { KiroStreamProtocolError, readKiroEvents, type KiroEvent } from "./kiro/events.ts";

type ToolCall = { id: string; name: string; arguments: string };

const OPENAI_FINISH_REASONS: Record<string, string> = {
  end_turn: "stop",
  stop_sequence: "stop",
  pause_turn: "stop",
  tool_use: "tool_calls",
  max_tokens: "length",
  refusal: "content_filter",
};

type StreamOptions = { onStreamFailure?: (code: string) => void };

function chatUsage(event: Extract<KiroEvent, { type: "usage" }>) {
  const prompt = event.inputTokens + (event.cacheReadInputTokens ?? 0) + (event.cacheCreationInputTokens ?? 0);
  return {
    prompt_tokens: prompt,
    completion_tokens: event.outputTokens,
    total_tokens: prompt + event.outputTokens,
    ...(event.cacheReadInputTokens === undefined ? {} : { prompt_tokens_details: { cached_tokens: event.cacheReadInputTokens } }),
  };
}

export function chatCompletionStream(response: Response, model: string, options: StreamOptions = {}): Response {
  const encoder = new TextEncoder();
  const abort = new AbortController();
  let responseId = `chatcmpl-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);
  const stream = new ReadableStream<Uint8Array>({ async start(controller) {
    const send = (data: unknown) => controller.enqueue(encoder.encode(`data: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`));
    const chunk = (delta: Record<string, unknown>, finish_reason: string | null = null) => send({ id: responseId, object: "chat.completion.chunk", created, model, choices: [{ index: 0, delta, logprobs: null, finish_reason }] });
    let usage: Extract<KiroEvent, { type: "usage" }> | undefined;
    try {
      for await (const event of readKiroEvents(response, model, abort.signal)) {
        if (abort.signal.aborted) return;
        switch (event.type) {
          case "start": if (event.responseId) responseId = event.responseId; chunk({ role: "assistant", content: "" }); break;
          case "text_delta": chunk({ content: event.text }); break;
          case "thinking_delta": chunk({ reasoning_content: event.thinking }); break;
          case "tool_start": chunk({ tool_calls: [{ index: event.index, id: event.id, type: "function", function: { name: event.name, arguments: "" } }] }); break;
          case "tool_input_delta": chunk({ tool_calls: [{ index: event.index, function: { arguments: event.delta } }] }); break;
          case "usage": usage = event; break;
          case "stop":
            chunk({}, OPENAI_FINISH_REASONS[event.reason]);
            if (usage) send({ id: responseId, object: "chat.completion.chunk", created, model, choices: [], usage: chatUsage(usage) });
            break;
        }
      }
      if (abort.signal.aborted) return;
      send("[DONE]");
      controller.close();
    } catch (error) {
      if (abort.signal.aborted) return;
      options.onStreamFailure?.(error instanceof KiroStreamProtocolError ? error.code : "stream_error");
      send({ error: { type: "api_error", code: "stream_error", message: "Upstream stream failed" } });
      controller.close();
    }
  }, cancel(reason) { abort.abort(reason); } });
  return new Response(stream, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" } });
}

export async function collectChatCompletion(response: Response, model: string): Promise<Response> {
  let responseId = `chatcmpl-${Date.now()}`;
  let content = "";
  let reasoning = "";
  const toolCalls: ToolCall[] = [];
  let usage: Extract<KiroEvent, { type: "usage" }> | undefined;
  let stop: Extract<KiroEvent, { type: "stop" }> | undefined;

  for await (const event of readKiroEvents(response, model)) {
    switch (event.type) {
      case "start":
        if (event.responseId) responseId = event.responseId;
        break;
      case "text_delta":
        content += event.text;
        break;
      case "thinking_delta":
        reasoning += event.thinking;
        break;
      case "tool_start":
        toolCalls[event.index] = { id: event.id, name: event.name, arguments: "" };
        break;
      case "tool_input_delta": {
        const call = toolCalls[event.index];
        if (call) call.arguments += event.delta;
        break;
      }
      case "usage":
        usage = event;
        break;
      case "stop":
        stop = event;
        break;
    }
  }
  if (!stop) throw new Error("Kiro stream ended before terminal event");

  // A tool index the upstream never opened leaves a hole in the array.
  const calls = toolCalls.filter(Boolean);
  const message: Record<string, unknown> = {
    role: "assistant",
    // Chat Completions reports a tool-only turn as a null content, not "".
    content: content || null,
    ...(reasoning ? { reasoning_content: reasoning } : {}),
    ...(calls.length
      ? {
          tool_calls: calls.map((call) => ({
            id: call.id,
            type: "function",
            function: { name: call.name, arguments: call.arguments },
          })),
        }
      : {}),
  };

  const usageValue = usage && chatUsage(usage);

  return Response.json({
    id: responseId,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message,
        logprobs: null,
        finish_reason: OPENAI_FINISH_REASONS[stop.reason] ?? "stop",
      },
    ],
    ...(usageValue ? { usage: usageValue } : {}),
  });
}
