/**
 * OpenAI Chat Completions surface.
 *
 * The executor already emits OpenAI `chat.completion.chunk` SSE, so a streaming
 * request is served by forwarding that stream untouched — `/v1/messages` is the
 * endpoint doing extra work, translating those chunks into Anthropic events.
 *
 * A non-streaming request still has a stream to fold up, which is what this
 * module does, over the same protocol-neutral event reader the Anthropic
 * collector uses.
 */

import { readKiroEvents, type KiroEvent } from "./kiro/events.ts";

type ToolCall = { id: string; name: string; arguments: string };

/**
 * The event reader normalizes stop reasons to Anthropic's vocabulary; map back
 * to the four values the Chat Completions schema defines. `pause_turn` has no
 * OpenAI equivalent and is reported as a normal stop.
 */
const OPENAI_FINISH_REASONS: Record<string, string> = {
  end_turn: "stop",
  stop_sequence: "stop",
  pause_turn: "stop",
  tool_use: "tool_calls",
  max_tokens: "length",
  refusal: "content_filter",
};

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

  const chatUsage = usage && {
    prompt_tokens: usage.inputTokens,
    completion_tokens: usage.outputTokens,
    total_tokens: usage.inputTokens + usage.outputTokens,
    ...(usage.cacheReadInputTokens === undefined
      ? {}
      : { prompt_tokens_details: { cached_tokens: usage.cacheReadInputTokens } }),
    ...(usage.estimated ? { estimated: true } : {}),
  };

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
    ...(chatUsage ? { usage: chatUsage } : {}),
  });
}
