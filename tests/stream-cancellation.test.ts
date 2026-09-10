import { expect, test } from "bun:test";
import { openAIStreamToAnthropic } from "../src/anthropic.ts";
import { responsesStream } from "../src/responses/response.ts";
import { KiroExecutor } from "../src/kiro/executor.ts";

for (const [name, adapt] of [["Anthropic", openAIStreamToAnthropic], ["Responses", responsesStream]] as const) {
  test(`${name} 在首块前和输出中取消均停止上游`, async () => {
    for (const started of [false, true]) {
      const cancelled = Promise.withResolvers<unknown>();
      const source = new ReadableStream<Uint8Array>({
        start(controller) {
          if (started) controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'));
        },
        cancel(reason) { cancelled.resolve(reason); },
      });
      const upstream = new Response(source);
      const reader = adapt(upstream, "test-model").body!.getReader();
      if (started) await reader.read();
      await reader.cancel("用户中止");
      expect(await cancelled.promise).toBe("用户中止");
      expect(await reader.read()).toEqual({ done: true, value: undefined });
    }
  });
}

test("转换后的响应取消会停止等待中的原始 HTTP 响应体", async () => {
  const cancelled = Promise.withResolvers<unknown>();
  const raw = new Response(new ReadableStream<Uint8Array>({ cancel(reason) { cancelled.resolve(reason); } }));
  const response = new KiroExecutor().transformEventStreamToSSE(raw, "claude-opus-5");
  await response.body!.cancel("停止原始流");
  expect(await cancelled.promise).toBe("停止原始流");
});
