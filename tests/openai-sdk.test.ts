import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import OpenAI from "openai";
import type { AppConfig } from "../src/config.ts";
import type { ExecutorLike } from "../src/server.ts";
import { createHandler } from "../src/server.ts";

const servers: Bun.Server<unknown>[] = [];

function upstream(text: string): Response {
  const frames = [
    { id: "resp_upstream", choices: [{ index: 0, delta: { role: "assistant" } }] },
    {
      choices: [{ index: 0, delta: { content: text }, finish_reason: "stop" }],
      usage: { prompt_tokens: 3, completion_tokens: 2 },
    },
  ];
  return new Response(
    `${frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("")}data: [DONE]\n\n`,
    { headers: { "content-type": "text/event-stream" } }
  );
}

async function fixture() {
  const dir = await mkdtemp(join(tmpdir(), "kiro-sdk-"));
  const credentialFile = join(dir, "account.json");
  await writeFile(
    credentialFile,
    JSON.stringify({ accessToken: "access", refreshToken: "refresh", region: "us-east-1" }),
    { mode: 0o600 }
  );
  const config: AppConfig = {
    listen: "127.0.0.1:0",
    models: { "gpt-5.6-sol": { contextLength: 272_000 } },
    accounts: [{ credentialFile }],
  };
  const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
  const executor: ExecutorLike = {
    execute: async (input) => {
      seen.push(input);
      return { response: upstream("sdk-ok") };
    },
    refreshCredentials: async () => null,
  };
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: createHandler(config, executor),
  });
  servers.push(server);
  const client = new OpenAI({ apiKey: "local", baseURL: `${server.url}v1` });
  return { client, dir, seen };
}

afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true);
});

describe("official OpenAI SDK", () => {
  test("creates a non-streaming Response and sends image bytes", async () => {
    const { client, dir, seen } = await fixture();
    try {
      const response = await client.responses.create({
        model: "gpt-5.6-sol",
        input: [
          {
            role: "user",
            content: [
              { type: "input_text", text: "describe this image" },
              { type: "input_image", image_url: "data:image/png;base64,iVBORw0KGgo=", detail: "auto" },
            ],
          },
        ],
      });
      expect(response.object).toBe("response");
      expect(response.output_text).toBe("sdk-ok");
      expect(response.status).toBe("completed");
      expect(JSON.stringify(seen[0]?.body)).toContain('"images"');
      expect(JSON.stringify(seen[0]?.body)).toContain("iVBORw0KGgo=");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("iterates streaming Response events through the HTTP server", async () => {
    const { client, dir } = await fixture();
    try {
      const stream = await client.responses.create({
        model: "gpt-5.6-sol",
        input: "hello",
        stream: true,
      });
      const eventTypes: string[] = [];
      for await (const event of stream) eventTypes.push(event.type);
      expect(eventTypes).toContain("response.output_text.delta");
      expect(eventTypes.at(-1)).toBe("response.completed");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
