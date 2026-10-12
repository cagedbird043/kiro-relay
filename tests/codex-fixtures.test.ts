import { afterEach, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AppConfig } from "../src/config.ts";
import type { KiroResponsesRequest } from "../src/responses/request.ts";
import { createHandler, type ExecutorLike } from "../src/server.ts";

const dirs: string[] = [];

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

async function fixture(name: string): Promise<Record<string, unknown>> {
  const value: unknown = await Bun.file(new URL(`./fixtures/codex/${name}`, import.meta.url)).json();
  if (!isRecord(value)) throw new Error(`Invalid Codex fixture: ${name}`);
  return value;
}

function upstreamToolCall(): Response {
  const chunks = [
    {
      id: "resp_fixture",
      choices: [
        {
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_fixture",
                function: {
                  name: "functions__exec",
                  arguments: '{"input":"text(\\\"fixture-ok\\\")"}',
                },
              },
            ],
          },
        },
      ],
    },
    { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
  ];
  return new Response(chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join(""));
}

async function setup() {
  const dir = await mkdtemp(join(tmpdir(), "kiro-codex-fixture-"));
  dirs.push(dir);
  const credentialFile = join(dir, "account.json");
  await writeFile(credentialFile, JSON.stringify({ accessToken: "token", region: "us-east-1" }), {
    mode: 0o600,
  });
  const config: AppConfig = {
    listen: "127.0.0.1:0",
    models: { "gpt-5.6-luna": { contextLength: 272_000 } },
    accounts: [{ credentialFile }],
  };
  return { config };
}

afterEach(async () => {
  await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

test("Codex Responses Lite fixture produces a custom tool call", async () => {
  const requestFixture = await fixture("responses-lite-request.json");
  const expectedCall = await fixture("custom-tool-call.json");
  const { config } = await setup();
  const seen: Parameters<ExecutorLike["execute"]>[0][] = [];
  const executor: ExecutorLike = {
    execute: async (input) => {
      seen.push(input);
      return { response: upstreamToolCall() };
    },
    refreshCredentials: async () => null,
  };
  const handler = createHandler(config, executor);
  const body = {
    ...requestFixture,
    model: "gpt-5.6-luna",
    stream: true,
  } satisfies KiroResponsesRequest;
  const response = await handler(
    new Request("http://relay/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    })
  );
  expect(response.status).toBe(200);
  const events = (await response.text())
    .split("\n")
    .filter((line) => line.startsWith("data: ") && line !== "data: [DONE]")
    .map((line) => JSON.parse(line.slice(6)));
  const callDone = events.find(
    (event) => event.type === "response.output_item.done" && event.item?.type === "custom_tool_call"
  );
  const { id: _fixtureId, ...expectedFields } = expectedCall;
  expect(callDone?.item).toMatchObject(expectedFields);
  expect(JSON.stringify(seen[0]?.body)).toContain('"name":"functions__exec"');
  expect(JSON.stringify(seen[0]?.body)).toContain('"input":{"type":"string"}');
});

test("Codex custom tool output fixture becomes a Kiro tool result", async () => {
  const requestFixture = await fixture("responses-lite-request.json");
  const call = await fixture("custom-tool-call.json");
  const output = await fixture("custom-tool-output.json");
  const input = requestFixture.input;
  if (!Array.isArray(input)) throw new Error("Fixture input must be an array");
  if (typeof requestFixture.model !== "string") throw new Error("Fixture model must be a string");
  const body = {
    ...requestFixture,
    model: requestFixture.model,
    stream: false,
    input: [...input, call, output],
  } satisfies KiroResponsesRequest;
  const { config } = await setup();
  let seenBody: unknown;
  const executor: ExecutorLike = {
    execute: async (request) => {
      seenBody = request.body;
      return {
        response: new Response(
          `data: ${JSON.stringify({ choices: [{ delta: { content: "done" }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`
        ),
      };
    },
    refreshCredentials: async () => null,
  };
  const response = await createHandler(config, executor)(
    new Request("http://relay/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    })
  );
  expect(response.status).toBe(200);
  expect(JSON.stringify(seenBody)).toContain('"toolUseId":"call_fixture"');
  expect(JSON.stringify(seenBody)).toContain('"text":"fixture-ok"');
});
