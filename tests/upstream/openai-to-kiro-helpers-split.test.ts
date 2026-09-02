import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// Split-guard for the openai-to-kiro message-helper extraction.
// The pure tool/message helpers (parseToolInput / normalizeKiroToolSchema /
// serializeToolResultContent) live in the sibling leaf `request/messageHelpers.ts`;
// the host imports them back. They were module-private, so the public export set is
// unchanged (no re-export needed).
const HERE = dirname(fileURLToPath(import.meta.url));
const REQ = join(HERE, "../../src/kiro/request");
const HOST = join(REQ, "openai-to-kiro.ts");
const LEAF = join(REQ, "messageHelpers.ts");

test("leaf hosts the pure helpers and does not import the host", () => {
  const src = readFileSync(LEAF, "utf8");
  for (const sym of ["parseToolInput", "normalizeKiroToolSchema", "serializeToolResultContent"]) {
    assert.match(src, new RegExp(`export function ${sym}\\b`));
  }
  assert.doesNotMatch(src, /from "\.\.\/openai-to-kiro\.ts"/);
});

test("host imports the helpers back from the leaf", () => {
  const src = readFileSync(HOST, "utf8");
  assert.match(src, /from "\.\/messageHelpers\.ts"/);
  // buildKiroPayload stays exported on the host module.
  assert.match(src, /export function buildKiroPayload\(/);
});

test("normalizeKiroToolSchema strips empty required arrays and additionalProperties", async () => {
  const { normalizeKiroToolSchema } =
    await import("../../src/kiro/request/messageHelpers.ts");
  const out = normalizeKiroToolSchema({
    type: "object",
    required: [],
    additionalProperties: false,
    properties: { a: { type: "string" } },
  });
  assert.equal("required" in out, false);
  assert.equal("additionalProperties" in out, false);
  assert.deepEqual(out.properties, { a: { type: "string" } });
});
