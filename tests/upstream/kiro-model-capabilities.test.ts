import { test, expect } from "bun:test";
import { buildKiroPayload } from "../../src/kiro/request/openai-to-kiro.ts";
import {
  KIRO_MODEL_CAPABILITIES,
  getKiroModelCapabilities,
  supportsKiroImages,
} from "../../src/kiro/modelCapabilities.ts";

const PNG = "data:image/png;base64,cmVhbC1pbWFnZS1ieXRlcw==";

function imagePayload(model: string) {
  return buildKiroPayload(
    model,
    {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Describe this image" },
            { type: "image_url", image_url: { url: PNG } },
          ],
        },
      ],
    },
    false,
    null
  );
}

test("capability table enables images and reasoning for all GPT-5.6 variants", () => {
  for (const model of ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]) {
    const capabilities = getKiroModelCapabilities(model);
    expect(capabilities).toEqual({
      contextLength: 272_000,
      supportsAdaptiveReasoning: false,
      supportsNativeReasoning: true,
    });
    expect(supportsKiroImages(model)).toBe(true);
  }
});

test("Claude target models use adaptive reasoning and retain image support", () => {
  for (const model of ["claude-opus-5", "claude-sonnet-5"]) {
    const capabilities = getKiroModelCapabilities(model);
    expect(supportsKiroImages(model)).toBe(true);
    expect(capabilities?.supportsAdaptiveReasoning).toBe(true);
    expect(capabilities?.supportsNativeReasoning).toBe(false);
    expect(capabilities?.contextLength).toBe(1_000_000);
  }
});

test("GPT image bytes reach userInputMessage.images", () => {
  const payload = imagePayload("gpt-5.6-luna");
  const images = payload.conversationState.currentMessage.userInputMessage.images;
  expect(images).toEqual([{ format: "png", source: { bytes: "cmVhbC1pbWFnZS1ieXRlcw==" } }]);
});

test("offline text-only denylist omits bytes and preserves a positional image notice", () => {
  for (const model of ["deepseek-3.2", "minimax-m2.1", "minimax-m2.5", "glm-5", "qwen3-coder-next"]) {
    expect(supportsKiroImages(model)).toBe(false);
    const images = imagePayload(model).conversationState.currentMessage.userInputMessage.images;
    expect(images).toBeUndefined();
    expect(imagePayload(model).conversationState.currentMessage.userInputMessage.content).toContain("[image omitted: this model does not support image input]");
  }
});

test("unknown models default to image support", () => {
  for (const model of ["unknown-model", "claude-haiku-5.5"]) {
    expect(supportsKiroImages(model)).toBe(true);
    expect(imagePayload(model).conversationState.currentMessage.userInputMessage.images).toHaveLength(1);
  }
});

test("capability context lengths match the configured Kiro catalog", () => {
  expect(Object.keys(KIRO_MODEL_CAPABILITIES)).toContain("gpt-5.6-sol");
  expect(KIRO_MODEL_CAPABILITIES["claude-opus-5"].contextLength).toBe(1_000_000);
});
