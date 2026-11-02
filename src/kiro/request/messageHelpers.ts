// Pure message/tool helpers for the OpenAI -> Kiro request translator.
// Extracted verbatim from openai-to-kiro.ts (no host imports).

export function parseToolInput(value: unknown) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value;
  }
  if (typeof value !== "string") {
    return {};
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return {};
  }

  try {
    const parsed = JSON.parse(trimmed);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

export function serializeToolResultContent(content: unknown): string {
  if (typeof content === "string") {
    return content || "(no output)";
  }
  if (!Array.isArray(content)) {
    if (content !== null && content !== undefined) {
      try {
        return JSON.stringify(content);
      } catch {
        return "(no output)";
      }
    }
    return "(no output)";
  }
  const parts: string[] = [];
  for (const block of content as Array<Record<string, unknown>>) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "text" && typeof block.text === "string") {
      if (block.text) parts.push(block.text);
    } else if (block.type === "image" || block.type === "image_url") {
      const src = block.source as Record<string, unknown> | undefined;
      const mediaType = src?.media_type ?? block.media_type ?? "image";
      parts.push(`[image: ${mediaType}]`);
    } else {
      try {
        const str = JSON.stringify(block);
        if (str && str !== "{}") parts.push(str);
      } catch {
        // skip unserializable block
      }
    }
  }
  return parts.join("\n") || "(no output)";
}
