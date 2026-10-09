/**
 * Kiro enforces Anthropic's per-image cap on the base64 payload and rejects an
 * oversized image with a reasonless 400. Every entry checks it up front and
 * names the limit, so clients can run their image-shrink recovery instead of
 * treating the request as permanently malformed.
 */
export const MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024;

/** The "image exceeds" wording Anthropic uses, or undefined when the image fits. */
export function imageSizeViolation(base64: string): string | undefined {
  return base64.length > MAX_IMAGE_BASE64_BYTES
    ? `image exceeds 5 MB maximum: ${base64.length} bytes > ${MAX_IMAGE_BASE64_BYTES} bytes`
    : undefined;
}

/** First oversized data-URL image in Chat Completions `messages`, as an OpenAI-style error message. */
export function chatImageSizeError(messages: unknown[]): string | undefined {
  for (const [messageIndex, message] of messages.entries()) {
    const content = message && typeof message === "object" ? (message as Record<string, unknown>).content : undefined;
    if (!Array.isArray(content)) continue;
    for (const [partIndex, part] of content.entries()) {
      if (!part || typeof part !== "object") continue;
      const imageUrl = (part as Record<string, unknown>).image_url;
      const url = typeof imageUrl === "string"
        ? imageUrl
        : imageUrl && typeof imageUrl === "object" ? (imageUrl as Record<string, unknown>).url : undefined;
      if (typeof url !== "string" || !url.startsWith("data:")) continue;
      const violation = imageSizeViolation(url.slice(url.indexOf(",") + 1));
      if (violation) return `Invalid 'messages[${messageIndex}].content[${partIndex}].image_url': ${violation}`;
    }
  }
  return undefined;
}
