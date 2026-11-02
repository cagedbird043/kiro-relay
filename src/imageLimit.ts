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

export const UNSUPPORTED_MODEL_IMAGE = "[image omitted: this model does not support image input]";
export const REMOTE_IMAGE = "[image omitted: remote image URLs are not supported, send base64 inline]";

/** All inline image forms accepted by the request translators. */
export function parseInlineImage(value: unknown): { format: string; bytes: string } | undefined {
  if (!value || typeof value !== "object") return undefined;
  const block = value as Record<string, unknown>;
  const source = block.source && typeof block.source === "object" ? block.source as Record<string, unknown> : undefined;
  if (block.type === "image" && source?.type === "base64" && typeof source.data === "string" && source.data) {
    const mediaType = typeof source.media_type === "string" ? source.media_type : "image/jpeg";
    return { format: (mediaType.split("/")[1] || "jpeg").toLowerCase().replace(/^jpg$/, "jpeg"), bytes: source.data };
  }
  const imageUrl = block.image_url && typeof block.image_url === "object" ? block.image_url as Record<string, unknown> : undefined;
  const url = block.type === "image" ? block.image : typeof block.image_url === "string" ? block.image_url : imageUrl?.url;
  if (typeof url !== "string") return undefined;
  const match = /^data:image\/([^;,]+);base64,([A-Za-z0-9+/]+={0,2})$/i.exec(url);
  if (!match) return undefined;
  return { format: match[1].toLowerCase().replace(/^jpg$/, "jpeg"), bytes: match[2] };
}

/** Size checks traverse embedded tool observations as well as ordinary content. */
export function chatImageSizeError(messages: unknown[]): string | undefined {
  const inspect = (content: unknown, path: string): string | undefined => {
    if (!Array.isArray(content)) return undefined;
    for (const [index, part] of content.entries()) {
      const image = parseInlineImage(part);
      const violation = image && imageSizeViolation(image.bytes);
      if (violation) return `Invalid '${path}[${index}]${part?.type === "image_url" ? ".image_url" : ""}': ${violation}`;
      if (part?.type === "tool_result") {
        const nested = inspect(part.content, `${path}[${index}].content`);
        if (nested) return nested;
      }
    }
  };
  for (const [index, message] of messages.entries()) {
    const error = inspect(message && typeof message === "object" ? (message as Record<string, unknown>).content : undefined, `messages[${index}].content`);
    if (error) return error;
  }
}
