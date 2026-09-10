// Kiro 原生推理联合类型；JSON 传输中的二进制内容使用 base64。
export type KiroReasoningContent =
  | { reasoningText: { text: string; signature?: string } }
  | { redactedContent: string };


// 保留原生签名或隐藏内容的类型和模型归属；客户端应将此字段视为不透明数据。
export function encodeReasoning(model: string, content: KiroReasoningContent): string {
  return `kiro:${Buffer.from(JSON.stringify({ model, content })).toString("base64url")}`;
}

export function decodeReasoning(value: string, model: string): KiroReasoningContent {
  if (!value.startsWith("kiro:")) throw new Error("reasoning.encrypted_content must originate from this Kiro relay");
  let envelope: unknown;
  try { envelope = JSON.parse(Buffer.from(value.slice(5), "base64url").toString("utf8")); }
  catch { throw new Error("Invalid Kiro reasoning content"); }
  if (!envelope || typeof envelope !== "object" || !("model" in envelope) || envelope.model !== model || !("content" in envelope) || !envelope.content || typeof envelope.content !== "object") {
    throw new Error("Kiro reasoning content must belong to the requested model");
  }
  const content = envelope.content;
  if ("redactedContent" in content && typeof content.redactedContent === "string" && content.redactedContent.length > 0 && !("reasoningText" in content)) {
    return { redactedContent: content.redactedContent };
  }
  if ("reasoningText" in content && content.reasoningText && typeof content.reasoningText === "object" && "text" in content.reasoningText && typeof content.reasoningText.text === "string" && !("redactedContent" in content)) {
    const { text } = content.reasoningText;
    const signature = "signature" in content.reasoningText ? content.reasoningText.signature : undefined;
    if (signature !== undefined && typeof signature !== "string") throw new Error("Invalid Kiro reasoning signature");
    return { reasoningText: { text, ...(typeof signature === "string" ? { signature } : {}) } };
  }
  throw new Error("Invalid Kiro reasoning content");
}
