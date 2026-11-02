import { mergeKiroReasoning } from "../reasoning.ts";
import type { KiroHistoryItem, KiroUserInputMessageContext } from "./openai-to-kiro.ts";

export type KiroCurrentMessage = { userInputMessage: NonNullable<KiroHistoryItem["userInputMessage"]> };
const join = (a: string, b: string) => [a, b].filter(Boolean).join("\n\n");

/** Normalize both request surfaces into the same native conversation rules. */
export function normalizeKiroConversation(history: KiroHistoryItem[], currentMessage: KiroCurrentMessage, model: string) {
  const turns: KiroHistoryItem[] = structuredClone([...history, currentMessage]);
  const merged: KiroHistoryItem[] = [];
  for (const item of turns) {
    const previous = merged.at(-1);
    const user = item.userInputMessage;
    if (user && previous?.userInputMessage) {
      const target = previous.userInputMessage;
      target.content = join(target.content, user.content);
      if (user.images?.length) (target.images ??= []).push(...user.images);
      if (user.userInputMessageContext) {
        const context = target.userInputMessageContext ??= {};
        const next = user.userInputMessageContext;
        if (next.toolResults) (context.toolResults ??= []).push(...next.toolResults);
        if (next.tools) (context.tools ??= []).push(...next.tools);
        if (next.toolChoice) context.toolChoice = next.toolChoice;
      }
    } else if (item.assistantResponseMessage && previous?.assistantResponseMessage) {
      const assistant = item.assistantResponseMessage;
      const target = previous.assistantResponseMessage;
      const nextReasoning = assistant.reasoningContent;
      const reasoning = nextReasoning
        ? target.toolUses?.length ? undefined : mergeKiroReasoning(target.reasoningContent, nextReasoning)
        : target.reasoningContent;
      if (nextReasoning && !reasoning) {
        if (target.toolUses?.length) {
          assistant.toolUses = [...target.toolUses, ...(assistant.toolUses ?? [])];
          delete target.toolUses;
        }
        merged.push(item);
      } else {
        target.content = join(target.content, assistant.content);
        if (reasoning) target.reasoningContent = reasoning;
        if (assistant.toolUses?.length) (target.toolUses ??= []).push(...assistant.toolUses);
      }
    } else merged.push(item);
  }
  if (merged[0]?.assistantResponseMessage) merged.unshift({ userInputMessage: { content: "", modelId: model, origin: "AI_EDITOR" } });
  const normalized: KiroHistoryItem[] = [];
  for (const item of merged) {
    const user = item.userInputMessage;
    if (user) {
      const context = user.userInputMessageContext;
      if (context?.toolResults) {
        const ids = new Set(normalized.at(-1)?.assistantResponseMessage?.toolUses?.map(tool => tool.toolUseId));
        const matched: NonNullable<KiroUserInputMessageContext["toolResults"]> = [];
        for (const result of context.toolResults) {
          if (!result.content.map(part => part.text).join("\n").trim()) result.content = [{ text: "(no output)" }];
          if (ids.has(result.toolUseId)) matched.push(result);
          else user.content = join(user.content, `${result.toolUseId ? `[Tool Result (${result.toolUseId})]` : "[Tool Result]"}\n${result.content.map(part => part.text).join("\n")}`);
        }
        if (matched.length) context.toolResults = matched;
        else delete context.toolResults;
      }
      if (item !== merged.at(-1) && context) delete context.tools;
      if (context && !Object.keys(context).length) delete user.userInputMessageContext;
      if (!user.content?.trim() && !user.images?.length && !context?.toolResults?.length) user.content = "...";
      user.modelId ||= model;
      user.origin ||= "AI_EDITOR";
      if (normalized.at(-1)?.userInputMessage) normalized.push({ assistantResponseMessage: { content: "" } });
    }
    normalized.push(item);
  }
  return { history: normalized.slice(0, -1), currentMessage: normalized.at(-1) as KiroCurrentMessage };
}

/** Only converted native tool uses are authoritative for historical declarations. */
export function synthesizeHistoryToolSpecs(history: readonly KiroHistoryItem[]) {
  const names = new Set<string>();
  for (const item of history) for (const tool of item.assistantResponseMessage?.toolUses ?? []) if (tool.name) names.add(tool.name);
  return [...names].map(name => ({ toolSpecification: { name, description: `Tool: ${name}`, inputSchema: { json: { type: "object", properties: {} } } } }));
}

/** Lower accepted public tool-choice shapes after name sanitization. Parallel constraints are best effort. */
export function lowerKiroToolChoice(choice: unknown, declared: { sanitizedNames: ReadonlySet<string>; nameMap: ReadonlyMap<string, string> }): NonNullable<KiroUserInputMessageContext["toolChoice"]> {
  const object = choice && typeof choice === "object" && !Array.isArray(choice) ? choice as Record<string, unknown> : undefined;
  const type = typeof choice === "string" ? choice : object?.type;
  if (type === "auto") return { auto: {} };
  if (type === "none") return { none: {} };
  if (!declared.sanitizedNames.size) throw new Error("tool_choice requires declared tools");
  if (type === "any" || type === "required") return { any: {} };
  const fn = object?.function && typeof object.function === "object" ? object.function as Record<string, unknown> : undefined;
  const name = type === "function" ? fn?.name ?? object?.name : object?.name;
  if ((type === "tool" || type === "function" || type === "custom") && typeof name === "string" && name) {
    const sanitized = [...declared.nameMap].find(([, original]) => original === name)?.[0] ?? name;
    if (!declared.sanitizedNames.has(sanitized)) throw new Error(`tool_choice references undeclared tool: ${name}`);
    return { tool: { name: sanitized } };
  }
  throw new Error("Invalid tool_choice");
}
