/**
 * Kiro conversationState 的结构规则，按实测上游行为整理（不符合时 Kiro 返回
 * REQUEST_BODY_INVALID 或 Improperly formed request）。三个入口生成的载荷都必须满足。
 */
type ToolUse = { toolUseId: string };
type Turn = {
  userInputMessage?: {
    content?: string;
    modelId?: string;
    origin?: string;
    images?: unknown[];
    userInputMessageContext?: { toolResults?: Array<{ toolUseId: string; content?: Array<{ text?: string }> }> };
  };
  assistantResponseMessage?: { content?: string; reasoningContent?: unknown; toolUses?: ToolUse[] };
};
export type KiroState = { history: Turn[]; currentMessage: Turn };

export function kiroHistoryViolations(state: KiroState): string[] {
  const violations: string[] = [];
  const turns = [...state.history, state.currentMessage];
  if (!state.currentMessage?.userInputMessage) violations.push("currentMessage 必须是用户回合");
  if (state.history.length > 0 && !state.history[0].userInputMessage) violations.push("history 必须以用户回合开头");
  const seen = new Set<string>();
  turns.forEach((turn, index) => {
    const user = turn.userInputMessage;
    const assistant = turn.assistantResponseMessage;
    if (user && assistant) violations.push(`#${index} 同时是用户和助手回合`);
    if (!user && !assistant) violations.push(`#${index} 既不是用户也不是助手回合`);
    if (user && turns[index - 1]?.userInputMessage) violations.push(`#${index} 连续用户回合`);
    if (user && index < state.history.length && (!user.modelId || !user.origin)) violations.push(`#${index} 用户回合缺少 modelId 或 origin`);
    if (user && !user.content?.trim() && !user.images?.length && !user.userInputMessageContext?.toolResults?.length) {
      violations.push(`#${index} 用户回合正文为空且没有工具结果或图片`);
    }
    if (assistant?.toolUses?.length) {
      if (turns[index + 1]?.assistantResponseMessage) violations.push(`#${index} 连续助手消息中非最后一条带 toolUses`);
      for (const use of assistant.toolUses) {
        if (seen.has(use.toolUseId)) violations.push(`toolUseId 重复：${use.toolUseId}`);
        seen.add(use.toolUseId);
      }
    }
    if (user) {
      const expected = turns[index - 1]?.assistantResponseMessage?.toolUses?.map((use) => use.toolUseId) ?? [];
      const results = user.userInputMessageContext?.toolResults ?? [];
      const got = results.map((result) => result.toolUseId);
      for (const id of got) if (!expected.includes(id)) violations.push(`#${index} 工具结果 ${id} 没有对应的工具调用`);
      for (const id of expected) if (!got.includes(id)) violations.push(`#${index} 工具调用 ${id} 缺少工具结果`);
      if (new Set(got).size !== got.length) violations.push(`#${index} 工具结果重复`);
      if (results.some((result) => !result.content?.length || result.content.some((part) => !part.text))) {
        violations.push(`#${index} 工具结果正文为空`);
      }
    }
  });
  return violations;
}

/** 历史中所有助手推理槽，按出现顺序。 */
export const reasoningSequence = (state: KiroState) =>
  state.history.flatMap((turn) => (turn.assistantResponseMessage?.reasoningContent ? [turn.assistantResponseMessage.reasoningContent] : []));

/** 历史中所有工具调用 ID，按出现顺序。 */
export const toolUseSequence = (state: KiroState) =>
  state.history.flatMap((turn) => turn.assistantResponseMessage?.toolUses?.map((use) => use.toolUseId) ?? []);

/** 历史中所有助手正文，按出现顺序拼接。 */
export const assistantText = (state: KiroState) =>
  state.history.map((turn) => turn.assistantResponseMessage?.content ?? "").filter(Boolean).join("\n\n");

/** 一次助手回合可能出现的块：签名思考、隐藏思考、正文、工具调用。 */
export type BlockKind = "S" | "R" | "X" | "U";

/** 枚举长度 1..maxLength 的全部块排列。 */
export function* blockSequences(maxLength: number, kinds: readonly BlockKind[] = ["S", "R", "X", "U"]): Generator<BlockKind[]> {
  const walk = function* (prefix: BlockKind[]): Generator<BlockKind[]> {
    if (prefix.length > 0) yield prefix;
    if (prefix.length === maxLength) return;
    for (const kind of kinds) yield* walk([...prefix, kind]);
  };
  yield* walk([]);
}

/** 线上形状指纹：只保留角色、推理槽类型、是否有正文和工具调用个数，用于去重。 */
export function wireShape(state: KiroState): string {
  return [...state.history, state.currentMessage].map((turn) => {
    if (turn.userInputMessage) return `U${turn.userInputMessage.userInputMessageContext?.toolResults?.length ?? 0}`;
    const assistant = turn.assistantResponseMessage!;
    const reasoning = assistant.reasoningContent
      ? "redactedContent" in (assistant.reasoningContent as object) ? "r" : "s"
      : "";
    return `A${reasoning}${assistant.content ? "x" : ""}${assistant.toolUses?.length ?? 0}`;
  }).join(" ");
}
