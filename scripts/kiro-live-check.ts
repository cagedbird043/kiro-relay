/**
 * 真实 Kiro 安全门：把会触发上游结构校验的历史形状逐个发给一个运行中的 relay，
 * 要求全部返回 200。不进 CI，在改动请求转换的大重构前后手动运行：
 *
 *   KIRO_RELAY_KEY=... bun run live-check            # 默认 http://127.0.0.1:4000
 *   KIRO_RELAY_URL=https://relay.example KIRO_RELAY_KEY=... bun run live-check
 *
 * 被测的是 relay 运行的那份代码，所以验证重构时先用工作区代码启动本地 relay。
 * 签名推理必须是 Kiro 真签发的，脚本先向 relay 请求几次真实回答来取得签名。
 * 会消耗少量 Kiro 额度；只打印用例名和状态码，不打印正文、签名或密钥。
 */
import { anthropicToOpenAI } from "../src/anthropic.ts";
import { buildKiroPayload } from "../src/kiro/request/openai-to-kiro.ts";
import { blockSequences, wireShape, type BlockKind, type KiroState } from "../tests/kiro-history-rules.ts";

const BASE = (process.env.KIRO_RELAY_URL ?? "http://127.0.0.1:4000").replace(/\/+$/, "");
const KEY = process.env.KIRO_RELAY_KEY ?? "";
const CLAUDE = process.env.KIRO_LIVE_CLAUDE_MODEL ?? "claude-opus-5-5";
const GPT = process.env.KIRO_LIVE_GPT_MODEL ?? "gpt-5.6-sol";
const CONCURRENCY = 4;

type Json = Record<string, unknown>;
const headers = { "content-type": "application/json", "anthropic-version": "2023-06-01", ...(KEY ? { "x-api-key": KEY } : {}) };

/** 流式请求，返回 "200" 或失败原因；流内 error 事件也算失败。 */
async function send(path: string, body: Json): Promise<string> {
  try {
    const response = await fetch(`${BASE}${path}`, { method: "POST", headers, body: JSON.stringify({ ...body, stream: true }), signal: AbortSignal.timeout(180_000) });
    const text = await response.text();
    if (!response.ok) return `${response.status} ${text.slice(0, 160)}`;
    const errorAt = text.search(/event: error|"type":"response\.failed"/);
    return errorAt < 0 ? "200" : `stream error ${text.slice(errorAt, errorAt + 160)}`;
  } catch (error) {
    return `request failed: ${error instanceof Error ? error.message : String(error)}`;
  }
}

async function post(path: string, body: Json): Promise<Json> {
  const response = await fetch(`${BASE}${path}`, { method: "POST", headers, body: JSON.stringify(body), signal: AbortSignal.timeout(180_000) });
  if (!response.ok) throw new Error(`${path} ${response.status}: ${(await response.text()).slice(0, 200)}`);
  return await response.json() as Json;
}

async function pool<T>(jobs: Array<() => Promise<T>>): Promise<T[]> {
  const results: T[] = new Array(jobs.length);
  let next = 0;
  await Promise.all(Array.from({ length: CONCURRENCY }, async () => {
    while (next < jobs.length) {
      const index = next++;
      results[index] = await jobs[index]();
    }
  }));
  return results;
}

/** 取得 Kiro 真签发的 Claude 思考签名。 */
async function claudeSignatures(count: number): Promise<string[]> {
  const replies = await pool(Array.from({ length: count }, (_, index) => () => post("/v1/messages", {
    model: CLAUDE, max_tokens: 2048, thinking: { type: "adaptive" }, output_config: { effort: "high" },
    messages: [{ role: "user", content: `先仔细思考再回答：${17 + index} 乘以 23 等于多少？` }],
  })));
  const signatures = replies.flatMap((reply) => ((reply.content as Json[]) ?? [])
    .flatMap((block) => (block.type === "thinking" && typeof block.signature === "string" && block.signature ? [block.signature] : [])));
  if (signatures.length === 0) throw new Error(`${CLAUDE} 没有返回签名思考，无法构造签名用例`);
  return signatures;
}

/** 取得 Kiro 真签发的 GPT 推理（relay 的 encrypted_content 封装）。 */
async function gptReasoning(count: number): Promise<Json[]> {
  const replies = await pool(Array.from({ length: count }, (_, index) => () => post("/v1/responses", {
    model: GPT, max_output_tokens: 2048, reasoning: { effort: "high" }, include: ["reasoning.encrypted_content"], store: false,
    input: `一个三位数各位数字之和是 ${14 + index}，百位比个位大 2，十位是个位的 2 倍。逐步推理求出它。`,
  })));
  const items = replies.flatMap((reply) => ((reply.output as Json[]) ?? [])
    .flatMap((item) => (item.type === "reasoning" && typeof item.encrypted_content === "string" ? [{ type: "reasoning", summary: [], encrypted_content: item.encrypted_content }] : [])));
  if (items.length === 0) throw new Error(`${GPT} 没有返回推理内容，无法构造 Responses 签名用例`);
  return items;
}

const readTool = { name: "read", description: "Read a file", input_schema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } };
const toolUse = (id: string) => ({ type: "tool_use", id, name: "read", input: { path: "a.txt" } });
const toolResult = (id: string, content: unknown = "文件内容：hello") => ({ type: "tool_result", tool_use_id: id, content });
const blank = [{ type: "text", text: " " }];
const pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

function messagesBody(messages: unknown[], extra: Json = {}): Json {
  return { model: CLAUDE, max_tokens: 16, tools: [readTool], messages, ...extra };
}

/** 一个助手回合的块排列 → Anthropic 请求；签名轮流取用。 */
function sequenceBody(kinds: BlockKind[], signatures: string[]): Json {
  let used = 0;
  const content = kinds.map((kind, index) => kind === "S"
    ? { type: "thinking", thinking: "", signature: signatures[used++ % signatures.length] }
    : kind === "X" ? { type: "text", text: `正文${index}` } : toolUse(`toolu_${index}`));
  const ids = content.flatMap((block) => (block.type === "tool_use" ? [(block as Json).id] : []));
  return messagesBody([
    { role: "user", content: "读一下文件" },
    { role: "assistant", content },
    ids.length ? { role: "user", content: ids.map((id) => toolResult(id as string)) } : { role: "user", content: "继续" },
  ], { thinking: { type: "adaptive" }, output_config: { effort: "low" } });
}

/** 按 relay 生成的 Kiro 线上形状去重，每种形状只发一次。 */
function distinctSequences(): BlockKind[][] {
  const byShape = new Map<string, BlockKind[]>();
  for (const kinds of blockSequences(4, ["S", "X", "U"])) {
    const state = buildKiroPayload("claude-opus-5", anthropicToOpenAI(sequenceBody(kinds, ["sig"])), true, {}).conversationState;
    const shape = wireShape(state as unknown as KiroState);
    if (!byShape.has(shape)) byShape.set(shape, kinds);
  }
  return [...byShape.values()];
}

function edgeCases(signatures: string[]): Array<[string, string, Json]> {
  const chatTools = [{ type: "function", function: { name: "read", parameters: readTool.input_schema } }];
  const toolCalls = (ids: string[], content: string | null = null) => ({
    role: "assistant", content, tool_calls: ids.map((id) => ({ id, type: "function", function: { name: "read", arguments: "{\"path\":\"a.txt\"}" } })),
  });
  const chat = (messages: unknown[]): Json => ({ model: CLAUDE, max_tokens: 16, tools: chatTools, messages });
  const loop = [{ role: "user", content: "任务" }, ...[0, 1, 2].flatMap((round) => [
    { role: "assistant", content: [
      { type: "thinking", thinking: "", signature: signatures[(2 * round) % signatures.length] },
      toolUse(`toolu_${round}a`), toolUse(`toolu_${round}b`),
      { type: "thinking", thinking: "", signature: signatures[(2 * round + 1) % signatures.length] },
    ] },
    { role: "user", content: [toolResult(`toolu_${round}b`), toolResult(`toolu_${round}a`)] },
  ]), { role: "user", content: "总结" }];
  return [
    ["以助手开头", "/v1/messages", messagesBody([{ role: "assistant", content: "我先说" }, { role: "user", content: "好" }])],
    ["只有助手消息", "/v1/messages", messagesBody([{ role: "assistant", content: "我先说" }])],
    ["以助手结尾", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "assistant", content: "做完了" }])],
    ["连续用户消息", "/v1/messages", messagesBody([{ role: "user", content: "一" }, { role: "user", content: "二" }, { role: "assistant", content: "收到" }, { role: "user", content: "三" }, { role: "user", content: "四" }])],
    ["连续助手纯文本", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "assistant", content: "甲" }, { role: "assistant", content: "乙" }, { role: "user", content: "继续" }])],
    ["空白用户开头", "/v1/messages", messagesBody([{ role: "user", content: blank }, { role: "assistant", content: "一" }, { role: "user", content: "继续" }])],
    ["中间空白用户", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "assistant", content: "一" }, { role: "user", content: blank }, { role: "assistant", content: "二" }, { role: "user", content: "继续" }])],
    ["当前空白用户", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "assistant", content: "一" }, { role: "user", content: blank }])],
    ["孤儿工具结果（当前回合）", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "user", content: [toolResult("toolu_ghost", "孤儿")] }])],
    ["孤儿工具结果（历史）", "/v1/messages", messagesBody([{ role: "user", content: [{ type: "text", text: "任务" }, toolResult("toolu_ghost", "旧")] }, { role: "assistant", content: "好" }, { role: "user", content: "继续" }])],
    ["并行工具结果乱序", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "assistant", content: [toolUse("toolu_a"), toolUse("toolu_b"), toolUse("toolu_c")] }, { role: "user", content: [toolResult("toolu_c"), toolResult("toolu_a"), toolResult("toolu_b")] }])],
    ["空工具结果", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "assistant", content: [toolUse("toolu_a"), toolUse("toolu_b")] }, { role: "user", content: [toolResult("toolu_a", ""), toolResult("toolu_b", [])] }])],
    ["工具结果带图片", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "assistant", content: [toolUse("toolu_s")] }, { role: "user", content: [toolResult("toolu_s", [{ type: "text", text: "截图" }, { type: "image", source: { type: "base64", media_type: "image/png", data: pixel } }])] }])],
    ["空助手与只有思考的助手", "/v1/messages", messagesBody([{ role: "user", content: "任务" }, { role: "assistant", content: "" }, { role: "user", content: "再来" }, { role: "assistant", content: [{ type: "thinking", thinking: "", signature: signatures[0] }] }, { role: "user", content: "继续" }])],
    ["多轮签名工具循环", "/v1/messages", messagesBody(loop, { thinking: { type: "adaptive" }, output_config: { effort: "low" } })],
    ["Chat：每个结果一条 tool 消息", "/v1/chat/completions", chat([{ role: "user", content: "任务" }, toolCalls(["a", "b"]), { role: "tool", tool_call_id: "b", content: "乙" }, { role: "tool", tool_call_id: "a", content: "甲" }])],
    ["Chat：结果之间夹助手文本", "/v1/chat/completions", chat([{ role: "user", content: "任务" }, toolCalls(["a", "b"]), { role: "tool", tool_call_id: "a", content: "甲" }, { role: "assistant", content: "插一句" }, { role: "tool", tool_call_id: "b", content: "乙" }, { role: "user", content: "继续" }])],
    ["Chat：正文加工具调用", "/v1/chat/completions", chat([{ role: "user", content: "任务" }, toolCalls(["a"], "先读一下"), { role: "tool", tool_call_id: "a", content: "甲" }])],
  ];
}

function responsesCases(reasoning: Json[]): Array<[string, string, Json]> {
  const tools = [{ type: "function", name: "read", parameters: readTool.input_schema }];
  const call = (id: string) => ({ type: "function_call", call_id: id, name: "read", arguments: "{\"path\":\"a.txt\"}" });
  const output = (id: string) => ({ type: "function_call_output", call_id: id, output: "文件内容：hello" });
  const at = (index: number) => reasoning[index % reasoning.length];
  const body = (input: unknown[]): Json => ({ model: GPT, max_output_tokens: 16, store: false, tools, input });
  return [
    ["Responses：推理-调用-推理", "/v1/responses", body([{ role: "user", content: "任务" }, at(0), call("a"), at(1), output("a")])],
    ["Responses：两段推理后调用", "/v1/responses", body([{ role: "user", content: "任务" }, at(0), at(1), call("a"), output("a")])],
    ["Responses：推理-正文-调用-推理-调用", "/v1/responses", body([{ role: "user", content: "任务" }, at(0), { role: "assistant", content: "先读" }, call("a"), at(1), call("b"), output("b"), output("a")])],
    ["Responses：并行调用结果乱序", "/v1/responses", body([{ role: "user", content: "任务" }, call("a"), call("b"), output("b"), output("a"), { role: "user", content: "继续" }])],
    ["Responses：以助手开头", "/v1/responses", body([{ role: "assistant", content: "我先说" }, { role: "user", content: "好" }])],
    ["Responses：空白用户", "/v1/responses", body([{ role: "user", content: " " }, { role: "assistant", content: "一" }, { role: "user", content: " " }])],
  ];
}

console.log(`relay: ${BASE}  models: ${CLAUDE} / ${GPT}`);
const [signatures, reasoning] = await Promise.all([claudeSignatures(4), gptReasoning(2)]);
console.log(`取得 ${signatures.length} 个 Claude 签名、${reasoning.length} 段 GPT 推理`);

const cases: Array<[string, string, Json]> = [
  ...distinctSequences().map((kinds): [string, string, Json] => [`块排列 ${kinds.join("")}`, "/v1/messages", sequenceBody(kinds, signatures)]),
  ...edgeCases(signatures),
  ...responsesCases(reasoning),
];
const started = Date.now();
const results = await pool(cases.map(([, path, body]) => () => send(path, body)));
const failures = cases.flatMap(([name], index) => (results[index] === "200" ? [] : [`${name}: ${results[index]}`]));
console.log(`${cases.length - failures.length}/${cases.length} 通过，用时 ${Math.round((Date.now() - started) / 1000)}s`);
for (const failure of failures) console.log(`FAIL ${failure}`);
process.exitCode = failures.length ? 1 : 0;
