# Kiro Relay

一个用 Bun + TypeScript 写的精简 Kiro 反向代理。它把 Kiro（Amazon Q Developer）账号提供的模型，按原生协议暴露成三套标准接口：

- Anthropic Messages：`POST /v1/messages`、`POST /v1/messages/count_tokens`
- OpenAI Chat Completions：`POST /v1/chat/completions`
- OpenAI Responses：`POST /v1/responses`（仅 GPT-5.6 系列）

每个入口都直接翻译到 Kiro 的 `conversationState`，不会在协议之间二次转发。

## 设计取向

- **协议保真**：保持增量流式节奏与并行工具调用；Anthropic 与 Responses 保留思考签名和隐藏推理，Chat 只提供 `reasoning_content`。不注入思考提示词，也不伪造推理内容。
- **明确能力边界**：`stop` / `stop_sequences`、JSON `response_format`、Responses 的服务端存储与 `previous_response_id` 等不支持的功能返回 400 并说明原因；图片占位、工具约束的尽力兼容行为见“已知限制”。
- **保持精简**：没有数据库、管理后台、WebUI，也不做多供应商路由。只有一个 YAML 配置和一个凭据文件。
- **日志脱敏**：只记录请求 ID、模型、状态码、阶段、固定错误码及经过校验的异常类型，不记录密钥、提示词、思考内容、签名或异常消息原文。流式失败以 `stream_failed` 阶段单独记录。

## 快速开始

需要 [Bun](https://bun.sh) 1.x。

```sh
bun install

# 1. 登录 Kiro，凭据写到 ~/.config/kiro-relay/account.json（权限 0600）
bun run auth:login
# 企业 IAM Identity Center 账号：
bun run auth:login -- --start-url https://<your-org>.awsapps.com/start --region us-east-1

# 2. 准备配置
cp config.example.yaml config.yaml

# 3. 启动
bun run start
```

登录走 AWS SSO-OIDC 设备码流程：命令会打印一个链接和设备码，在任意设备的浏览器里打开确认即可，所以也适合在无图形界面的服务器上使用。

登录参数：

| 参数 | 说明 |
|---|---|
| `--output <path>` | 凭据文件路径，默认 `~/.config/kiro-relay/account.json` |
| `--start-url <url>` | 默认是 AWS Builder ID；企业账号填组织的 start URL |
| `--region <region>` | 默认 `us-east-1` |
| `--client-id` / `--client-secret` | 复用已注册的 OIDC 客户端 |
| `--client-registration <file>` | 从含 `clientId`、`clientSecret` 的 JSON 文件读取客户端信息 |

凭据中的 token 快过期时会自动刷新，并原子写回凭据文件；多个请求同时触发刷新时只会真正刷新一次。每次刷新最多等待 15 秒（含响应体读取），临时网络故障、限流或上游 5xx 返回 503；主动刷新遇到临时故障且旧 token 仍有效时继续使用旧 token。凭据失效或配置无效返回 401，需重新运行 `auth:login`。登录取得 token 后立即落盘，再尽力发现并补存 profile ARN。凭据写入使用跨进程文件锁；补存 profile 时若 token 已轮换，保留当前凭据并跳过补存。凭据所在目录必须可写，以便创建临时文件和锁目录。

## 配置

配置文件路径默认是当前目录下的 `config.yaml`，可以用环境变量 `KIRO_RELAY_CONFIG` 指定。

```yaml
listen: 127.0.0.1:4000

# 开启后，模型列表来自 Kiro 的 ListAvailableModels，即账号实际可用的模型。
discovery:
  enabled: true

# 发现失败时的兜底列表；也可以用来修正上游报错的上下文窗口。
models:
  claude-opus-5:
    contextLength: 1000000
  gpt-5.6-sol:
    contextLength: 272000

# 访问密钥，客户端用 `Authorization: Bearer <key>` 或 `x-api-key` 发送。
# 不配置则不做鉴权，只适合监听在回环地址时使用。
auth:
  keys:
    - replace-with-a-long-random-string

accounts:
  - credentialFile: ~/.config/kiro-relay/account.json
```

> [!WARNING]
> 监听到非回环地址或对外提供服务前，务必配置 `auth.keys`，否则任何能访问端口的人都能使用你的 Kiro 账号。

## 客户端示例

Anthropic SDK / Claude Code 一类客户端：

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_API_KEY=<auth.keys 中的一项>
```

OpenAI SDK：

```ts
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://127.0.0.1:4000/v1", apiKey: process.env.KIRO_RELAY_KEY });
const reply = await client.chat.completions.create({
  model: "claude-sonnet-5",
  messages: [{ role: "user", content: "Hello" }],
});
```

`GET /v1/models` 返回当前可用模型及其 `context_length`、`input_modalities`（`["text","image"]` 或 `["text"]`），支持 `ETag` / `If-None-Match`。图片能力优先采用 Kiro 目录的 `supportedInputTypes`；离线时仅 deepseek-3.2、minimax-m2.1、minimax-m2.5、glm-5、qwen3-coder-next 禁用图片，其余模型默认支持。浏览器预检允许 `If-None-Match`，客户端可读取 `ETag` 和 `Retry-After` 响应头。`GET /healthz` 用于健康检查，不需要鉴权。

`GET /v1/usage` 查询 Kiro 账号本期额度（与 `/v1` 同一套鉴权，不消耗额度），数据来自 Kiro 的 `GetUsageLimits`：

```sh
curl -H "Authorization: Bearer <key>" https://relay.example/v1/usage
# {"subscription":"KIRO POWER","overage_enabled":false,"resets_at":"2026-12-01T00:00:00.000Z",
#  "buckets":[{"resource":"CREDIT","unit":"INVOCATIONS","used":4939.23,"limit":10000,"remaining":5060.77,"percent":49.39,...}]}
```

## 已知限制

- 图片只发送 base64 内联，单张 base64 不超过 5 MB（含工具结果内嵌图片）；超限仍报错，客户端可以据此自动压缩后重试。远程 URL 在原位置替换为 `[image omitted: remote image URLs are not supported, send base64 inline]`，不支持图片的模型收到内联图时替换为 `[image omitted: this model does not support image input]`；两种情况均不发送图片字节，也不拒绝请求。`document` 内容块仍不支持。
- Responses 接口仅支持 GPT-5.6 系列这类原生推理模型；`store` 必须为 `false`，不支持 `previous_response_id`、`conversation`、`background`、`prompt`、`truncation`。
- Responses 的 `text.format.type` 仅接受 `text`；Anthropic 不支持 `top_k`，存在即返回 400。
- Responses 的 `reasoning.effort` 可省略或为 `null`；`summary` 独立校验，仅有 summary 时沿用上游默认推理设置，不发送推理信封。
- Chat 流式与非流式响应均输出标准停止原因，usage 的输入 token 包含缓存读取与写入，`cached_tokens` 仅为缓存读取。Chat 不暴露内部停止详情、签名、隐藏推理和估算标记。
- 工具参数按调用 ID 统一组装：字符串片段即时流出，对象快照只发布最终值；嵌套工具包装校验通过才公开，混用片段类型返回流错误。Responses 的 `response.function_call_arguments.done.name` 使用恢复后的公开工具名。
- 验证变更路径时可用 `bun run live-check --quick`，只发送 developer、required 工具选择、Responses 助手开头、glm-5 图片占位四个请求，不采集签名；完整历史形状检查沿用默认模式。`--summary-only` 可额外探测仅 summary 请求。脚本只打印用例名和安全状态，不输出响应正文或凭据。
- 当前只使用 `accounts` 中的第一个账号。
- Anthropic / Chat 的工具说明超过 Kiro 单项长度限制时，全文保留在带 `<system-reminder>` 标记的初始上下文中；后续用户消息和工具结果续接不再追加文档。相同工具集与初始输入保持该段字节稳定；实际缓存命中仍取决于上游，不能据此保证命中率。
- Chat / Anthropic 的 `system` 与 `developer` 消息统一包在 `<system-reminder>` 中。三个入口共用历史规整：补齐中性起始回合、按工具 ID 配对结果、孤儿结果降为正文、空结果填 `(no output)`；省略工具声明时由历史调用合成最小声明，Responses 保留 custom / namespace 元数据。
- 接受 `strict: true`，但 Kiro 不保证严格的参数格式，只能尽力遵循 schema。`parallel_tool_calls` 与 `disable_parallel_tool_use` 均忽略，不限制或丢弃并行调用。`tool_choice` 支持 auto、none、required / any 和各入口的指定工具形状；指定工具必须有声明（含历史合成声明）。

## 部署

```sh
bun run build          # 输出 dist/server.js
bun dist/server.js
```

用 systemd 运行时，建议加上 `ProtectSystem=strict`、`ProtectHome=read-only`，并只给凭据目录写权限（例如 `ReadWritePaths=/home/<user>/.config/kiro-relay`，systemd 不展开 `~`），让 token 刷新后能写回。

若健康检查正常但生成请求返回 403，上游可能在拒绝过期客户端：本次实测返回 `FEATURE_NOT_SUPPORTED`，消息为 “This version of Kiro is no longer supported”。生成请求的 `User-Agent` 与 `X-Amz-User-Agent` 必须包含上游识别的 `KiroIDE-<版本>` 标记；版本可核对[官方稳定版元数据](https://prod.download.desktop.kiro.dev/stable/metadata-linux-x64-stable.json)。更新 `src/kiro/config.ts` 后重新构建、重启，并以真实生成请求验证，不能只看 `/healthz`。

线上更新通过 Git 同步源码，再安装锁定依赖、构建并重启服务：

```sh
git pull --ff-only
bun install --frozen-lockfile
bun run build
sudo systemctl restart kiro-relay
```

## 开发

```sh
bun test            # 单元测试与协议黄金样例
bun run typecheck   # 类型检查（含 tests、scripts）
bun run dev         # 监听文件变化自动重启
```

`tests/kiro-history-shapes.test.ts` 会枚举助手回合里思考、正文、工具调用的各种排列，以及历史结构的边界情况，并检查生成的 Kiro 请求是否满足 `tests/kiro-history-rules.ts` 中整理的上游结构规则。

改动请求转换的大重构前后，再用真实 Kiro 跑一遍安全门。它会消耗少量额度，不在 CI 中运行：

```sh
bun run start &                                   # 用工作区代码启动本地 relay
KIRO_RELAY_KEY=<auth.keys 中的一项> bun run live-check
# 或检查已部署的实例：KIRO_RELAY_URL=https://relay.example KIRO_RELAY_KEY=... bun run live-check
```

脚本先向 relay 取得 Kiro 真签发的思考签名，再把各类历史形状逐个发出，全部返回 200 才算通过（退出码 0）。输出只有用例名和状态码。

## 免责声明

本项目与 Amazon Web Services、Anthropic、OpenAI 均无关联。使用前请确认你的用法符合 Kiro / Amazon Q Developer 的服务条款，账号与用量风险由使用者自行承担。

## 许可

[MIT](LICENSE)。Kiro 协议实现改编自 [OmniRoute](https://github.com/diegosouzapw/OmniRoute)（MIT），部分 Responses 协议测试样例改编自 [OpenAI Codex](https://github.com/openai/codex)（Apache-2.0），详见 [NOTICE](NOTICE)。
