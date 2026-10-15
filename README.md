# Kiro Relay

一个用 Bun + TypeScript 写的精简 Kiro 反向代理。它把 Kiro（Amazon Q Developer）账号提供的模型，按原生协议暴露成三套标准接口：

- Anthropic Messages：`POST /v1/messages`、`POST /v1/messages/count_tokens`
- OpenAI Chat Completions：`POST /v1/chat/completions`
- OpenAI Responses：`POST /v1/responses`（仅 GPT-5.6 系列）

每个入口都直接翻译到 Kiro 的 `conversationState`，不会在协议之间二次转发。

## 设计取向

- **协议保真**：流式节奏、思考内容与签名、并行工具调用都按上游原样转发。不注入思考提示词，也不伪造推理内容。
- **不支持就明确拒绝**：Kiro 做不到的功能直接返回 400 并说明原因，例如 `stop` / `stop_sequences`、JSON `response_format`、Responses 的服务端存储与 `previous_response_id`。不会悄悄忽略。
- **保持精简**：没有数据库、管理后台、WebUI，也不做多供应商路由。只有一个 YAML 配置和一个凭据文件。
- **日志脱敏**：只记录请求 ID、模型、状态码、阶段、错误类别等元数据，不记录密钥、提示词、思考内容和签名。

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

凭据中的 token 快过期时会自动刷新，并原子写回凭据文件；多个请求同时触发刷新时只会真正刷新一次。

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

`GET /v1/models` 返回当前可用模型及其 `context_length`，支持 `ETag` / `If-None-Match`。`GET /healthz` 用于健康检查，不需要鉴权。

## 已知限制

- 图片只接受 base64 内联，单张 base64 不超过 5 MB；不支持远程图片 URL 和 `document` 内容块。超限时的报错措辞与 Anthropic 官方一致，客户端可以据此自动压缩后重试。
- Responses 接口仅支持 GPT-5.6 系列这类原生推理模型；`store` 必须为 `false`，不支持 `previous_response_id`、`conversation`、`background`、`prompt`、`truncation`。
- 当前只使用 `accounts` 中的第一个账号。

## 部署

```sh
bun run build          # 输出 dist/server.js
bun dist/server.js
```

用 systemd 运行时，建议加上 `ProtectSystem=strict`、`ProtectHome=read-only`，并只给凭据目录写权限（例如 `ReadWritePaths=/home/<user>/.config/kiro-relay`，systemd 不展开 `~`），让 token 刷新后能写回。

## 开发

```sh
bun test            # 单元测试与协议黄金样例
bun run typecheck   # 类型检查（含 tests）
bun run dev         # 监听文件变化自动重启
```

## 免责声明

本项目与 Amazon Web Services、Anthropic、OpenAI 均无关联。使用前请确认你的用法符合 Kiro / Amazon Q Developer 的服务条款，账号与用量风险由使用者自行承担。

## 许可

[MIT](LICENSE)。Kiro 协议实现改编自 [OmniRoute](https://github.com/diegosouzapw/OmniRoute)（MIT），部分 Responses 协议测试样例改编自 [OpenAI Codex](https://github.com/openai/codex)（Apache-2.0），详见 [NOTICE](NOTICE)。
