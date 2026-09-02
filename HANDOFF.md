# Kiro Relay 实施交接

## 状态

待执行。当前仓库除本交接文档外没有实现代码。

## 目标

使用 **Bun + TypeScript** 实现一个小型、可靠的 Kiro 反向代理，供 OMP 通过 Anthropic API 调用。

目标不是重新研究或重写 Kiro 协议，而是从相邻的 OmniRoute 浅克隆中**直接复制并改造已经踩过坑的 Kiro 实现和回归测试**，移除其复杂控制面。

## 硬性要求

1. 必须从 `../OmniRoute` 复制、裁剪和适配 Kiro 核心代码；禁止从零重写 Kiro 协议、AWS EventStream、Tool Use、Thinking、Token Refresh 等已有实现。
2. 使用 Bun 和 TypeScript。
3. 保留复制代码的原始版权声明、MIT License 和来源说明。
4. 不修改 `../OmniRoute`，不修改当前运行中的 OmniRoute 容器，不修改 Kiro CLI 或其他 Kiro 官方产品。
5. 不提交任何账户凭据、Access Token、Refresh Token、Client Secret、Profile ARN 或真实请求内容。
6. 必须完成真实请求自测并与现有 OmniRoute 做差分验证；只有行为通过后才算完成。
7. 不要实现 WebUI、数据库控制面、多 Provider 路由、Compression、Combo Model 或 Usage Dashboard。

## 源码基线

```text
源码：../OmniRoute
上游分支：release/v3.8.51
基线提交：97041954171ee3aa3f51a7fb01748797c23716f8
许可证：MIT
Copyright (c) 2026 diegosouzapw
```

开始工作前确认相邻仓库仍处于该提交或更新的明确上游提交，并记录实际采用的基线。不得把相邻源码仓库当作可写工作区。

## 必须优先复制和适配的实现

Kiro 主链路：

```text
../OmniRoute/open-sse/executors/kiro.ts
../OmniRoute/open-sse/executors/kiroThinking.ts
../OmniRoute/open-sse/executors/kiroToolCallValidation.ts
../OmniRoute/open-sse/translator/request/openai-to-kiro.ts
../OmniRoute/open-sse/translator/request/openai-to-kiro/
../OmniRoute/open-sse/translator/response/kiro-to-openai.ts
../OmniRoute/open-sse/utils/kiroSanitizer.ts
../OmniRoute/open-sse/services/kiroModels.ts
../OmniRoute/open-sse/services/kiroRegion.ts
../OmniRoute/open-sse/services/kiroExternalIdp.ts
../OmniRoute/open-sse/services/tokenRefresh/providers/kiro.ts
```

按上述文件的真实 Import Graph 继续复制必要的共享叶子模块，例如 AWS EventStream、SSE 编码和纯类型/标量辅助函数。不要为了省事把 OmniRoute 整个 Router、数据库或 Dashboard 一起搬进来。

回归测试来源：

```text
../OmniRoute/tests/unit/*kiro*.test.ts
../OmniRoute/tests/unit/executor-kiro.test.ts
../OmniRoute/tests/unit/translator-openai-to-kiro.test.ts
../OmniRoute/tests/unit/translator-resp-kiro-to-openai.test.ts
```

当前约有 49 个 Kiro 命名的回归测试文件。先复制适用于协议核心的测试和 Fixtures，再改接口边界；不要删掉失败测试来获得绿色结果。

## 必须保留的既有行为

从 OmniRoute 抽取时必须保留：

- Kiro 不支持普通 System Role 时的 `system-reminder` 包装。
- JSON Schema 递归清洗。
- 超过 64 字符的 Tool 名称 Hash 及响应反向映射。
- 长 Tool Description 移入 System Prompt。
- Tool Call Wrapper、Tool Name、Tool Input 校验。
- 字符串 Tool Argument Delta 的流式转发。
- 累积对象形式 Tool Arguments 的缓存和单次发出，避免重复 JSON。
- 原生 Reasoning Event 和内联 `<thinking>` 的拆分。
- Claude 图片输入处理。
- AWS EventStream 的分片解析。
- `profileArn` Runtime Region 与 OIDC Region 分离。
- API Key、External IdP 和普通 Token 的不同 Header。
- 401、403、404 Endpoint Fallback。
- Token Refresh 和必要的客户端重新注册。
- 账户身份不能只用 `profileArn` 去重。
- Kiro `contextUsagePercentage` 到绝对 Token Usage 的正确换算。

## 已知必须修复的 Bug

OmniRoute 3.8.51 会按照 Model Registry 的 `contextLength` 将 Kiro 的 `contextUsagePercentage` 转换成 `prompt_tokens`，但当前 Registry 没有 `claude-opus-5`，查不到时回退到 200,000。

新 Relay 必须明确配置：

```text
claude-opus-5 contextLength = 1,000,000
```

换算公式：

```text
prompt_tokens = round(contextUsagePercentage / 100 × contextLength)
```

已知真实对照样本：

```text
相同 Prompt：443,983 字符，65,000 个编号词
Kiro contextUsagePercentage 反推值：26.943%
旧网关错误值：53,886 tokens（错误乘以 200k）
Xiaokld Anthropic Usage 锚点：263,214 tokens
按 1M 换算：约 269,430 tokens
```

允许不同 Provider/Tokenizer 产生小幅差异，但绝不能再返回约 53,886。OMP 侧主动限制为 500k 属于用户策略，不是本项目要修的问题。

## 对外接口

最小接口：

```text
POST /v1/messages
POST /v1/messages/count_tokens
GET  /v1/models
GET  /healthz
```

`/v1/messages` 必须兼容 OMP 使用的 Anthropic Messages API，包括：

- 流式与非流式响应。
- Text、Image、Tool Use、Tool Result 和 Thinking。
- Anthropic SSE Event 顺序和结束语义。
- 正确的 Stop Reason。
- 正确的 Input、Output 和 Cache Usage 字段；Kiro 没有真实 Cache 明细时不要伪造。

OmniRoute Kiro 核心内部主要使用 OpenAI 风格中间结构。可以复用其转换核心，但对外必须增加一个窄的 Anthropic Adapter；不要因此搬入整个 OmniRoute 通用 Router。

## 配置边界

只实现必要配置：

```yaml
listen: 127.0.0.1:4000
models:
  claude-opus-5:
    contextLength: 1000000
accounts:
  - credentialFile: ~/.config/kiro-relay/account.json
```

凭据文件必须位于仓库外。写回刷新 Token 时使用原子替换并限制文件权限。若实现多账户，只做简单、可解释的 Sticky 选择和失败冷却，不做管理后台。

## 实施顺序

1. 初始化最小 Bun/TypeScript 项目、测试命令和 License/NOTICE。
2. 根据 Import Graph 复制 Kiro 核心及相关测试，先保持行为一致。
3. 用本项目自己的窄接口替换 OmniRoute 的 BaseExecutor、Router、数据库、日志和凭据依赖。
4. 实现 Anthropic `/v1/messages` Ingress/Egress Adapter。
5. 加入 `claude-opus-5 = 1,000,000` 并修复 Usage 换算。
6. 跑移植后的 Kiro 回归测试，修复真实兼容问题，不得用跳过或删除测试掩盖失败。
7. 使用独立端口启动新 Relay，与现有 OmniRoute 并行做真实请求差分。
8. 验证完成后给出 OMP `models.yml` 的切换片段，但未经用户明确授权不要修改其 dotfiles 或停掉现有 OmniRoute。

## 验收

以下全部通过才算完成：

- Bun TypeScript 构建/类型检查通过。
- 移植的 Kiro 协议回归测试通过。
- `/healthz`、`/v1/models`、`/v1/messages/count_tokens` 可用。
- 普通非流式对话可用。
- Anthropic SSE 流式对话可用且正常结束。
- OMP Tool Use 往返成功，长名称和复杂 Schema 正常。
- Thinking 能正确显示且不污染正文。
- 图片输入可用。
- Token 过期后刷新成功，凭据写回不损坏文件。
- 长上下文实验的 Usage 接近 1M 窗口口径，不再按 200k 低报。
- 同一批代表性请求在新 Relay 与现有 OmniRoute 上做过差分，输出结构和关键行为一致。
- 真实运行日志中不泄露凭据和完整敏感 Prompt。
- 没有引入 WebUI、数据库、压缩器或多 Provider 控制面。

## 完成时应交付

- 可直接运行的 Bun/TypeScript Relay。
- 最小配置示例，不含真实凭据。
- 保留来源与许可证的 NOTICE。
- 单元测试与真实差分验证结果。
- 一条启动命令。
- 一段供 OMP 使用的 Provider 配置示例。
- 当前已知限制；不得用占位实现、静默降级或“后续再做”冒充完成。
