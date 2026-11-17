# ASR server API 设计：OpenAI 兼容层与通用抽象

日期：2026-11-17
状态：设计 + 可运行原型（`examples/paraformer/api_server.cpp`）

## 0. 结论摘要

1. **对外用 OpenAI 兼容的 `POST /v1/audio/transcriptions`**，因为智能座舱前端已经在 VAD 端点
   切好整段，一次请求一段音频正好对应 Paraformer 的整句推理；这条路径不改引擎就能落地。
2. **对外再提供 `/asr/segment` 原始 float32 端点**给设备端客户端，省掉 WAV 封装与解析。
3. **流式（边说边出字）走 OpenAI Realtime 的 WebSocket 语义**，与现有二进制帧
   （OPEN/APPEND/COMMIT/CANCEL）一一对应；本批不实现 WS，只定义映射与验收点。
4. 内部保留原生二进制帧协议，它比 HTTP 少一层解析，适合设备内进程间调用与压测。
5. 通用 ASR server 的关键不在端点名字，而在**能力协商 + 统一的流式语义 + 可解释的排队指标**；
   端点只是外壳。
6. 现有引擎缺三件东西才够“通用”：词/字级时间戳的对外暴露（引擎已有 char 时间戳）、
   partial 结果、以及每请求的语言判定。

## 1. 目标与约束

- 前端已有 VAD，服务端**只返回最终文本**；产品指标是 VAD 端点到最终文本的 P95/P99。
- 最多 4 路并发，开口与结束时间可能错开；单机（SM6650）部署，也可在服务器上跑同一套接口。
- 需要同时支持：整段 POST（VAD 切好的段）、设备端低开销调用、以及将来可能需要的流式。
- 现有引擎能力：整句非自回归推理、char 级时间戳、中文/英文单模型、无标点、无说话人分离。

## 2. 三种接口面

| 面 | 端点 | 用途 | 状态 |
|---|---|---|---|
| A. OpenAI 兼容 HTTP | `POST /v1/audio/transcriptions`、`GET /v1/models`、`GET /health`、`GET /metrics` | 对外标准、评测、第三方客户端直接复用 | **已实现** |
| B. 设备端点 | `POST /asr/segment`（float32 PCM body） | 设备内/座舱前端最低开销调用 | **已实现** |
| C. 流式 WS | `ws /v1/realtime` | 边说边出字、可打断 | **已实现** |
| D. 原生帧 | TCP `u32 op, u32 session, u32 bytes` | 内部压测、IPC | 已有（`server.cpp`） |

### 2.1 流式语义映射

| OpenAI Realtime 事件 | 原生帧 | 说明 |
|---|---|---|
| `session.update` | （启动参数） | 语言、模型、是否要时间戳 |
| `input_audio_buffer.append` | `APPEND` | 20 ms PCM，可乱序到达多路 |
| `input_audio_buffer.commit` | `COMMIT` | VAD 端点，触发最后的尾部与解码 |
| `input_audio_buffer.clear` | `CANCEL` | 丢弃当前段 |
| `conversation.item.input_audio_transcription.completed` | `FINAL` 响应 | 最终文本 |
| `input_audio_buffer.speech_started/stopped` | 前端 VAD 自己产生 | 服务端不回送 |

关键点：**OpenAI 的 commit 就是我们的 VAD 端点**，语义天然对齐；这也是为什么不需要把
VAD 搬到服务端。

## 3. OpenAI 兼容层细节

### 3.1 请求（multipart/form-data）

| 字段 | 必填 | 我们的行为 |
|---|---|---|
| `file` | 是 | 16 kHz 单声道 16-bit PCM WAV；其它格式直接 400（不做隐式重采样，避免掩盖客户端 bug） |
| `model` | 否 | 回显在 `/v1/models`；服务端固定加载一个模型，未匹配时不报错（与 OpenAI 不同，文档写明） |
| `language` | 否 | 仅回显到 `verbose_json`；引擎不做 LID |
| `prompt` | 否 | 接受但忽略（引擎无 prompt 条件），文档写明 |
| `response_format` | 否 | `json`（默认）、`text`、`verbose_json`、`srt`；其它 400 |
| `temperature` | 否 | 接受但忽略（NAR argmax 无采样） |
| `stream` | 否 | 本批返回 400；Realtime 走 WS |
| `timestamp_granularities[]` | 否 | 本批忽略 |

### 3.2 响应

`json`：`{"text": "..."}`
`text`：纯文本 body
`verbose_json`：`task`、`language`、`duration`、`text`、`segments[{id,start,end,text}]`
（当前整段一个 segment；字级时间戳引擎已有，见 §6）
`srt`：单条字幕

### 3.3 错误（OpenAI 形状）

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "file", "code": null}}
```

| 状态 | 场景 |
|---|---|
| 400 | 缺 `file`、非 WAV、采样率/声道/位深不符、`response_format` 不支持 |
| 429 | 队列满（`--queue`），带 `Retry-After` 语义（响应体提示下次重试） |
| 500 | 推理失败 |
| 503 | 排队超时（`--timeout-ms`） |

实测：`--queue 1` 单引擎四路并发，2 个 200、2 个 429，符合预期。

## 4. 参考 vLLM 的几点

1. **请求模型 + 统一错误对象**：vLLM 用 Pydantic 校验并把异常翻译成 OpenAI 错误结构；我们用
   `form` 字段显式校验，错误体同形状。
2. **引擎池与排队**：vLLM 用 `max_num_seqs` 控制并发、超限排队；我们对应 `--engines` +
   `--queue` + 429。
3. **请求取消**：vLLM 在客户端断开时释放槽位；我们在 worker 动手前用 `MSG_PEEK` 检查连接
   （`server.cpp` 已实现）。HTTP 侧本批未做，因为 httplib 的 handler 是同步的；要做得在
   handler 里轮询 `req` 的 socket 状态，或改成异步。
4. **`/v1/models` 与 `/health`**：与 vLLM 一致，便于编排系统探活。
5. **SSE**：vLLM 的转写接口支持 `stream=true` 发 SSE；我们留到 Realtime 实现时一起做，
   避免两种流式语义并存。

## 5. 通用 ASR server 抽象

端点只是外壳，真正通用的是下面四件事：

1. **能力协商**：`GET /v1/models` 里为每个模型返回 `capabilities`：
   `{streaming, timestamps: "char"|"word"|"segment", languages: ["zh","en"], punctuation, diarization}`。
   客户端据此决定用 A 还是 C 面，不用猜。
2. **统一流式语义**：`open/append/commit/cancel/close`，与 OpenAI Realtime 对齐；批量端点等价于
   “一次 open+append+commit”。
3. **可解释的排队**：每个响应带 `queue_ms` 与 `infer_ms`（写在 header，不污染 OpenAI body），
   这样 P95 可以分解成“排队”和“推理”，这正是我们前几轮调优时最缺的。
4. **多模型路由**：同一进程按 `model` 字段路由到不同引擎池（zh 大模型 / en 模型 / 小模型），
   路由键可以是显式 `model`，也可以是 VAD 段级 LID 结果。

## 6. 与现有引擎的差距（要补什么）

| 缺口 | 现状 | 补法 |
|---|---|---|
| 字级时间戳未暴露 | `paraformer_transcribe_with_timestamps()` 已有 char 时间戳 | `verbose_json` 的 `words`/`segments` 用它切分，中文字符直接给 start/end |
| 无 partial | 只有整句 final | Realtime 的 `...transcription.delta` 需要滑窗解码（设计文档 §5 B 档） |
| 无 LID | 中英单模型 | 段级分类或上游标签；`language` 从回显变成实测 |
| 无标点 | 引擎不输出 | 语义层或标点模型；接口上 `punctuation: false` 先声明 |
| 无取消（HTTP） | 原生帧有 | 异步 handler 或轮询 socket |

## 7. 兼容性测试清单

```sh
curl -sS localhost:8899/health
curl -sS localhost:8899/v1/models
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@a.wav
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@a.wav -F response_format=text
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@a.wav -F response_format=verbose_json
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@a.wav -F response_format=srt
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@notwav.bin          # 400
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F model=x                   # 400
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@a.wav -F response_format=weird  # 400
# 队列打满：--queue 1 --engines 1，四路并发 -> 2x200 + 2x429
```

## 8. 非目标

- 不做说话人分离、不做标点模型、不做热词上传（热词仍是启动配置）。
- 不做 OpenAI 的 `/v1/audio/translations`（等有 en 专用模型再说）。
- 不把 VAD 搬进服务端：前端已有，服务端只消费 commit。

## 9. 实测：REST（SM6650，`--device HTP0:0 --engines 2`）

单段 2.7 s 中文命令返回 `{"text":"把主驾空调调到二十一度"}`，四路并发全部 200（0.31–0.67 s）
且转写逐条一致；`--queue 1` 时 2x200 + 2x429；设备侧经 `adb forward` 与宿主行为一致。
每个响应带 `X-Queue-Ms` / `X-Infer-Ms`（设备实测 0 / 299 ms）。

## 10. 实测：Realtime WebSocket（已实现）

客户端：`examples/paraformer/tools/ws_client.py`（20 ms 分包或一次性发送，收到 delta 打印）。

**partial 策略**：Paraformer 是整句非自回归，partial 只能"把当前已收音频整体再解码一次"，所以：

- 触发条件：新增音频 ≥ `--partial-ms`（默认 800 ms）且距上次 partial ≥
  `--partial-interval-ms`（默认 600 ms），且该会话没有在飞的 partial；
- partial 任务以**低优先级**入队，**commit 任务插队**，保证端点不被 partial 拖慢；
- 结果用"与上次 partial 的最长公共前缀"求增量，只发改动部分（OpenAI 的 delta 语义）；
- commit/clear 会让 `generation` 自增，在飞的 partial 回来时直接丢弃（不会把旧文本贴到新段上）。

**实测（SM6650，HTP+CPU 两引擎）**：

```text
event: session.created
event: session.updated
delta: '把主驾'
delta: '空调垫'
delta: '调到二十'
event: input_audio_buffer.committed
completed: 把主驾空调调到二十一度
endpoint_to_final_ms: 368
server: queue_ms=1 infer_ms=355
```

宿主 CPU 上同一段是 `endpoint_to_final 165 ms`。设备上每次 partial 都要跑一次完整推理
（约 330 ms），所以 partial 的间隔实际上是"每 600 ms 语音换一次增量"；如果产品要更细的增量，
只能走滑窗解码（设计文档 §5 的 B 档），不是靠调小间隔。

**打断/清空**：`input_audio_buffer.clear` 立刻清空缓冲并让在飞任务失效，随后可以立即开始新段。

**已知偏差（写进文档以免误用）**：

1. 只接受 16 kHz PCM16；24 kHz 会做线性重采样（质量说明见实现注释），产品路径请直接用
   `/asr/segment` 或 16 kHz。
2. 没有服务端 VAD：`speech_started/stopped` 事件不发，端点仍由客户端 commit 决定。
3. 没有 `conversation.item.*` 的完整会话树，只发转写相关的四个事件 + `x.pf.metrics`
   （非标准，便于观测 queue/infer）。
