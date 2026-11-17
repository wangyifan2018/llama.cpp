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

## 11. 词级时间戳与标点层（已实现）

### 11.1 引擎接口

新增 `paraformer_transcribe_tokens()`：返回与原接口相同的文本，外加每个 token 的
`{text, t_ms, posterior}`（CIF fire frame 时间）。中文 token 是字，英文是 BPE 片段
（以 `@@` 结尾表示同词续接）。CLI 的 `--timestamps` 现在输出 `n_tokens`、`ts_cs`、`tok`
三列，便于离线分析。

### 11.2 API 层

- `response_format=verbose_json` 现在返回 `words[{word,start,end}]` 与按停顿切分的
  `segments[{id,start,end,text}]`；`srt` 也按停顿分块（多行字幕）。
- `punctuate=true`（默认 false，保持 OpenAI 兼容的原文）按停顿插分隔符：
  词间静音 ≥ `--punct-gap-ms`（默认 600 ms）插 `，`/`,`，结尾补 `。`/`.`。
- `--seg-gap-ms` 控制 segments 的切分阈值（默认 600 ms）。

### 11.3 为什么用停顿而不是关键词

上一轮试过"在起始词前插逗号"，命令字段从 56.2% 掉到 46.9%（`把左前车窗打开一半` 被切开）。
这次改用 ASR 自己的 CIF 时间：只在**词间静音超过阈值**处插分隔符。同一批 32 条、同一权重：

| 口径 | command | device | state | value | zh CER | en WER |
|---|---:|---:|---:|---:|---:|---:|
| raw（无偏置无标点） | 56.2% | 69.0% | 64.3% | 77.8% | 0.60% | 12.21% |
| 关键词插逗号（§18.1，已否） | 46.9% | 69.0% | 59.5% | 50.0% | 0.60% | 12.21% |
| 热词 + span rewrite，无标点 | 71.9% | 78.6% | 76.2% | 94.4% | 0.60% | 6.87% |
| 热词 + span rewrite + 停顿标点 | **78.1%** | **85.7%** | **83.3%** | 94.4% | 0.60% | 8.40% |

阈值扫描（无热词的 raw 口径）：250/300 ms -> 12.5%，350 -> 18.8%，400 -> 28.1%，
500 -> 59.4%，550-700 -> 65.6%，800 -> 59.4%。取 600 ms 为默认；太小会把单句切碎，
太大等于不切。

这条路径把设计文档 §14 里那个"仅存在于未提交脚本的 78.1%"变成了**可运行、可复现**的接口
（`curl -F punctuate=true` 或 CLI `--timestamps` + `tools/pause_punct.py`）。

### 11.4 实测（设备 SM6650）

```text
verbose_json: text=别关氛围灯把顶灯关掉  words=10
  segments: (0.00-0.84 别关氛围灯把) (1.56-2.28 顶灯关掉)
punctuate=true: 把主驾空调调到二十一度。
srt:
  1  00:00:00,000 --> 00:00:00,900  别关氛围灯把
  2  00:00:01,620 --> 00:00:02,280  顶灯关掉
```

### 11.5 已知偏差

- **分隔符可能晚一个词**：`别关氛围灯，把顶灯关掉` 实际被切成 `别关氛围灯把，顶灯关掉`。
  原因是 `把` 与后字连读，它的 CIF fire 时间落到了停顿里。试过"把/给/在… 这类单字词前移一位"
  的修正，32 条命令字段**没有变化**（75.0% 对 75.0%），所以没有并入默认逻辑；要根治得改
  时间戳回归或做真正的标点模型。
- 时间戳是 fire frame 时间（token 结束时刻），最后一字通常比音频结尾早 200-400 ms。
- 英文标点只有逗号与句号，没有问号/感叹号（模型无语调信息）。

## 31. M21：realtime partial 的背压与上限（2026-11-18）

**问题**：partial 是"整段重解码"，设备上每次约 330 ms。4 路并发、每段 2.3 s 时，仅 partial 就会
占掉相当于 4 次完整推理的引擎时间；更糟的是**commit 到达时可能正好有一个 partial 在飞**，
commit 抢队列也没用——它得等这一次解码结束。

**两条规则**（`api_server.cpp`）：
1. **队列非空就不排 partial**：只要有任何请求在等（commit 或其他 append），partial 直接跳过；
2. **`--partial-max-s`（默认 2.0）**：音频超过该长度后不再生成 partial，避免"提交前刚好排一个
   全段解码"。

**实测**（宿主 CPU，2 引擎，4 会话齐发、每段 2.32 s、3 轮，`tools/bench_ws_realtime.py`）：

| 配置 | p50 | p95 | max | partial 次数 | 跳过次数 |
|---|---:|---:|---:|---:|---:|
| 无上限（原行为 + 队列规则） | 275 | **386** | 386 | 24 | 101 |
| 上限 2.0 s | **225** | **326** | 327 | 24 | 307 |
| 上限 0.9 s | 225 | 327 | 327 | **11** | 903 |

**结论**：上限把 4 路并发的 p95 从 386 ms 降到 326 ms（−16%），单段 commit 的耗时不变
（单会话 164 ms / infer 139 ms）。0.9 s 上限只再减少 partial 次数（容量更好），尾延迟与 2.0 s
相同；默认取 2.0 s 保留一部分预览文本，容量优先时用 0.9 s。

`/metrics` 新增 `partials` 与 `partials_skipped`，可以直接观察这条背压是否在生效。

**真机（SM6650，HTP+CPU 两引擎，4 会话齐发、每段 2.32 s、2 轮）**：上限 2.0 s 时
p50 **528 ms**、p95 **700 ms**、max 700 ms，partial 实际执行 16 次、跳过 293 次（`/metrics`）。
对照的 0.9 s 与 `--no-partials` 两臂因 adb forward 在后台任务里间歇性挂起未跑完，缺这两个点；
宿主上的相对结论（p95 386→326 ms）可作参考。

**边界**：宿主为 CPU 引擎，真机为 HTP+CPU；两端的 partial 单价不同（真机约 330 ms），
所以绝对值不可跨端比较，只看同端内的相对变化。

## 33. M23：短句的静态尺寸余量 与 服务端 VAD 评估（2026-11-18）

### 33.1 短句（座舱常态）的每请求成本

设备实测，1.2 s 音频（T_lfr≈19，warm）：

| 阶段 | 耗时 | 占比 |
|---|---:|---:|
| feat（fbank+LFR+CMVN） | 2.6 ms | 1.8% |
| encoder compute | **113.7 ms** | **78.4%** |
| CIF（host） | 3.2 ms | 2.2% |
| decoder compute | 24.3 ms | 16.8% |
| logits 回读 | 0.1 ms | 0.1% |
| **合计** | **145 ms** | |
| 冷启动（同一条，首次） | 160 ms | +15 ms 建图 |

结论：**静态尺寸本身没有浪费**（T 是精确的，没有 padding），可优化的是**建图**：

- encoder 按精确 T 缓存，未命中要 ~15 ms；若做 2 帧一档的预建桶，padding 最多 1 帧
  ≈ 6 ms（6.0 ms/帧），净省 ~9 ms（6%）；
- decoder 按 (T,N) 缓存，短句 N 只有 6~12 且变化大，未命中要 ~10 ms；N 分桶能省 ~7%。

两项合计约 **17% 的短句尾延迟**（145 → ~120 ms），但都要求"padding + mask"，会改尾字数值
（FSMN 居中卷积），必须逐条对拍 32 条夹具才能上线。建议顺序：先 decoder 的 N 分桶
（占比更大、只影响 decoder），再考虑 encoder 的 T 分桶。

### 33.2 服务端 VAD：建议加，用 silero

**可行性（已实测）**：设备上已有 `libsherpa-onnx-c-api.so`，它自带 VAD 封装
（`SherpaOnnxVoiceActivityDetector`）；模型 `silero_vad.onnx`（2.2 MB，
sha256 `2623a2953f6ff3d2…`）可从 ModelScope 的 `pengzhendong/silero-vad` 直接下载。
**不需要任何新依赖**。

**成本**：设备 CPU 上 0.64–0.74% 实时（2.3 s 音频 15 ms、10 s 音频 74 ms），与 HTP 上的 ASR 并行，
不占尾延迟。

**分段实测**（`tools/vad_segments.cpp`）：

| 输入 | 检测结果 |
|---|---|
| 2.32 s 单句 | `[0.06–2.30]`（去掉 60 ms 首部静音） |
| 4.97 s 中英拼接（zh 2.32 + 0.4 s 静音 + en） | `[0.06–2.39]`、`[2.72–4.96]`（与真实边界一致） |
| 10 s 长句（中间有停顿） | `[0.06–2.68]`、`[2.94–9.98]` |

**建议**：加，但定位要清楚——**不是替代客户端 VAD**，而是覆盖客户端覆盖不了的情况：

1. **连续流客户端**（测试台、没有 VAD 的第三方）：服务端自己断句，逐段返回 final；
2. **超长 commit**（>18 s，超出模型训练长度）：先分段再识别，避免整段退化；
3. **首尾静音裁剪**：每段省 50–100 ms 的编码计算，也让说话人 embedding 更干净。

不加的理由也成立：我们的产品前端已有 VAD、每次 commit 就是一段，服务端 VAD 在主路径上收益
接近 0。所以设计成**可选**（`--vad-model` + `--vad-split-s`，默认关）。

下一批实现：REST/commit 路径的裁剪与分段（>18 s 才切），以及 WS 的 `vad: true` 连续流模式
（append 持续喂、服务端检测到端点就发一个 completed，不需要客户端 commit）。
