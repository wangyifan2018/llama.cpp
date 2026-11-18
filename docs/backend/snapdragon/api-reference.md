# Paraformer ASR 服务：调用方 API 参考

版本：2026-11-18（与本仓库 HEAD 同步）
适用：座舱前端在 VAD 端点切段后，一次一段音频请求最终文本；也可用于离线批量。

## 1. 概览

| 项 | 值 |
|---|---|
| 传输 | HTTP/1.1（REST）与 WebSocket（实时） |
| 音频 | 16 kHz、单声道、16-bit PCM |
| 端口 | 默认 8080（设备示例用 8899） |
| 鉴权 | 无（内网/设备内使用；对外请加反向代理） |
| 并发 | 由 `--engines` 决定引擎数，`--queue` 决定等待上限 |
| 模型 | 启动时固定加载一个 GGUF；`/v1/models` 返回其 id 与能力 |

## 2. 端点

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 探活、能力和预热图的后端落点。`backends` 包含各引擎请求与实际设备、`flash_attn` 和两阶段计算节点数 |
| GET | `/v1/models` | OpenAI 形状的模型列表 + `capabilities` |
| GET | `/metrics` | Prometheus 文本格式（0.0.4），`pf_` 前缀，见下 | 
| GET | `/metrics.json` | 同一份快照的 JSON，字段名与应用内部一致 |
| POST | `/v1/audio/transcriptions` | OpenAI 兼容转写（multipart/form-data） |
| POST | `/asr/segment` | 设备端低开销转写（body 直接是 float32 PCM） |
| WS | `/v1/realtime` | 实时：append / commit / clear，返回 partial 与最终结果 |
| GET | `/` | 浏览器测试台（`--web-dir` 存在时） |

### 认证与监听地址

默认不启用认证（与所有已记录测量一致）。设了 `--api-key` 或 `PF_API_KEY` 后，
**接收音频的端点**（`POST /v1/audio/transcriptions`、`POST /asr/segment`、
`WS /v1/realtime` 的升级握手）要求 `Authorization: Bearer <token>`；比较是常量时间，
避免通过时序逐字节还原 token。拒绝时返回 401，错误体与其他错误同一形状：

```json
{"error":{"code":"invalid_api_key","message":"missing or invalid bearer token","type":"authentication_error"}}
```

`/health`、`/v1/models`、`/metrics` 保持开放：容器健康检查无法持凭证，拒绝它会把
运行中的服务变成永久 unhealthy。它们只暴露引擎数与队列深度，不含音频。
`/metrics` 多一个 `auth_rejected` 计数。

`PF_API_KEY` 优先于 `--api-key`，因为命令行参数在 `/proc/<pid>/cmdline` 里对同机
所有进程可读；两者都设时以显式参数为准。

`--bind` 默认 `127.0.0.1`。默认不认证的语音服务不应因为一个没人选择的默认值出现在
所有网卡上；要放到局域网上必须显式 `--bind 0.0.0.0`，此时若未设 token，启动日志会
打印一段 WARNING。

验证用 [`tools/check_auth.py`](../../../examples/paraformer/tools/check_auth.py)，
覆盖拒绝（无 token / 错 token / 缺 Bearer scheme / 空 token / 同长度错 token）、
放行（正确 token、小写 `bearer`）、探针开放和 WS 升级两条路径；`--no-auth` 是
针对未设 token 服务的镜像检查。

### 指标

`/metrics` 返回 Prometheus 文本格式（version=0.0.4），`/metrics.json` 是同一份快照的
JSON。两者由同一个采集函数渲染：分成两个读取点的话，计数在两个请求之间会走，仪表盘
上画的数和部署门禁读的数就成了两个值。

字段（JSON 名 → 文本名）：

| JSON | Prometheus | 类型 | 含义 |
|---|---|---|---|
| `requests` | `pf_requests` | counter | 已开始的转写/分段请求 |
| `rejected` | `pf_rejected` | counter | 被准入控制拒绝 |
| `failed` | `pf_failed` | counter | 以错误响应结束 |
| `deadline_expired` | `pf_deadline_expired` | counter | 达到 `--deadline-ms` |
| `cancelled_queued` | `pf_cancelled_queued` | counter | 出队前被丢弃 |
| `auth_rejected` | `pf_auth_rejected` | counter | token 缺失或错误 |
| `partials` | `pf_partials` | counter | 实时会话里真正跑的 partial |
| `partials_skipped` | `pf_partials_skipped` | counter | 为了让 COMMIT 优先而跳过的 partial |
| `speaker_started` | `pf_speaker_started` | counter | 提交给评分线程 |
| `speaker_assessed` | `pf_speaker_assessed` | counter | 取到分数 |
| `speaker_deferred` | `pf_speaker_deferred` | counter | 未取到结果 |
| `cpu_fallback` | `pf_cpu_fallback` | gauge | 1 表示已回落到 CPU |
| `queue` | `pf_queue_depth` | gauge | 等待 worker 的任务数 |
| `queue_bytes` | `pf_queue_bytes` | gauge | 排队任务持有的音频字节 |
| `queue_bytes_limit` | `pf_queue_bytes_limit` | gauge | 上面那个的上限 |
| `inflight` | `pf_inflight` | gauge | worker 正在执行的任务数 |
| `accepted` | `pf_accepted` | gauge | 已接受且尚未答复（含排队与执行中） |
| `ws_sessions` | `pf_ws_sessions` | gauge | 活跃实时会话 |
| `engines` | `pf_engines` | gauge | 本进程的引擎数 |

两条命名约定是刻意的：**没有任何名字带 `_total`**，后缀按惯例意味着单调递增的
counter，Prometheus 抓取时还会把它当 `_total` 前身改写序列名；`queue`、`inflight`、
`accepted`、`engines` 都是可升可降的 gauge，带上后缀会让 `rate()` / `increase()`
算出垃圾。**尺寸类指标按整数输出**：`size_t` 走 `double` 会打成 `8.38861e+06`，
Prometheus 照收不误，但序列里的值已经不等于配置的上限。

校验用 [`tools/check_metrics.py`](../../../examples/paraformer/tools/check_metrics.py)：
格式合法性、`_total` 与类型一致性、整型输出、以及文本与 JSON 逐值一致。后者只在两次
JSON 读数相同时才强制逐值比较——计数器在两次请求之间会走，否则会把正常流量报成不一致。

### 优雅退出

`SIGTERM` / `SIGINT` 的语义是"把已接受的请求答完再退"，这是滚动重启时服务管理器
期待的行为；直接断连会把每次发布变成一批客户端可见的失败。`--shutdown-grace-ms`
（默认 5000）是等待上限，到期后照样关监听并在日志里说明，不会假装排空成功。

```
pf-api: shutdown requested; draining before closing the listener
pf-api: drain start, accepted=3 inflight=1, grace 5000 ms
pf-api: closing the listener
pf-api: shut down cleanly
```

三个细节都是实测踩出来的，不是设计时想到的：

- **排空等的是 `accepted` 而不是 `inflight`**。`inflight` 只在 worker 弹出任务时
  自增，请求还在队列里、或者已有答案正在序列化时它读 0。按它排空会在客户端仍在
  等待时宣布成功——正是这个功能要防的失效。
- **`srv.stop()` 必须放在排空之后**。它先撕掉活跃连接，于是排空等待的 handler
  提前结束：计数归零、日志写 "clean"、客户端只收到已缓冲的字节。曾把这个顺序写反，
  现象是 `accepted=0` 却仍有 1.66 s 的在途请求返回完整 200。
- **宽限期到期后仍在跑的请求通常还是能拿到完整响应**，因为关监听不会杀掉正在执行
  的 handler。所以超时文案只写"结果不再保证"，不去预言 reset——预言错了会把事后
  排查引向错误方向。

显式指定的加速后端不可用时，服务默认启动失败。`--allow-cpu-fallback` 允许启动时改用 CPU，此时 `/health` 返回 HTTP 200 和 `status: degraded`，`/metrics` 返回 `cpu_fallback: true`。正常启动的状态为 `ok`。

## 3. POST /v1/audio/transcriptions

`multipart/form-data`：

| 字段 | 必填 | 说明 |
|---|---|---|
| `file` | 是 | 16 kHz 单声道 16-bit PCM WAV；其他格式返回 400（不做隐式重采样） |
| `model` | 否 | 仅回显；服务端只加载一个模型 |
| `language` | 否 | 仅用于 `verbose_json` 的提示；实测语言见 `X-Language` |
| `response_format` | 否 | `json`（默认）`text` `verbose_json` `srt` |
| `punctuate` | 否 | `true` 时按停顿插入 `，`/`,` 并以 `。`/`.` 结尾（默认 false） |

响应：

```json
// response_format=json
{"text": "把主驾空调调到二十一度"}
```

```json
// response_format=verbose_json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 2.32,
  "text": "把主驾空调调到二十一度。",
  "words":  [{"word": "把", "start": 0.00, "end": 0.18}, ...],
  "segments": [{"id": 0, "start": 0.00, "end": 2.10, "text": "把主驾空调调到二十一度", "language": "zh"}],
  "speaker_score": 0.921,
  "speaker_ok": true
}
```

声纹评分完成时才会返回 `speaker_score` 和 `speaker_ok`，具体状态见第 7 节。

`words` 的开始和结束用秒表示，由 CIF 锚点推导。热词改写增减 token 时保留原时间区间，后续词保留原时间。

### 3.1 响应头（所有格式）

| 头 | 含义 |
|---|---|
| `X-Queue-Ms` | 完整请求进入服务端队列后到 worker 开始处理的时间 |
| `X-Infer-Ms` | 推理耗时（VAD 分段时为各段之和） |
| `X-Vad-Ms` | 能量检查、VAD 和分段规划耗时 |
| `X-Language` | 检测到的语言为 `zh`、`en` 或 `mixed`，空转写为 `und` |
| `X-Vad-Pieces` | 本请求实际送进 ASR 的段数。长音频裁剪为一段时也为 1 |
| `X-Tail` | `hot` = 音频以语音结尾（客户端可能把尾音切掉了）；`silence` = 结尾有静音 |
| `X-ASR-Backend` | 实际处理这次请求的 ASR 引擎 |
| `X-Empty` | 值为 `silence` 时表示判定为静音、直接返回空文本 |
| `X-Speaker-Score` / `X-Speaker-Ok` | 声纹相似度与是否通过阈值（开启门控时） |
| `X-Speaker-Status` / `X-Speaker-Reason` | 评分状态及未评估原因，适用于两个 POST 接口 |

## 4. POST /asr/segment

请求体：raw `float32` little-endian PCM（16 kHz 单声道），`Content-Type: application/octet-stream`。

```json
{"text": "把主驾空调调到二十一度", "duration": 2.322}
```

可选 query：`format=json|text`。适合设备内部调用（省掉 WAV 封装）。

两种 POST 接口和 WebSocket 的单段音频上限为 30 秒。HTTP 请求体上限为 2 MiB。原始 PCM 样本必须是有限值。

## 5. WS /v1/realtime

客户端事件与响应：

| 客户端发送 | 服务端返回 |
|---|---|
| `{"type":"session.update","session":{...}}` | `session.created`、`session.updated` |
| `{"type":"input_audio_buffer.append","audio":"<base64 PCM16>"}` | `conversation.item.input_audio_transcription.delta`（节流 partial） |
| `{"type":"input_audio_buffer.commit"}` | `input_audio_buffer.committed` → `...transcription.completed`（含 `speaker_score`）+ `x.pf.metrics` |
| `{"type":"input_audio_buffer.clear"}` | `input_audio_buffer.cleared` |
| 出错 | `{"type":"error","error":{...}}` |

每个已接收的 commit 有独立的 `item_id`，连接存续时会收到对应的最终结果或带同一 `item_id` 的错误。允许连续提交，调用方按 `item_id` 匹配结果。`clear` 丢弃尚未提交的音频和预览。队列满时服务端返回错误，保留尚未接收的音频供客户端重试。

最终事件附带 `vad_pieces`，静音时附带 `empty: silence`。`x.pf.metrics` 里的 `vad_ms`、`queue_ms`、`infer_ms` 和 `backend` 与 HTTP 响应头口径一致。

```json
// completed
{"type":"conversation.item.input_audio_transcription.completed",
 "item_id":"item_1","content_index":0,
 "transcript":"把主驾空调调到二十一度",
 "speaker_score":0.921,"speaker_ok":true}
// x.pf.metrics（非标准，便于观测）
{"type":"x.pf.metrics","item_id":"item_1","queue_ms":1.2,"infer_ms":305.4}
```

partial 的节流：新增音频 ≥ `--partial-ms`（默认 800 ms）且距上次 ≥ `--partial-interval-ms`（600 ms），
且队列为空、音频长度 ≥ `--partial-min-s` 且 < `--partial-max-s`（两者默认都是 2.0 s）。partial 永远给 commit 让路。

`--partial-min-s` 默认等于 `--partial-max-s`，也就是默认**不发** partial，只发整句结果。
原因不是省算力：Paraformer 会随音频增长修正前面的字，而 delta 只能追加不能改，
服务端只能把修正后的结果丢掉。实测 1.8 s 的「把主驾空调调到二十」中途只弹出一次
「把主架扩」，然后停在屏幕上 1.1 s 直到整句完成——错字比空白更糟。短命令等 final，
说完到出字 187–369 ms（1.2/1.8/2.3 s 三个长度）。

要开预览必须**同时**放宽两个阈值，例如 `--partial-min-s 0.8 --partial-max-s 8`，
并接受短句上的错字停留。

## 5.1 连续流 + 服务端断句（推荐）

客户端只负责推音频，断句由服务端 VAD 决定：中间出现声学停止就出一次结果，不需要客户端 commit。

```json
// 客户端首帧
{"type":"session.update","session":{"turn_detection":{
  "type":"server_vad","silence_duration_ms":500,"min_speech_duration_ms":200,"threshold":0.5}}}
// 服务端确认
{"type":"session.updated","session":{"turn_detection":{"type":"server_vad"}}}
```

之后持续 `input_audio_buffer.append`（20 ms 一片）即可，每次判定到停顿服务端发一对事件：

```json
{"type":"input_audio_buffer.committed","item_id":"item_1","previous_item_id":null}
{"type":"conversation.item.input_audio_transcription.completed","item_id":"item_1",
 "transcript":"把主驾空调调到二十一度","vad_pieces":1,"usage":{"type":"duration","seconds":2.3}}
{"type":"x.pf.metrics","item_id":"item_1","queue_ms":1,"infer_ms":278,"vad_ms":16}
```

要点：

- **结果延迟 = `silence_duration_ms` + 推理时间**：500 ms 停顿 + 约 280 ms 推理，即最后一个字之后
  约 0.8 s 出结果。这是"服务端断句"的固有代价，比客户端自己切段晚一个停顿时长。
- 段落音频由检测器给出，**包含尾部静音**，因此不会出现"客户端切太早、末尾被截断"的问题。
- 停顿短于 `silence_duration_ms` 不会断句；一句话中间的长停顿会把它切成两段（实测 2 s 停顿
  切出两段，500 ms 停顿不切）。
- 仍然可以显式 `commit`（例如用户手动结束），服务端会把剩余音频作为最后一段。
- 预览（`x.pf.partial`）默认**关闭**；它是"整段假设的刷新"，与断句结果无关。

## 6. 错误

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "file", "code": null}}
```

| 状态 | 场景 |
|---|---|
| 400 | 缺 `file`、非 WAV、采样率/声道/位深不符、`response_format` 不支持 |
| 429 | 排队请求数或 PCM 缓冲区超过 `--queue` 或 `--queue-mib` |
| 500 | 推理失败 |
| 503 | 排队超时（`--timeout-ms`） |
| 504 | 从提交最终请求起超过 `--deadline-ms`，`error.code` 为 `deadline_exceeded` |

服务最多接受四个活跃 WS 会话，每个会话最多有四条等待终态的 COMMIT。第五个会话在升级后收到关闭码 1008。单条 COMMIT 超过截止时间时，服务用同一 `item_id` 发送一次 `error`。`/metrics` 提供 `ws_sessions`、`inflight`、`queue_bytes`、`queue_bytes_limit` 和 `deadline_expired`。

## 7. 说话人门控（可选）

评分默认关闭。启用时需要 `-DXASR_SPEAKER=ON`，同时提供 `--speaker-model` 和 `--enroll-dir`。请求启用但模型或注册目录不可用时，服务启动失败。注册目录存放司机的 16 kHz WAV，服务在启动时建立质心声纹。`/health` 的 `speaker_gate` 表示评分是否启用。

评分使用一个独立 CPU 线程，最多处理一个声纹任务。ASR 有排队或运行中的请求时，后来的请求跳过评分。评分只复制音频前 `--speaker-max-s` 秒，默认 1.5 秒。ASR 完成时立即读取已就绪的评分。阈值 `--speaker-threshold` 默认为 0.55，由调用方根据业务条件决定如何使用结果。

REST 响应头为 `X-Speaker-Status` 和 `X-Speaker-Reason`。`verbose_json`、原始 PCM JSON、WS 最终事件分别以 `speaker_status` 和 `speaker_reason` 返回同样的状态。`speaker_score` 和 `speaker_ok` 仅在 `assessed` 时返回，此时 `speaker_ok: false` 表示完成了评分且低于阈值。

| 状态 | 原因 |
|---|---|
| `disabled` | 功能关闭 |
| `assessed` | 评分已完成 |
| `not_assessed` | `busy`、`in_progress`、`no_audio_in_prefix`、`no_embedding` 或 `silence` |

`X-Speaker-Deferred: 1` 保留给跳过的非静音评分。`/metrics` 里的 `speaker_started`、`speaker_assessed` 和 `speaker_deferred` 分别统计提交给评分线程、有结果、未取得结果的请求。真机对照与数据边界见 [生产化记录](../../../examples/paraformer/PRODUCTION.md)。

## 8. 服务端 VAD（可选）

**默认关闭**：给出 `--vad-model silero_vad.onnx` 才启用。设计上刻意保守，因为座舱前端通常
已经用自己的 VAD 切好段、尾部静音很少：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--vad-split-s` | 18 | 只有超过该长度的 commit 才可能被拆分；正常短句**原样送 ASR** |
| `--vad-piece-s` | 15 | 合并相邻语音区间的目标长度。连续长语音交给模型内部分窗 |
| `--vad-threshold` | 0.5 | silero 语音概率阈值 |
| `--vad-min-silence` | 0.15 s | 判定段结束所需静音（比常见 0.3–0.5 s 更短，适配短命令） |
| `--vad-min-speech` | 0.20 s | 最短语音段，低于则丢弃该段 |
| `--silence-rms` | 0.004 | 整段能量低于此且 VAD 无语音时，直接返回空文本（防幻觉） |

行为：

1. 普通短句原样送入 ASR。能量低于 `--silence-rms` 时，才额外用 VAD 确认是否存在语音。
2. 长段按 VAD 语音区间合并，最多保留 100 ms 不重叠的边界上下文。即便只得到一段，也按实际区间裁剪。REST 的词级时间加回分段偏移，保持整段音频的时间基准。
3. 全零 PCM 直接返回空文本。开启 VAD 时，低能量且无语音也返回空文本。HTTP 保留请求的响应格式，WS 仍返回正式完成事件。
4. 能量高于阈值且 VAD 没找到语音时，整段交由 ASR 处理。

WAV、原始 PCM 和 WS 的最终转写共用上述流程。一个请求占用一个引擎，直至它的所有片段完成或连接取消。帧协议压测工具单独运行模型内分窗。

## 8.1 HTP 看门狗

`--htp-watchdog-ms N` 在单次后端调用超过 N 毫秒时直接退出进程（状态 75），不展开栈：崩溃点可能停在 FastRPC 里，回收资源本身不安全。覆盖范围为后端发现、引擎预热和单次推理。0 表示关闭，默认关闭。

预热和发现比单次推理慢得多：SM6650 上 Q8_0 分别是 1.3 s 和 0.52–0.56 s。早先这段复用了推理阈值，阈值低于约 1.4 s 时服务根本起不来，进程在预热阶段就被判定超时。现在预热与发现使用 `--htp-watchdog-ms` 的二十倍作为上限，也可用 `--htp-warmup-ms N` 单独指定。同样的 400 ms 推理阈值现在能在 1332 ms 内完成启动，并在推理真正超时时照常退出。

阈值必须高于部署音频长度下的最慢单次推理。400 ms 阈值配 13.9 s 音频会按设计退出，因为那段音频需要约 1.4 s。预热不会弥补这个差距：预热后首条推理 560 ms、稳态 522 ms，应按稳态加余量取值，而不是按预热耗时。

## 9. 示例

```sh
curl -sS localhost:8899/health
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@zh.wav
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@zh.wav -F punctuate=true
curl -sS -X POST localhost:8899/v1/audio/transcriptions -F file=@zh.wav -F response_format=verbose_json
```

```python
import asyncio, base64, json, wave, websockets

async def main(path='zh.wav', url='ws://127.0.0.1:8899/v1/realtime'):
    with wave.open(path, 'rb') as f:
        pcm = f.readframes(f.getnframes())
    async with websockets.connect(url) as ws:
        await ws.recv()                     # session.created
        for i in range(0, len(pcm), 640):   # 20 ms
            await ws.send(json.dumps({'type': 'input_audio_buffer.append',
                                      'audio': base64.b64encode(pcm[i:i+640]).decode()}))
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
        while True:
            m = json.loads(await ws.recv())
            if m['type'].endswith('completed'):
                print(m['transcript'], m.get('speaker_score'))
                break

asyncio.run(main())
```

## 10. 验收脚本

部署后运行自检，检查项随启用的能力变化。

```sh
python3 examples/paraformer/tools/smoke_production.py --port 8899 \
    --wav ~/Library/Caches/qwen3-asr-htp/paraformer-models/tingting/hvac_zh01.wav \
    --expect-gate --expect-vad
```

覆盖：`/health` 能力位、`/v1/models`、四种 `response_format`、`punctuate`、响应头
（语言/排队/推理/门控）、400 错误（不支持的格式、缺 file）、`/asr/segment`、静音返回空文本、
WS 单路与四路并发（含一致性）、空提交、连续提交后清空、非有限 PCM、超长音频和 `/metrics`。退出码 0 表示本次启用的检查全部通过。

最初的真机参考输出（SM6650，门控 + VAD 都开）：18 项 PASS，json 355 ms、WS 单路 331 ms、
四路并发 max 808 ms 且转写一致。

## 11. 客户端集成要点

1. **端点即 commit**：前端 VAD 判定说完后，把这一段一次发完再 commit；不需要客户端发送静音尾。
2. **超时**：单段推理约 0.3 s（2.3 s 音频），队列满会返回 429；建议客户端超时 ≥ 5 s 并重试一次。
3. **不要并发超限**：并发数 = `--engines`；超过会排队，`X-Queue-Ms` 会体现。
4. **语言**：用 `X-Language` 而不是请求里的 `language`（后者只是提示）。
5. **门控**：如果要用司机声纹，需按行程注册（同麦、同增益），阈值上线前用真实录音标定。
