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
| GET | `/metrics` | 计数：`requests, rejected, failed, partials, partials_skipped, queue, engines` |
| POST | `/v1/audio/transcriptions` | OpenAI 兼容转写（multipart/form-data） |
| POST | `/asr/segment` | 设备端低开销转写（body 直接是 float32 PCM） |
| WS | `/v1/realtime` | 实时：append / commit / clear，返回 partial 与最终结果 |
| GET | `/` | 浏览器测试台（`--web-dir` 存在时） |

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
且队列为空、音频长度 < `--partial-max-s`（2.0 s）。partial 永远给 commit 让路。

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
