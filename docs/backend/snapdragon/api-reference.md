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
| GET | `/health` | 探活 + 能力：`engines, queue, speaker_gate, server_vad, partials, model` |
| GET | `/v1/models` | OpenAI 形状的模型列表 + `capabilities` |
| GET | `/metrics` | 计数：`requests, rejected, failed, partials, partials_skipped, queue, engines` |
| POST | `/v1/audio/transcriptions` | OpenAI 兼容转写（multipart/form-data） |
| POST | `/asr/segment` | 设备端低开销转写（body 直接是 float32 PCM） |
| WS | `/v1/realtime` | 实时：append / commit / clear，返回 partial 与最终结果 |
| GET | `/` | 浏览器测试台（`--web-dir` 存在时） |

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

`speaker_score`/`speaker_ok` 仅在服务端开启声纹门控时出现（见 §7）。

### 3.1 响应头（所有格式）

| 头 | 含义 |
|---|---|
| `X-Queue-Ms` | 请求在服务端队列里等待的时间（VAD 分段时为 0） |
| `X-Infer-Ms` | 推理耗时（VAD 分段时为各段之和） |
| `X-Language` | 检测到的语言：`zh` / `en` / `mixed` |
| `X-Vad-Pieces` | 本请求实际送进 ASR 的段数（1 表示未拆分） |
| `X-Empty` | 值为 `silence` 时表示判定为静音、直接返回空文本 |
| `X-Speaker-Score` / `X-Speaker-Ok` | 声纹相似度与是否通过阈值（开启门控时） |

## 4. POST /asr/segment

请求体：raw `float32` little-endian PCM（16 kHz 单声道），`Content-Type: application/octet-stream`。

```json
{"text": "把主驾空调调到二十一度", "duration": 2.322}
```

可选 query：`format=json|text`。适合设备内部调用（省掉 WAV 封装）。

## 5. WS /v1/realtime

客户端事件与响应：

| 客户端发送 | 服务端返回 |
|---|---|
| `{"type":"session.update","session":{...}}` | `session.created`、`session.updated` |
| `{"type":"input_audio_buffer.append","audio":"<base64 PCM16>"}` | `conversation.item.input_audio_transcription.delta`（节流 partial） |
| `{"type":"input_audio_buffer.commit"}` | `input_audio_buffer.committed` → `...transcription.completed`（含 `speaker_score`）+ `x.pf.metrics` |
| `{"type":"input_audio_buffer.clear"}` | `input_audio_buffer.cleared` |
| 出错 | `{"type":"error","error":{...}}` |

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
| 429 | 队列满（`--queue`） |
| 500 | 推理失败 |
| 503 | 排队超时（`--timeout-ms`） |

## 7. 说话人门控（可选）

**默认关闭**：只有同时给出 `--speaker-model` 和 `--enroll-dir` 才启用（二进制需以
`-DXASR_SPEAKER=ON` 构建；否则该参数被忽略）。`/health` 的 `speaker_gate` 字段反映实际状态。

- 注册：`--enroll-dir` 目录下若干条司机 16 kHz WAV，启动时算质心声纹并打印平均自相似度；
- 判定：每次 commit 把该段与声纹做余弦，CPU 上与 HTP 的 ASR **并行**；
- 阈值：`--speaker-threshold`（默认 0.55）；
- 输出：`X-Speaker-Score`/`X-Speaker-Ok`、`verbose_json` 与 WS completed 事件里的同名字段；
- **只上报、不拦截**：拒绝与否由调用方（语义层）决定。

真机参考值（TTS 音色，仅作量级参考）：司机 0.921、乘客 0.363、注册自相似度 0.904。

## 8. 服务端 VAD（可选）

**默认关闭**：给出 `--vad-model silero_vad.onnx` 才启用。设计上刻意保守，因为座舱前端通常
已经用自己的 VAD 切好段、尾部静音很少：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--vad-split-s` | 18 | 只有超过该长度的 commit 才可能被拆分；正常短句**原样送 ASR** |
| `--vad-piece-s` | 15 | 拆分后每段的上限；连续语音段会被合并成整块，避免每遇逗号就切 |
| `--vad-threshold` | 0.5 | silero 语音概率阈值 |
| `--vad-min-silence` | 0.15 s | 判定段结束所需静音（比常见 0.3–0.5 s 更短，适配短命令） |
| `--vad-min-speech` | 0.20 s | 最短语音段，低于则丢弃该段 |
| `--silence-rms` | 0.004 | 整段能量低于此且 VAD 无语音时，直接返回空文本（防幻觉） |

行为：

1. 正常 commit（≤ `--vad-split-s`）→ 1 段，音频不改动，只多花约 15–25 ms 的 VAD 检测；
2. 超长 commit → 按语音段**合并**成 ≤ `--vad-piece-s` 的块，逐块识别后拼接；
3. 纯静音 → `X-Empty: silence`、空文本（实测未开启时模型会输出"嗯没有没有没有"这类幻觉）；
4. VAD 检测不到语音但能量不低（如轻声）→ 仍然整段送 ASR，不丢音频。

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

## 10. 客户端集成要点

1. **端点即 commit**：前端 VAD 判定说完后，把这一段一次发完再 commit；不需要客户端发送静音尾。
2. **超时**：单段推理约 0.3 s（2.3 s 音频），队列满会返回 429；建议客户端超时 ≥ 5 s 并重试一次。
3. **不要并发超限**：并发数 = `--engines`；超过会排队，`X-Queue-Ms` 会体现。
4. **语言**：用 `X-Language` 而不是请求里的 `language`（后者只是提示）。
5. **门控**：如果要用司机声纹，需按行程注册（同麦、同增益），阈值上线前用真实录音标定。
