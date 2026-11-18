# Paraformer cabin ASR — model assets (v1)

These are the exact weights the measured numbers depend on. Everything else the
service needs (binaries, launcher, config, hotword rules, web console) is in the
repository; the fixtures are generated locally with the platform TTS, so they are
not shipped here.

Install into the model directory the tools expect:

```sh
tar -xzf paraformer-zh-htp-v1.tar.gz -C "$HOME/models/paraformer" --strip-components=1
sha256sum -c "$HOME/models/paraformer/SHA256SUMS"
```

| file | bytes | sha256 | role |
|---|---:|---|---|
| `paraformer-zh-q8_0.gguf` | 237,454,336 | `e73232ecf7e484907fd58f6a481fa27c1ceff454a1bc441d4a032a3a63e9783f` | ASR weights, Q8_0, 8404-vocab zh checkpoint (60000 h zh+en). **Ships.** |
| `silero_vad.onnx` | 2,327,524 | `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f` | Server-side VAD (streaming endpoints, long-audio split, silence guard). |
| `spk_od.onnx` | 28,280,942 | `f85d7a17471bd3640301d35df98b9be2b6936122b1529aed0b654289ea2fa073` | CAM++ speaker embedding with the metadata sherpa-onnx requires. Optional: the gate is off by default. |
| `campplus_cn_common.bin` | 28,036,335 | `3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8` | Upstream CAM++ checkpoint, kept so `spk_od.onnx` can be re-derived and audited. |

## Where each file comes from

**paraformer-zh-q8_0.gguf** — converted from the FunASR Paraformer-large zh
checkpoint (`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`,
8404-token vocabulary, 60000 h zh+en training) and quantized to Q8_0. This is a
build output, not a download: the SHA-256 above is what
`examples/paraformer/tools/model_manifest.txt` pins and what
`tools/fetch_models.py` verifies. Re-derive it with the converter in this
repository; a different conversion moves the logit gaps that
`hotwords_vehicle.txt` was tuned against.

**silero_vad.onnx** — downloaded unmodified from ModelScope
`pengzhendong/silero-vad` (`silero_vad.onnx`, 2.2 MB). Upstream is
https://github.com/snakers4/silero-vad (MIT).

**campplus_cn_common.bin** — downloaded unmodified from ModelScope
`iic/speech_campplus_sv_zh-cn_16k-common` (CAM++, 192-dim, 16 kHz).

**spk_od.onnx** — derived from the file above in two steps, both in the repo:

```sh
# 1) export with funasr's CAMPPlus (fixed 2 s graph; dynamic axes are numerically
#    wrong at some lengths, see design doc section 30)
python3 examples/paraformer/tools/export_campplus_onnx.py \
    --bin campplus_cn_common.bin --out campplus_t200.onnx
# 2) add the metadata sherpa-onnx insists on (framework, output_dim, ...)
python3 examples/paraformer/tools/add_campplus_metadata.py \
    --in campplus_t200.onnx --out spk_od.onnx
```

## Verify a deployment uses these bytes

```sh
python3 examples/paraformer/tools/fetch_models.py --out "$PF_MODELS" --verify-only
examples/paraformer/tools/accept.sh 8899        # 23/23, 87.5%, 280/287 ms
```

## Licences and attribution

- Paraformer and CAM++ weights: FunASR Model License (Alibaba). Commercial use
  requires attribution; check the licence text before shipping.
- Silero VAD: MIT.
- The fixture voices are rendered locally with the macOS `say` TTS; no third-party
  audio is redistributed here.
