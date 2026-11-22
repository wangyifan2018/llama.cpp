# E36 五层 Q8 predictor 的安卓端到端 A/B

2026-11-22。在 SM6650 上用同一 `llama-tts`、同一 Talker、同一参考音色和同一份原精度 vocoder 合成四条独立短句，中英文各两条。基线使用原混合精度 mmproj，Q8 使用 E35 的全主干 Q8_0 文件。每对使用相同文本与 seed，两对按原版再 Q8、两对反序。各次启动新的进程，使用真实跨帧 Talker 反馈，并保存整句音频。

| 四条短句的中位数 | 原版 | predictor 主干 Q8_0 |
|---|---:|---:|
| predictor 每帧耗时，先去掉每条首帧再取中位数 | 198.70 ms | 97.68 ms |
| 首个 PCM chunk 的请求延迟 | 2.588 s | 1.821 s |
| RTF | 3.926 | 2.677 |
| 请求总耗时 | 11.395 s | 8.885 s |
| 音频时长 | 2.920 s | 3.320 s |

Predictor 约 2.03x，首包约减少 30%，RTF 约减少 32%，请求总耗时约减少 22%。四对的 Q8 请求都更短，生成音频都更长，分别多 0.80、0.16、0.24 和 0.56 s。当前样本不能区分这些时长差异来自语速、停顿或句尾。两档均未达到实时。所有请求均在第 8 帧给出首个 PCM chunk，且都在 96 帧上限前结束。TTFA 和请求总耗时沿用程序口径，从模型加载和参考音色预处理完成后开始，包含 prompt prefill、生成与 flush。首次进程初始化也有原始记录。

本地 Qwen3-ASR-0.6B 对 8 条音频均转写出对应输入。英文按词，中文按字，忽略大小写与标点后，每档均为 **0/15 英文词错误、0/24 中文字错误**。这个小样本提供了自动转写的可懂度对照。

当前 CLI 在流式模式下忽略仅包含 WAV 头的第一次回调，因此本次输出文件虽有 wav 后缀，内容实际为 PCM16。听测和 ASR 文件根据采样率与帧数补上标准 WAV 头，原始 PCM 字节逐个核对。没有裁剪、归一化或重采样。匿名 A/B 文件位于 `~/Desktop/qwen3-tts-e36-audio/blind/`，答案单独放在上一级的 `blind-key.tsv`。

## 第一轮人工听测

一位听者反馈「发音还算清晰」。以下只对照已评论的标签。

| 样本 | 听感 | 标签核对 | 文件时长 |
|---|---|---|---|
| P01 | A 较慢，B 较快 | A 为 Q8，B 为原版 | A 4.00 s，B 3.20 s |
| P02 | B 音色比较明亮 | B 为原版 | B 2.40 s，另一版 2.56 s |
| P03 | B 音色明亮，而且比较快 | B 为原版 | B 2.64 s，另一版 2.88 s |

P01 和 P03 的速度听感与文件时长方向一致。整段时长含停顿与句尾，无法由这组数据单独定位到发音速度。音色观察来自一位听者的两条短句。P04 尚未收到听感。

## 复跑

先按 [E28](README.md) 配置独立研究 worktree 和安卓 CPU 库，按 [E35](quant-predictor.md) 生成并推送全主干 Q8 权重。E36 的研究补丁只给 MTMD worker 添加可选逐核绑定。其他应用无需使用该开关。

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
run=/tmp/qwen3tts-e36-run
out="$HOME/Desktop/qwen3-tts-e36-audio"
adb="$HOME/Library/Android/sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-e2e

git -C "$src" apply "$exp/e2e-pin.patch"
cmake --build "$src/build-phone" --target llama-tts -j 6
mkdir -p "$run"
"$adb" shell "mkdir -p $device"
"$adb" push "$src/build-phone/bin/llama-tts" "$src/build-phone/bin/libllama-common.so" \
    "$src/build-phone/bin/libmtmd.so" "$exp/e2e-device.sh" "$exp/e2e-prompts.tsv" "$device/"
"$adb" shell "cd $device && chmod 755 llama-tts && sh e2e-device.sh" | tee "$run/device-progress.log"
for id in p01 p02 p03 p04; do
    for mode in original q8; do
        "$adb" pull "$device/$id-$mode.wav" "$device/$id-$mode.log" "$run/"
    done
done
python3 "$exp/e2e-summary.py" "$run" "$out" >"$run/summary.txt"

asr="$HOME/Library/Caches/qwen3-asr-htp"
QWEN_THREADS=4 "$PWD/build-qwen-host/bin/llama-qwen3-asr-batch" "$asr/Qwen3-ASR-0.6B-Q8_0.gguf" \
    "$asr/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf" CPU 1 "$out/asr-trace.tsv" >"$out/asr.jsonl" 2>"$out/asr.log"
python3 "$exp/e2e-asr-score.py" "$out"
```

ASR 构建见 [本地 Qwen3-ASR 批处理入口](../../../../examples/qwen3-asr-0.6b/README.md)。安卓采用 CPU4–7 严格绑核，trace 开启，streaming chunk 8 帧，top-k 50、top-p 0.8、temperature 1、模型加载模式 none。逐对运行参数见 [短句和执行顺序](e2e-prompts.tsv)。计时记录包括实际手机后台环境，采用每个配置同样的 CPU 绑定与执行脚本。

[逐对计时](results/e2e-paired.csv) · [汇总](results/e2e-summary.txt) · [ASR 转写和错误数](results/e2e-asr.csv) · [安卓输入散列](results/e2e-android-inputs.sha256) · [ASR 输入散列](results/e2e-asr-inputs.sha256) · [音频散列](results/e2e-audio.sha256)。原始 PCM、手机日志和 ASR 输出保存在 `~/Desktop/qwen3-tts-spec-raw-20261122/e2e/`。
