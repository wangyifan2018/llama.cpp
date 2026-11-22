# E29 用 1024 个参数蒸馏 CodePredictor 草稿

2026-11-22。用同一份 Qwen3-TTS 12 Hz 0.6B Base 权重，在 Mac M2 CPU 上训练窗口 2 的未来 RVQ head 共享偏置。老师、验证器与原有 head 全部冻结。

## 方法

- 在 E28 的模型与参考音色上，用 8 条新短句采集训练帧，4 条不同短句采集验证帧。两组各有一半中文和一半英文。每句最多生成 40 帧。E28 的 96 帧另作回归集。按整条短句划分，三个集合共用一个音色。
- 回放器批量输出目标 logits (p_g)、原草稿 logits (q_g) 和原始 F16 head 矩阵。共享偏置 (b) 只改变草稿，修正后 logits 为 (q_g+W_gb)。用原草稿 logits 加预计算修正即可训练，无需另导出 predictor hidden。
- 比较零偏置、原 AR 实际采样的 code 作硬标签的 CE、老师原始 softmax 分布的 KL。两次训练均从零初始化，Adam 全批量 120 步，学习率 0.01。训练只看深度 2–14。窗宽为 2 时，最后一个 code 无法再产生 bonus。
- 接受率按部署过滤 top-k 50、top-p 0.8、temperature 1 后的 (\sum_x\min(p(x),q(x))) 计算。结果是固定 AR 前缀上的理论接受率。完整投机请求仍需验证拒绝回退、整帧执行耗时和新音色泛化。

## 结果

| 数据 | 帧数 | 原 head | CE 偏置 | KL 偏置 |
|---|---:|---:|---:|---:|
| 新短句验证，深度 2–14 | 160 | 42.42% | 44.11% | **44.91%** |
| 新短句验证，深度 8–14 | 160 | 53.12% | 53.62% | **54.67%** |
| E28 回归，深度 2–14 | 96 | 41.23% | 43.47% | **44.32%** |

KL 在四条新短句上的整体接受率都提高了，增幅为 2.26–2.74 个百分点。训练集包含 320 帧。两次拟合、三个模型的评估和结果写盘合计 11.75 秒 wall time。采集和老师回放另计。

在新短句验证集上，KL 把 `code2` 从 2.68% 提到 4.65%，仍低于安卓约 28.67% 的计算盈亏线。`code8` 从 48.37% 提到 51.55%，`code14` 从 55.74% 到 55.80%。代入 E28 安卓四线程的单轮计时，`code8` 的计算加速估计从 1.15× 到 1.17×，`code14` 约为 1.20×。在 `code8`，KL 只多释放约 0.38 ms/轮的开销余量。偏置有作用，本轮窗口 2 没有获得足以改变部署选择的增量。不同音色与设备仍需要验证。

[逐深度数据](results/offset-summary.txt) · [逐句数据](results/offset-utterances.csv) · [CPU 时间](results/offset-cpu-time.txt) · [复跑一致性](results/offset-repeat.txt) · [文件散列](results/offset.sha256)

## 复现

先按 [E28](README.md) 构建带研究补丁的 `llama-tts`。在仓库根目录运行：

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
models=/tmp/qwen3tts-spec-models
run=/tmp/qwen3tts-offset-run
regression="$HOME/Desktop/qwen3-tts-spec-raw-20261122/frames.bin"

bash "$exp/build-replay.sh" "$src" "$src/build-host"
bash "$exp/capture-offset.sh" "$src/build-host/bin/llama-tts" "$models" "$run"
cat "$run"/tr{01,02,03,04,05,06,07,08}.bin \
    "$run"/ev{01,02,03,04}.bin "$regression" > "$run/offset-frames.bin"
TTS_SPEC_FLASH=1 TTS_SPEC_TRAIN_EXPORT="$run/heads.bin" \
    "$src/build-host/bin/qwen3tts-spec-replay" "$models/mmproj-qwen3-tts-0.6b-f16.gguf" \
    "$run/offset-frames.bin" "$run/offset.scores" 4 1 -1 >"$run/replay.csv" 2>"$run/replay.log"
OPENBLAS_NUM_THREADS=4 /usr/bin/time -p python3 "$exp/train-offset.py" \
    "$run" "$run/offset.scores" "$run/heads.bin" "$run/results" \
    --regression "$regression" >"$run/summary.txt" 2>"$run/train.log"
```

用的是 Python 3.12.2 和 NumPy 1.26.4。训练导出模式的 scores 文件只含单步目标 logits 和未来 head 的草稿 logits，帧内格式与普通回放的前两部分相同。

原始捕获、teacher logits、权重、逐帧结果和导出的偏置保存在本机 `~/Desktop/qwen3-tts-spec-raw-20261122/offset-distillation/`。
