# Qwen3-TTS 重训 pipeline（T1 predictor / T2 vocoder）

配套文档：[训练计划](../..//docs/backend/snapdragon/qwen3-tts-training-plan.md)（卡/数据/机时）、
[最终方案](../../docs/backend/snapdragon/qwen3-tts-v73-plan.md)（为什么改这两处）、
[P2a 规格](../../docs/backend/snapdragon/qwen3-tts-predictor-redesign.md)（算法细节）。

目标很窄：把 predictor 从 16 次自回归前向改成 1 次（T1），把 vocoder 的残差单元从 3 个减到 2 个（T2）。
前者让 LM 从 182 → 44 ms/帧，后者让 vocoder 从 111 → 84 ms/帧。

---

## 0. 五分钟先看这里

```
prepare_pairs.py ──> shards ──> train_predictor.py ──> ckpt ──> export_patch.py ──> patch
                                                                                    │
                                                              patch_mmproj.py <────┘
                                                                     │
                                                                     ▼
                                                          mmproj.gguf（上设备验证）
```

| 步骤 | 命令 | 需要 GPU | 耗时 |
|---|---|---|---|
| 0 环境自检 | `python smoke_test.py` | 否（CPU 1 分钟） | 1 min |
| 1 测 encoder 吞吐 | `prepare_pairs.py --dry-run` 之后用真数据跑 10 个 clip | 是 | 0.5 h |
| 2 生成 shards | `prepare_pairs.py --kind t1/t2` | 是 | 10–30 GPU·h |
| 3 **S0 判据** | `train_predictor.py --s0` | 是 | **8 GPU·h** |
| 4 T1 训练 | `train_predictor.py` | 是 | 10–20 GPU·h |
| 5 导出 + 打补丁 | `export_patch.py` + `patch_mmproj.py` | 否 | 2 min |
| 6 上设备验证 | `llama-tts` / `llama-vocbench` | 设备 | 0.5 h |
| 7 T2 | `QWEN3TTS_WAV_RES=2` 转 mmproj + 恢复训练 | 是 | 55–80 GPU·h |

**先跑第 0 步**：它不需要 GPU，能证明这套代码在你机器上是通的。

---

## 1. 环境

```sh
pip install torch safetensors numpy soundfile
# 只有 prepare_pairs.py 需要（读参考实现，生成数据）
pip install "transformers>=4.57" qwen-tts
```

**注意**：`qwen-tts` 要求 `transformers>=4.57`。本仓库开发机上装的是 4.41，
导入就直接失败（`ALL_ATTENTION_FUNCTIONS` 不存在），所以 `prepare_pairs.py` 那条路**没有被端到端跑过**，
其余部分（格式、模型、训练循环、导出、打补丁）都在 CPU 上跑通了（见 §6）。

---

## 2. 数据准备（prepare_pairs.py）

输入：24 kHz 单声道 wav（200–300 小时），输出两种 shard：

| kind | 内容 | 给谁 |
|---|---|---|
| `t1` | `hidden[T,1024] f16` + `codes[T,16] i32` | predictor 训练 |
| `t2` | `codes[T,16] i32` + `pcm[T*1920] f16` | vocoder 恢复训练 |

```sh
# T1：talker hidden + codes（教师强制，不需要文本）
python prepare_pairs.py --kind t1 --wavs "/data/corpus/**/*.wav" --out /data/shards_t1 \
    --speaker /data/voice/ref.wav --device cuda

# T2：codes + 波形
python prepare_pairs.py --kind t2 --wavs "/data/corpus/**/*.wav" --out /data/shards_t2 --device cuda
```

**两个坑**：

1. **prompt 要和部署一致**。T1 的 hidden 是"给定 prompt 后 talker 会输出什么"，
   用克隆 prompt 训出来的 predictor 只对克隆路径好；CustomVoice 要用对应的 voice 行。
2. **codes 必须来自部署用的同一个 tokenizer**，否则 T2 学到的是另一套码本。

每个 shard 旁边有 `.json` 记录形状（`n`/`n_embd`/`n_codes`/`hop`），下游按它校验。

---

## 3. S0 判据（8 GPU·h，决定后面 8 周做哪一档）

```sh
python train_predictor.py --s0 \
    --shards "/data/shards_t1/*.npz" --max-steps 400 --batch 256 --device cuda \
    --out /data/s0.ckpt.safetensors
```

**建议同时跑一组 `--kd`**（见 §6.5）：如果软标签能把 agreement 从 55% 推到 65%，整个项目就从 B 案跳到 A 案。

看最后一行的 `S0 GATE`：

| agreement | 结论 | 后续 |
|---|---|---|
| ≥ 60% | **A 案（全并行）** | 按本 pipeline 继续 |
| 50–60% | **B 案（2 次前向，8+7 分组）** | 改 `TowerLayer` 分成两组、图里跑两次 |
| < 50% | **C 案（蒸馏 3 层 ×512，保留 AR）** | 回归原结构，只砍宽度 |

同时看 `per-book`：前 4 个 book 应该明显高于后面的；如果全都 0，说明数据或 prompt 有问题，先别往下走。

---

## 4. T1 正式训练

```sh
python train_predictor.py \
    --shards "/data/shards_t1/shard_0*.npz" \
    --eval-shards "/data/shards_t1_eval/*.npz" \
    --init /path/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors \
    --epochs 30 --batch 512 --lr 1e-4 --device cuda \
    --out /data/predictor_1pass.safetensors
```

`--init` 用原 checkpoint 热启动：v/o/mlp/norm/heads 原样搬过来（q/k 丢掉，因为单位置注意力下它们不影响输出）。
不传 `--init` 就是随机初始化，只适合冒烟测试。

**验收**（三层，缺一层都可能翻车）：

| 层 | 指标 | 门槛 | 在哪看 |
|---|---|---|---|
| 逐码 | top-1 agreement | ≥ 60% | 训练日志最后一行 |
| 分布 | 每 book 熵 ≥ 教师 80% | 防坍缩到高频码 | 同上（`entropy`） |
| 波形 | 端到端 SNR / 盲听 | SNR ≥ 20 dB、MOS 差 < 0.2 | 见 §5 |

---

## 5. 导出 + 打补丁 + 上设备

```sh
python export_patch.py /data/predictor_1pass.safetensors /data/pred_patch.safetensors
python patch_mmproj.py mmproj.gguf /data/pred_patch.safetensors mmproj-new.gguf --verify
```

补丁是**按字节偏移原地覆盖**同名张量（形状必须一致），所以元数据、对齐、偏移都不变，
不需要重新串一遍 GGUF。张量映射（37 个）见 `export_patch.py` 的 `LAYER_MAP`/`TOP_MAP`。

验证：

```sh
# 设备上：数值（与旧 mmproj 逐样本比）与耗时
./llama-vocbench -m backbone.gguf -mm mmproj-new.gguf -dev CPU -n 32 -c 2 -o new.f32
./llama-tts ... -dev none -n 24 --seed 1234     # TTS_TRACE=1 看 gen_code ms
```

预期：`gen_code` 从 168–197 ms 掉到 ~15 ms（设备实测替身 11.6–14.7 ms，见 experiment log E23）。

---

## 6. T2（vocoder 减重）

**结构改造已经可用，不需要改代码**：

```sh
QWEN3TTS_WAV_RES=2 python convert_hf_to_gguf.py <model> --mmproj --outfile mmproj-res2.gguf
```

运行时按文件里的张量数自适应（`blk.res.size()`），所以直接就能跑，音质会降（实测 3.1 dB，见 E18）。
要恢复音质需要**恢复训练**，这一步本目录只给了规定，没有实现：

| 项 | 规定 |
|---|---|
| 数据 | `t2` shard（codes + pcm） |
| 初始化 | 原 vocoder 权重；被删的第 3 个残差单元直接丢弃 |
| 损失 | 时域 L1 + 多尺度 STFT（建议 3 尺度）+ 可选多周期判别器 |
| 冻结策略 | 先只训剩余残差单元与 ConvTranspose（10 epoch），再全量 finetune（40–80 epoch） |
| 验收 | PCM SNR ≥ 20 dB（对原 vocoder）、UTMOS 不低于原 -0.2、盲听 |
| 导出 | 训练脚本按 GGUF 名导出（`a.gen.wav.dac.*`），再用 `patch_mmproj.py` 打回 |

**为什么没实现**：需要参考实现的 decoder 在 torch 里跑，而它同样卡在 `transformers>=4.57`；
同时训练脚本依赖判别器设计，属于另一个工作项，硬写一个没跑过的 trainer 不如把规定写清楚。

---

---

## 6.5 KL 自蒸馏（`--kd`，可选但强烈建议在 S0 就试）

```sh
# S0 两组对照，各 8 GPU·h
python train_predictor.py --s0 ... --out /data/s0_ce.safetensors
python train_predictor.py --s0 ... --kd --kd-weight 1.0 --kd-temp 2.0 --out /data/s0_kd.safetensors
```

老师是**原 15 步 AR predictor 本身**（同一份权重，冻结，只用来产生每步的分布），
学生是 1-pass 塔。损失 = 硬标签 CE + `kd_weight` × KL(老师‖学生, T)。

实现：[model_ar_teacher.py](model_ar_teacher.py) 忠实复刻了部署端的 AR 链
（16 个位置、GQA 按 `i//2` 分组、NEOX RoPE θ=1e6、q/k 逐头 RMSNorm、因果 mask），
因为"老师和部署时的模型不一致"是这类蒸馏最隐蔽的坑。

### 优点

| 优点 | 说明 |
|---|---|
| **软标签降方差** | 2048 类的分类，硬标签只说"哪个对"；软标签还告诉"哪些接近"，梯度更稳 |
| **自动容忍多解** | RVQ 靠后的 book 常有多个合理 code，硬标签会惩罚合理答案，KD 不会 |
| **可能把 S0 从 B 案推到 A 案** | CE 55% vs CE+KD 65%，对应 RTF 1.6 vs 0.87 —— 这是本项目里性价比最高的一次赌博 |
| **零额外数据** | 老师就在同一份 checkpoint 里，不需要新标注、不需要 1.7B |
| **温度可调** | T=2 让分布更平滑，适合训练后期；T 小则更接近硬标签 |

### 缺点 / 代价

| 缺点 | 量级 | 缓解 |
|---|---|---|
| **训练时间 ×1.9** | 实测 30 步：13.5 s（CE）vs 25.7 s（CE+KD），CPU | 老师只有 5 层，可接受；GPU 上比例相近 |
| **显存增加** | 老师 logits 每样本 15×2048×4 B ≈ 120 KB，batch 512 时 63 MB；另有 16×5 层的中间激活 | batch 减半，或老师用 no_grad + 及时释放 |
| **老师有偏差会被学走** | 若原 AR predictor 在某 book 上系统性偏错，学生一样错 | 监控每 book 熵（学生熵不应低于老师的 80%） |
| **救不了结构性缺陷** | 若 1-pass 表达不了（如 book15 强依赖 book14），KD 只是让失败更平滑 | S0 闸门仍看真实 agreement，不看 loss |
| **多两个超参** | `kd_weight`、`kd_temp` | 从 (1.0, 2.0) 起；若学生熵过低，降到 0.5 |
| **老师必须逐位对齐部署** | 本实现按 C++ 逐行复刻，但**没有用真实权重做数值比对**（checkpoint 已清理） | 第一次跑真数据时，用同一批 codes 比老师 top-1 与 llama.cpp 的 AR 输出 |

### 怎么用它的结论

- 两组 S0 跑完比 agreement：**提升 ≥3 个点**就带着 KD 往下走；
- 提升 <3 个点 → 说明瓶颈是结构而不是标签，按 CE 那组的结论选 A/B/C；
- 两组都 <50% → C 案，KD 仍然值得保留（它让 C 案的质量更好）。

### 已验证到什么程度

- ✅ 老师前向能跑通（16 位置、注意力、GQA、RoPE），输出 `[B,15,2048]`；
- ✅ KL 在分布不同时为正、相同为 0、只对学生产生梯度（单测：随机分布 KL=1.000，自身 KL=0.000，结构老师 KL=1.276）；
- ✅ 训练循环带 KD 能跑完并下降（30 步，kd 项 0.399）；
- ❌ **没有用真实权重验证过老师的数值**（没跑过真数据），这条写在 README §8 的已知限制里。

---

## 7. 文件职责

| 文件 | 作用 | 跑过吗 |
|---|---|---|
| `datafmt.py` | shard 格式读写与校验 | ✅ smoke |
| `model_predictor.py` | 1-pass predictor（5 层 FFN 塔 + 15 头）+ 指标 | ✅ 用真实权重初始化验证 |
| `model_ar_teacher.py` | AR 老师（注意力/GQA/RoPE 逐行对齐部署端）+ KL 损失 | ✅ 前向/KL/梯度单测 |
| `train_predictor.py` | S0/T1 训练循环、agreement/熵评估、存 ckpt、`--kd` 蒸馏 | ✅ smoke（CE 与 CE+KD 都跑通） |
| `export_patch.py` | ckpt -> GGUF 命名的 safetensors 补丁 | ✅ 37 张量、94.4 M |
| `patch_mmproj.py` | 按字节偏移打补丁 + 回读校验 | ✅ 与真实 mmproj 往返后**逐字节相同** |
| `prepare_pairs.py` | 语料 -> shards（需要参考实现） | ⚠️ 未跑（transformers 版本） |
| `smoke_test.py` | CPU 一键自检 | ✅ `SMOKE OK` |

### 已经验证过的一件事（值得单独说）

用**真实 checkpoint** 初始化 1-pass 模型，再导出补丁、打回 mmproj，结果与原文件**逐字节相同**
（`cmp` 通过）。这证明张量命名、形状、字节序、dtype 全部对齐 —— 训练出来的权重可以安全地写回去。

---

## 8. 已知限制

1. `prepare_pairs.py` 没跑过：它调用参考实现的 `generate(codec_input=...)` 做教师强制，
   这个参数名/返回结构需要在你那边对着 4.57+ 的代码确认一次（半小时）。
2. T2 的恢复训练没实现（见 §6）。
3. 所有机时数字是估算（见计划文档 §3），实际以你那边 10 分钟的吞吐实测为准。
4. 验收最后一关是耳朵，自动化不了。
5. **KD 的老师没有用真实权重做过数值比对**。本实现按 C++ 逐行复刻（GQA `i//2`、NEOX RoPE、q/k 逐头 norm、位置 0..15），
   单测验过前向/KL/梯度，但真实 checkpoint 已清理，没跑过"同一批 codes 下老师的 top-1 与 llama.cpp AR 输出一致"这一步。
   **第一次上真数据时先补这个 10 分钟的检查**，否则学生会学到一个部署时不存在的老师。
