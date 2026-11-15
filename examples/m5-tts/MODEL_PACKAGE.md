# M5-TTS 模型包：来源、生成链与复现路径

这份文件回答三个问题：**哪些东西来自上游、哪些由我们的脚本生成、每一步用哪条命令**。
[README](README.md) 讲"怎么跑"，这里讲"**怎么造出跑起来需要的东西**"。
性能结论见[优化日志 §1–§30](../../docs/backend/snapdragon/m5-tts-optimization-log.md)，
移植细节见 [port notes](../../docs/backend/snapdragon/m5-tts-port-notes.md)。

[公开 Release 权重](https://github.com/wangyifan2018/llama.cpp/releases/tag/m5-repro-v1)可直接解包到 `$OUT`。当前 `reproduce.sh` 固定上游 ONNX revision 为 `6a009494173b36257c1dc8611625956de08b27f3`，并按 SHA256 校验每个需要的上游文件。四个派生权重都匹配时，`fetch` 只取运行和准备音色所需的七个文件。

配套脚本：[`reproduce.sh`](reproduce.sh)（幂等，按哈希决定是否重建）。
本文里所有路径都是本机实际路径，换机器时按 `reproduce.sh` 顶部的环境变量覆盖。

---

## 0. 一句话结论

复现需要**三样**东西，缺一样都跑不起来；三者都在公网，没有私有依赖：

| 来源 | 是什么 | 本地路径（本机） | 取法 |
|---|---|---|---|
| **代码** | M5-TTS 上游仓库（PyTorch/ONNX 推理 + Android 打包器） | `~/Desktop/M5-TTS` | `git clone https://github.com/ASLP-lab/M5-TTS.git` |
| **权重** | `ASLP-lab/M5-TTS-50M-ONNX` 模型包（9 个运行文件） | `~/m5-assets/android_onnx`（**当前未落盘，需重下**） | `hf download ASLP-lab/M5-TTS-50M-ONNX --local-dir ~/m5-assets/android_onnx` |
| **推理引擎** | 本仓库 llama.cpp 分支 `codex/m5-tts-htp` | `~/Desktop/llama.cpp` | GitHub（本仓库） |

GitHub 上**没有**权重：上游 `android/.gitignore` 明确忽略 `/outputs/`，模型包按设计走
Hugging Face（上游仓库的 `android/README.md` §4.1 给的就是 `hf download` 命令）。所以"只拿到 GitHub 代码"能拿到**全部脚本**，但拿不到 **1.3 GB 的
ONNX 权重**——这是唯一必须先联网的一步。

反过来，**HF 上也没有我们派生的东西**：四步重写后的 codec 和从权重转出来的 gguf/f32。
这几个走 Release 附件（`reproduce.sh release` 或 CI），仓库里只留脚本和哈希，见 §8。

---

## 1. 三个来源的精确坐标

### 1.1 代码：`ASLP-lab/M5-TTS` @ `4e7cc915`

```sh
git clone https://github.com/ASLP-lab/M5-TTS.git ~/Desktop/M5-TTS
cd ~/Desktop/M5-TTS
git checkout 4e7cc91545ad5ecd368d7726129cefb41fc6e1d7   # 本机是 shallow clone，就这一个提交
git apply ~/Desktop/llama.cpp/examples/m5-tts/android-stream.patch
```

- 提交：`4e7cc91545ad5ecd368d7726129cefb41fc6e1d7`（"Extend text normalization to
  Japanese Korean German and French"，2026-11）。
- `android-stream.patch`（本仓库，4 个文件：`M5Engine.java` / `MainActivity.java` /
  `NativeBackbone.java` / `NativeDepth.java`）**能干净应用到该提交**——已用干净工作树
  `git apply --check` 验证。
- **已知偏差**：本机 `~/Desktop/M5-TTS` 另有 5 个**已修改**文件（`M5Engine.java`、
  `MainActivity.java`、`ModelInstaller.java`、`build.gradle`、`android/README.md`）
  和 5 个**未跟踪**文件（`NativeBackbone/NativeDepth/NativeEngine/TokenDump.java`、
  `jniLibs/`），是后来做 App 形态流式播放与 HTP 时加的（`useLegacyPackaging = true`
  让 stage 可执行、`native_files` 装包支持、native 引擎的 JNI 入口）。
  **这些改动没有导出成 patch**，所以从零复现能跑通 **server / CLI 路径**，
  App 路径只能靠这份 diff：

  ```sh
  git -C ~/Desktop/M5-TTS diff > android-app-htp.patch                    # 改过的 5 个
  git -C ~/Desktop/M5-TTS status --porcelain | grep '^??'                 # 未跟踪的 5 个
  ```

- 许可：仓库根目录无 `LICENSE`，README 要求"按各 release 附带的条款"；内嵌的
  Qwen 12Hz codec 实现是 Apache-2.0（`m5/codec_qwen12hz/NOTICE`）。

### 1.2 权重：`ASLP-lab/M5-TTS-50M-ONNX`

```sh
# 必须 3.11/3.12：onnx 1.16 与 onnxruntime 1.18 没有 cp313 的 wheel，
# 用 3.13 会去源码编译 onnx（慢且常失败）。本机 python3 恰好是 3.13。
python3.12 -m venv ~/m5-assets/pyenv && . ~/m5-assets/pyenv/bin/activate
pip install "huggingface_hub[hf_transfer]"   # 或 pip install huggingface_hub
hf download ASLP-lab/M5-TTS-50M-ONNX --local-dir ~/m5-assets/android_onnx
```

包内 9 个运行文件（上游 `android/README.md` §4.1 列的就是这个布局）：

```text
~/m5-assets/android_onnx/
├── backbone.onnx              116 MiB   backbone 的 ONNX 图（含权重）
├── depth_frame.onnx           267 MiB   depth 15 步的 ONNX 图
├── codec_decoder_cached.onnx  ~435 MiB  codec 解码器（**FP32 原始版**）
├── codec_cached.json           3.2 KiB  codec 的 21 个 conv state 形状 + IO 名
├── codec_encoder.onnx         182 MiB   音色编码器（克隆用）
├── codec_identity.json          91 B    encoder 的 SHA256（打包/引用校验）
├── text_embeddings.npy        254 MiB
├── audio_embeddings.npy        64 MiB
├── tokenizer.json                      tokenizer 源（我们转成 tokenizer.bin）
└── manifest.json               535 B    架构与采样超参（下面逐字段解释）
```

`manifest.json` 是**唯一说明"这是什么模型"的凭据**，关键字段：

```json
{ "architecture": "m5-llama-rq16-qwen3-12hz", "layers": 8, "kv_heads": 2, "head_dim": 64,
  "hidden_size": 512, "codebooks": 16, "audio_vocab": 2051, "sample_rate": 24000,
  "hop_length": 1920, "topk": 10, "temperature": 0.9, "min_frames": 6, "max_sequence": 2048,
  "checkpoint": "checkpoints/m5-v20/model.safetensors",
  "checkpoint_sha256": "5202c33cb8688588bc1d2723722cd930a76eacc0b89de61399bf28abb5e7e075" }
```

也就是说这份 ONNX 是 **v20 checkpoint** 的导出物。**只有要重新导出 ONNX 或重训时**
才需要 PyTorch 权重（`ASLP-lab/M5-TTS` 模型仓库，`checkpoints/m5-v20/model.safetensors`，
哈希如上）；跑推理不需要它。

> 本机现状：`~/m5-assets/android_onnx` 已被清掉（占 1.3 GB，属于可重下产物）。
> `~/m5-assets/device-model-backup/` 里保留的是**派生结果与运行时文件**（见 §7），
> 不含 `backbone.onnx`/`depth_frame.onnx` 原文——重下模型包才能重新生成 gguf。

### 1.3 引擎：本仓库

```sh
cd ~/Desktop/llama.cpp && git checkout codex/m5-tts-htp
```

所有转换脚本都在 [`examples/m5-tts/`](.)，只依赖 `numpy` / `onnx` / `onnxruntime`
（量化脚本对 ORT 版本敏感，见 §6）。`convert_backbone.py` 另外要 `PYTHONPATH=gguf-py`
（仓库自带的 gguf writer）。

---

## 2. 产物清单：11 个文件，谁生成、怎么验

体积是本机实测；`sha256` 是**逐字节可比**的（只有二进制不是，见 §4.3）。
"缺失症状"一列是排查用的——不要凭"能出声"判断复现成功。

| # | 产物（设备上） | 体积 | 生成者 | 缺失/错版症状 |
|---|---|---:|---|---|
| 1 | `model/manifest.json` | 535 B | 模型包原样 | 引擎直接报 `model package is incomplete` |
| 2 | `model/tokenizer.bin` | 5,484,755 | `prepare_android_voice.py` | 文本 tokenize 失败/乱码 |
| 3 | `model/voices.json` | 4 KB 起 | `prepare_android_voice.py` | 无音色可选；`cannot read voice` |
| 4 | `model/text_embeddings.npy` | 266,223,744 | 模型包原样 | `cannot load embeddings` |
| 5 | `model/audio_embeddings.npy` | 67,207,296 | 模型包原样 | 同上 |
| 6 | `model/depth-weights.f32` | 202,670,080 | `pipe_depth.py --android` | `cannot start the depth stage` |
| 7 | `backbone-Q8_0.gguf` | 32,351,200 | `convert_backbone.py --quant Q8_0` | 同上（backbone stage） |
| 8 | `depth-Q4_0.gguf` | 10,718,656 | `convert_backbone.py --depth --quant Q4_0` | 同上（depth stage） |
| 9 | `model/codec_cached.json` | 3,234 | 模型包原样 | codec 建 session 失败 |
| 10 | `model/codec_decoder_cached.onnx` | **140,935,716** | **四步链**（§3） | **能出声但慢 30–60%**（最容易漏的一项） |
| 11 | `codec_encoder.onnx` | 190,632,314 | 模型包原样 | 克隆接口 500；合成不受影响 |

运行时**必需**的是 #1–#5、#6、#7、#8、#10（`Engine::open` 逐个检查，#11 只在
`POST /v1/voices/encode` 时加载）。#9 由 codec session 读取；`backbone.onnx` /
`depth_frame.onnx` 只在 ONNX-CPU 基线和重新生成 gguf 时用，部署时可以不带。

### 2.1 本机基线的哈希（可直接比对）

来自 `~/m5-assets/device-model-backup/` 与设备：

```text
# 模型包原文（HF，逐字节可比）
103fc4f5ff86cd19d11d1e45126cf001996824e6db40ca3577c9e8c9a330b7df  backbone.onnx
44e8be816445e4574007628cfbf5af6c088f5f5c0b37218d112c6b333d739b82  depth_frame.onnx
ae0364957dd52df283e88baea8ea09d1f1f34b579cd53d2806b38290ebc03b57  codec_encoder.onnx
39c3849c8bbad6c94bb8e33784453d0af069335ac85da353419d7b41144afdb5  text_embeddings.npy
5c0e175c1feae525d66931dd81d727cb4f244f70121504c5d014f250afc51551  audio_embeddings.npy
17818a7d9e695c3953cdb5b05302df103981c94815b937a4db5c553067901dcd  manifest.json
295c4e2fe834b73c8a6523dde1bf316b2c8fb5784773f9e423a85e427c861d68  codec_identity.json

# 我们生成的（逐字节可比）
379a89db4c03245720ad1edd834a09c468e7c861821f940cb79c5458cf2808c2  backbone-Q8_0.gguf
45ae171494281f5466a11da9cd29b9e4508322125929b1a9117984630a993fc5  depth-Q4_0.gguf
9e38455b6180c1db3f56e73d35f78955a5fc2aa30a71297e427def131f2e4bc2  depth-weights.f32
e3edd1c94d1cbcfa2af2dfe2bec21a60279e6c6640cd6d7160ed1b2e40458c42  tokenizer.bin
ac3045a87ef9d485701977f8187eb0e27adc0d029488bc717c5b8da4a016f252  codec_decoder_cached.onnx
```

`tokenizer.bin` 与 3 个 gguf/f32 一致，就说明**权重链路对了**；
`codec_decoder_cached.onnx` 一致，就说明**四步链一步没漏**。
`voices.json` 不列哈希：音色名来自参考 wav 的文件名（本机把 `aslp_default_voice`
改成了 `官方样例`），哈希随名字变；比 `encoder_sha256` 字段和 `codes` 帧数即可。

---

## 3. codec 四步链（最容易漏、漏了最贵）

模型包里的 `codec_decoder_cached.onnx` 是 **FP32**（435 MiB）。生产用的是
"Conv 量化 → ConvTranspose 分解 → MatMul 量化 → Snake 融合"之后的重写版（134 MiB）。
四步都是**数学等价改写 + 量化**，不是重新训练：

```sh
P=~/m5-assets/pyenv/bin/python
O=~/m5-assets/android_onnx
W=~/m5-assets/model-package
S=~/Desktop/llama.cpp/examples/m5-tts

# ① Conv/ConvTranspose 走 QDQ per-channel INT8，最后 4 个 Conv 留 FP32（456 → 362 MB）
$P $S/quantize_codec_static.py "$O/codec_decoder_cached.onnx" "$W/codec-qdq-ex4.onnx" \
   --meta "$O/codec_cached.json" --voices "$W/voices.json" --exclude-last 4

# ② 6 个 ConvTranspose → 单 MatMul + 交错/pad（362 → 282 MB）
$P $S/decompose_convtranspose.py "$W/codec-qdq-ex4.onnx" "$W/codec-ct.onnx"

# ③ 68 个权重 MatMul 再走 QDQ INT8（282 → 141 MB）
$P $S/quantize_codec_static.py "$W/codec-ct.onnx" "$W/codec-ctmm-q.onnx" \
   --meta "$O/codec_cached.json" --voices "$W/voices.json" --ops MatMul

# ④ 29 处 Snake 链 → 一个 m5::Snake 自定义算子（尺寸不变）
$P $S/snake_fuse.py "$W/codec-ctmm-q.onnx" "$W/codec_decoder_cached.onnx"
```

四步的体积序列是**可观测的指纹**：`456,464,460 → 362,056,576 → 281,959,623 →
140,865,832 → 140,935,716`。中间任何一步漏掉，后面的数字都对不上。

每步的收益（设备实测，详见 port notes / 日志 §7–§8）：

| 步 | 效果 | 依据 |
|---|---|---|
| ① Conv QDQ | 456 → 362 MB，单点 Conv 5.7 → 0.93 ms | 末 4 个 Conv 的量化误差最大，留 FP32 白赚 2.7 dB |
| ② CT→MatMul | 362 → 282 MB，corr 0.9995 / 30 dB | 6 个反卷积变 GEMM，`(c,r)` 展平顺序是坑 |
| ③ MatMul QDQ | 282 → 141 MB，corr 0.973 / 12.8 dB | 激活×激活 MatMul（QK^T/AV）必须排除，否则再掉 1.3 dB |
| ④ Snake 融合 | codec 6.08 → 4.68 s（−23%），corr 0.9998 / 34 dB | 5 遍读写变 1 遍；图级展开反而更慢 |

注意 ③ 的 `--ops MatMul` **不是**"Conv 已经量化完了所以不用再量化 Conv"，而是
**这一遍只补 MatMul**：`quantize_codec_static.py` 每次只量化 `--ops` 指定的算子，
所以 ① 和 ③ 是两次独立调用，缺一不可。

**漏掉某一步的症状非常有欺骗性**：模型照样出声、听感几乎无差，只是慢。
缺 ④：codec 6.08 → 4.68 s（RTF 0.898 → 0.837）；连 ②③ 一起缺：7.08–7.49 s；
回到 FP32 原始模型还要再慢一倍以上（port notes 的 2.16 RTF 基线）。
复现者很容易把这些差异归因于"机器不同"。

`--voices` 是量化校准的输入（6 条不同的 codes 序列），所以**必须先有 #3**；
它来自 `prepare_android_voice.py`，不是模型包里的文件。

---

## 4. 从零到出声

`reproduce.sh` 把下面四段串起来，也可一步一步手动跑。

### 4.1 生成模型侧（5 分钟，CPU）

```sh
cd ~/Desktop/llama.cpp
P=~/m5-assets/pyenv/bin/python
O=~/m5-assets/android_onnx
W=~/m5-assets/model-package
S=~/Desktop/llama.cpp/examples/m5-tts

# tokenizer.bin + voices.json（参考音色：官方样例 + 其转写）
$P $S/prepare_android_voice.py ~/Desktop/M5-TTS "$O" \
   ~/Desktop/M5-TTS/assets/aslp_default_voice.wav \
   '猪携带该疾病，然后通过蚊子传染给人类。' "$W"

# 两个 gguf + 音频头/嵌入
PYTHONPATH=gguf-py $P $S/convert_backbone.py --quant Q8_0  "$O/backbone.onnx"     "$W/backbone-Q8_0.gguf"
PYTHONPATH=gguf-py $P $S/convert_backbone.py --depth --quant Q4_0 "$O/depth_frame.onnx" "$W/depth-Q4_0.gguf"
$P $S/pipe_depth.py --android "$O/depth_frame.onnx" "$W/depth-weights.f32"
```

`convert_backbone.py` 只支持 `Q8_0` / `Q4_0`——这就是**生产档位**。别的量化
（Q5_K_M / Q6_K / mix4）是 `llama-quantize` 的实验副产品，已判死（日志 §23：
体积 −48% 只换 −3%，因为瓶颈是 dispatch 延迟而非带宽）。

### 4.2 生成 codec（10 分钟，CPU）

见 §3。产物改名/放到 `$W/codec_decoder_cached.onnx` 即可。

### 4.3 生成二进制（30–60 分钟）

```sh
# 原生侧：m5-server / m5-engine / libm5engine.so
M5_ORT_ROOT=~/m5-assets/ort-aar examples/m5-tts/build_android.sh ~/m5-assets/build

# HTP stage：backbone-resident（== depth-resident）+ 5 个 .so
examples/m5-tts/docker_build_stage.sh ~/m5-assets/stage
```

- `M5_ORT_ROOT` 是解包后的 `onnxruntime-android.aar`（`headers/` + `jni/arm64-v8a/`），
  版本 **1.23.2** 与 App 的 gradle 依赖对齐。
- stage 必须用 Qualcomm 的容器镜像（`ghcr.io/snapdragon-toolchain/arm64-android:v0.8`，
  自带 Hexagon SDK + NDK）。Apple Silicon 上是 x86 模拟，**首次构建很慢**，之后增量几分钟。
- 二进制**不可逐字节复现**（工具链时间戳、`-static-libstdc++` 链接顺序），
  验它们是验行为（§5 的门禁），不是验哈希。

### 4.4 部署

```sh
bash examples/m5-tts/reproduce.sh deploy     # 推包 + 起服务 + 门禁
# 等价手工（注意 push 的是 model/ 的内容，不是 model 目录本身）：
#   adb shell 'mkdir -p /data/local/tmp/llama-m5-htp-v73/model'
#   adb push $W/model/. /data/local/tmp/llama-m5-htp-v73/model/
#   adb push $W/backbone-Q8_0.gguf $W/depth-Q4_0.gguf $W/codec_encoder.onnx \
#            /data/local/tmp/llama-m5-htp-v73/
#   M5_DEFAULT_VOICE='官方样例' ./examples/m5-tts/deploy.sh --no-models
```

`deploy.sh` 推二进制、ARM64 ORT 运行库和 web 控制台，模型由 `reproduce.sh deploy` 推送（它做的是反向备份：
把设备上的模型包 pull 到 `~/m5-assets/device-model-backup`）。直接调用 `deploy.sh` 时，先准备设备上的模型文件。`/data/local/tmp` 在设备重启后会被清空，重启后要重跑。

---

## 5. 验收门禁

复现"看起来成功"和"真的成功"差三个检查：

1. **文件级**：§2.1 的 5 个派生哈希全对（2 个 gguf、`depth-weights.f32`、
   `tokenizer.bin`、codec）→ 权重与改写链没问题。
2. **服务级**（`reproduce.sh gate` 检查 HTTP、PCM 格式和指标，并报告参考输出差异）：

   ```sh
   adb forward tcp:18080 tcp:8080
   curl -s http://127.0.0.1:18080/health          # {"status":"ok"}
   curl -s -D- -o /tmp/g.wav -H 'Content-Type: application/json' \
     -d '{"input":"今天的天气很好，我们一起去公园散步，看看路边的花，听听树上的鸟叫。","voice":"官方样例","response_format":"wav"}' \
     http://127.0.0.1:18080/v1/audio/speech
   ```

   本机基线：`X-M5-Frames: 94`、`X-M5-Elapsed-S ≈ 4.67`、wav
   `sha256 = f6a5f5f48b73e8d8f0e510405bac98d8d6313289faf513ab8b15c46b55a089d5`
   （seed 1234、threads 3、first 2 / chunk 4；`deploy.sh` 默认 `--voice serena`，
   所以不指定 `voice` 时走的是 serena）。
   换音色（serena）会得到 87 帧 / `10b0bdc1…`——**音色不同帧数自然不同**，
   这不是失败。
   博客的 69 帧导航样句用 `reproduce.sh benchmark` 复跑，流程见 [README](README.md#reproduce-the-android-server-and-the-69-frame-benchmark)。
3. **听感级**：抽 2–3 句人耳试听。逐字节相同表示复现了参考 WAV。试听另外检查参考本身的底噪、破音与韵律。

失败时的判定树：

| 症状 | 先查 |
|---|---|
| `/health` 通，请求 500 | `model/` 是否缺 #1–#5（`Engine::open` 的错误会打到 server stderr） |
| 能出声但 RTF > 1 | codec 是否漏了 §3 四步中的某一步（尤其 ④ Snake）；`--threads` 是否被改 |
| 帧数对、哈希不对 | voices/codes 来源变了（参考 wav 换了？转录不对？）；codec 版本变了 |
| 首包 > 1 s | 是否漏了 `deploy.sh` 的 warmup / prime 缓存（日志 §19–§20） |
| 服务起不来，`/dev/adsprpc-smd` 报 EACCES | 进程不在 shell 域（App uid 拿不到 HTP，见日志 §13） |

---

## 6. 已知坑（按踩坑代价排序）

1. **`/tmp` 会被清**：`clone_voice.py` 的 `--model` 默认值是
   `/tmp/m5-official/onnx`，机器重启后失效。**永远显式传 `--model`**。
2. **音色名 = 参考 wav 的文件名**（`prepare_android_voice.py` 用 `reference.stem`）。
   改名 = 换 key，`voices.json` 的哈希随之变；Web 端音色下拉也按这个名字。
3. **`codec_identity.json` 会挡人**：`prepare_android_voice.py` 要求
   `codec_encoder.onnx` 的 SHA256 等于该字段，否则直接抛错。改了 encoder 就要同步改它。
4. **Python 与 ORT 版本**：量化脚本对 ORT 1.18 的 `adjust_tensor_ranges` 做了最小
   放宽，1.23 上行为未验证——**量化用 `python3.12` + `onnxruntime==1.18.1`，
   设备运行时用 1.23.2**，两者不要混。3.13 装不上这两个版本（无 cp313 wheel）。
   另外国内镜像经常只放行首页、拦 `.whl`（403），装不上时加
   `PIP_INDEX_URL=https://pypi.org/simple`。
5. **ORT AAR 解包路径**：`build_android.sh` 找 `$M5_ORT_ROOT/headers/onnxruntime_c_api.h`
   与 `jni/arm64-v8a/libonnxruntime.so`；AAR 里还带 `libonnxruntime4j_jni.so`，不需要。
6. **`deploy.sh` 不推模型**（§4.4），也**不会**帮你建 `model/` 目录。
7. **stage 的 `depth-resident` 是 `backbone-resident` 的副本**（同一二进制按 argv 决定角色），
   不是两个程序。漏了 `cp` 就只有 backbone 能起。

---

## 7. 本机路径总表

| 路径 | 内容 | 体积 | 进 git？ |
|---|---|---:|---|
| `~/Desktop/M5-TTS` | 上游代码 checkout（含 patch + App 改动） | 1.1 GB | 否（独立仓库） |
| `~/m5-assets/android_onnx` | HF 模型包 | 1.3 GB | 否，**当前已清，可重下** |
| `~/m5-assets/model-package` | 本章生成的产物（`reproduce.sh` 输出） | ~1 GB | 否 |
| `~/m5-assets/model-package/release` | `reproduce.sh release` 打的 Release 附件 | ~370 MB | 否（走 Release） |
| `~/m5-assets/device-model-backup` | 设备模型包备份（`deploy.sh` 的目标） | 698 MB | 否 |
| `~/m5-assets/build` | `m5-server` / `m5-engine` / `libm5engine.so` | 81 MB | 否 |
| `~/m5-assets/stage` | `backbone-resident` + 5 个 .so | 62 MB | 否 |
| `~/m5-assets/ort-aar` | 解包的 ORT 1.23.2 AAR（`headers/` + `jni/`） | 77 MB | 否 |
| `~/m5-assets/pyenv` | 转换用 Python 环境（`reproduce.sh` 自建） | ~300 MB | 否 |
| 设备 `/data/local/tmp/llama-m5-htp-v73` | 运行时全量（含实验变体） | 9.4 GB | 否（重启即清） |

> 设备目录里另有一堆实验变体（`codec-qdq-*`、`backbone-Q5_K_M` 等），是日志
> §7–§30 的对照物，**不属于复现路径**，不需要重建。

---

## 8. 用 Release 分发派生产物（为什么仓库里没有二进制）

判据很简单：**HF 上有的不存、能一条命令重建的不存、只剩"必须生成且生成代价高"的才存**。
符合最后一条的只有两包，其余 9 个运行时文件都在 HF 原始包里。

### 8.1 各载体的硬上限（2026-11 核对）

| 载体 | 上限 | 对我们的意义 |
|---|---|---|
| git 普通文件 | >50 MiB 警告，**>100 MiB 直接拒收**（网页上传 25 MiB） | 单个 ONNX 就 400 MB+，进不去 |
| 仓库整体 | 建议 <1 GB，**强烈建议 <5 GB** | 当前 pack 486 MB，没有余量再塞 1.3 GB |
| git LFS | 单文件 2 GB；免费 **10 GiB 存储 + 10 GiB/月流量** | 放得下，但每 clone 一次 1.3 GB，一个月约 7 次就耗尽，之后 LFS 停用 |
| **Release 附件** | 单文件 ≤2 GB，**总量与带宽不限** | 唯一"放得下又不按流量付费"的地方 |

### 8.2 两个附件

| 附件 | 内容 | 未压缩 |
|---|---|---:|
| `m5-codec-int8-<tag>.tar.zst` | `codec_decoder_cached.onnx`（四步链产物） | 140,935,716 B |
| `m5-models-q4-<tag>.tar.zst` | `backbone-Q8_0.gguf`、`depth-Q4_0.gguf`、`depth-weights.f32`、`tokenizer.bin`、`voices.json` | ≈264 MB |

外加 `SHA256SUMS`（每行 `<sha256>  <name>  <bytes>  reference|drift`）和
`RELEASE_NOTES.md`（来源坐标、安装步骤、哈希表、许可提醒）。

`manifest.json` / `text_embeddings.npy` / `audio_embeddings.npy` / `codec_cached.json` /
`codec_encoder.onnx` **不进附件**——它们在 HF 包里原样就有，`fetch` 已经按哈希验过。

### 8.3 本地打包

```sh
./examples/m5-tts/reproduce.sh release                 # -> $OUT/release/
./examples/m5-tts/reproduce.sh release --tag m5-v1     # 指定标签
./examples/m5-tts/reproduce.sh release --upload        # 需要 gh 已登录
```

`release` 只读产物、不改模型；`--upload` 走 `gh release create`（已存在则 `upload --clobber`）。

### 8.4 CI（`.github/workflows/m5-package.yml`）

手动触发，四个输入：`tag`、`hf_revision`、`upstream_ref`、`upload`（默认 false，
即只产出 artifact 不建 Release）。流程与 §4.1–§4.2 完全同一条路径：

```
checkout -> 分支守卫 -> python 3.12 -> clone 上游 M5-TTS@<ref> -> cache HF 包
  -> fetch（验输入哈希）-> models -> codec（四步）-> package -> release
  -> 漂移检查 -> upload-artifact -> 可选 gh release
```

**CI 能证的**：输入包确实是记录在案的那个 revision；四步链跑完；codec 体积落在
140,935,716 B；每个产物的哈希要么等于参考值（`reference`），要么被标成 `drift`。
**CI 不能证的**：端到端的 codes/WAV 等价——那需要真机，只能 `reproduce.sh gate`
在设备上跑（判据见 §5）。

**运行入口**：在 GitHub Actions 的 `m5-tts model package` 页面从默认分支 `master` 运行。该分支包含脚本与 workflow。默认生成 CI artifact；开启 `upload` 会创建或更新公开 Release。

工作目录全在 `$GITHUB_WORKSPACE/.m5-work/` 下（`M5_SRC` / `ONNX_DIR` / `OUT` / `VENV`），
留在 workspace 内部是刻意的：`upload-artifact` 对 workspace 外的绝对路径行为不一致。
该目录已进 `.gitignore`，本地用 `act` 跑也不会脏树。CI 里多跑一步 `shellcheck`
（runner 预装），守着 `reproduce.sh` 与 `deploy.sh` 不进 warning；workflow 本身用
`actionlint .github/workflows/m5-package.yml` 校验（本地一条命令，能抓到
`runner` 上下文用错这类只在运行时才炸的错误）。

`models` 那步会把 `soundfile` / `scipy` / `tokenizers` 按需装进 venv：这三个不是
codec 链的依赖，而是 `prepare_android_voice.py` → 上游 `infer_onnx` 的 import 链带来
的（纯 codec 运行不会碰它们，所以没塞进 `PIP_PKGS` 的默认值）。


`drift` 不是失败：onnx/onnxruntime 换版本会重编号节点、重算校准区间，字节会变。
真正要拦的是"某一步静默没生效"——那个用体积指纹（456→362→282→141 MB）和
codec 下限 100 MB 的断言拦。**drift 的 codec 不要直接配旧的 `voices.json` codes**，
要重新跑 `prepare_android_voice.py` 编码音色。

### 8.5 为什么不随仓库发模型

- **许可**：`ASLP-lab/M5-TTS` 根目录没有 LICENSE，README 只说"按各 release 附带的条款"，
  模型卡也没有明确许可字段。再分发权属不清，这一步不该由脚本替用户决定。
- **权威源**：HF 有 revision 和下载计数，我们镜像一份只会产生第二个"正确版本"。
  `fetch` 核对已记录的上游文件 SHA256，比镜像更可靠。
- **成本**：克隆要 1.3 GB，而 99% 的读者只想要代码和结论。

真要做离线备份，用私有 HF 仓库（同一套 CLI），别拿 GitHub 当备份盘——
官方文档原话是 "Git is not designed to serve as a backup tool"。
