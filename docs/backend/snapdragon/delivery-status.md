# 交付状态与可复现资产（2026-11-18）

这份文件回答三个问题：**现在能跑到什么程度**、**每个数字用什么跑的**、**怎么从零复现当前最好的版本**。
每批优化后更新本文件；历史实验的细节在各自的专题文档里。

## 1. 当前最好版本

| 项 | 值 |
|---|---|
| 源码 | 本仓库，`git rev-parse --short HEAD` 见文件末尾 |
| ASR 权重 | `paraformer-zh-q8_0.gguf`，226 MB，sha256 `e73232ecf7e48490…`（ModelScope/HF: `cstr/paraformer-zh-GGUF`） |
| 服务二进制 | `llama-paraformer-api`（OpenAI REST + Realtime WS + 测试台） |
| 部署目录 | `/data/local/tmp/llama.cpp`（bin/lib/gguf/web/夹具） |
| 设备 | SM6650（Hexagon v73，2 HVX，无 HMX，2 MB VTCM），Android 16 |

**指标（真机 HTP+CPU 两引擎，2.3–2.7 s 单段命令）**：

| 指标 | 值 | 说明 |
|---|---|---|
| endpoint→final（单段，warm） | **277–330 ms** | 客户端 commit 到 completed |
| 其中 encoder | 241 ms | 占 81%，compute-bound |
| decoder | 42 ms | 16 blocks |
| CIF（host） | 9 ms | 可上图优化，未做 |
| 4 路齐发 p50/p95 | 447/626 ms | 三引擎；两引擎 494/705 |
| zh CER（32 条车控夹具） | 0.60% | Tingting |
| en WER | 6.87% | 热词 + span rewrite |
| 命令字段 | **84.4%** | 热词 + span rewrite + 停顿标点（API `punctuate=true`） |
| device / state / value 字段 | 90.5% / 88.1% / **100%** | 同上，32 条夹具 |
| 内存（单进程） | 252 MB | Q8_0；Q4_K 146 MB |

## 1.1 本批修掉的一个质量 bug

span rewrite（`footwall→footwell` 这类跨 token 数的纠正）原先只在**不请求时间戳**时生效
（代码里 `if (!hotword_repl.empty() && !out_tokens)`）。而 `verbose_json`、`punctuate=true`、
`srt` 都要 token 时间，于是**产品实际走的路径一直丢掉了这条纠正**。

改成"先重写、再从重写后的序列生成 token 记录（时间取该位置的时间）"之后：

| 口径 | 修复前 | 修复后 |
|---|---:|---:|
| 命令字段（API punctuate） | 78.1% | **84.4%** |
| device / state / value | 85.7% / 83.3% / 94.4% | **90.5% / 88.1% / 100%** |
| en WER | 8.40% | **6.87%** |
| zh CER | 0.60% | 0.60% |

## 2. 资产清单

| 资产 | 路径 | 大小 | 哈希（前 16 位） | 来源 |
|---|---|---:|---|---|
| ASR 权重 Q8_0 | `~/Library/Caches/qwen3-asr-htp/paraformer-models/paraformer-zh-q8_0.gguf` | 226 MB | `e73232ecf7e48490` | cstr/paraformer-zh-GGUF |
| ASR 权重 Q4_K | 同目录 `paraformer-zh-q4_k.gguf` | 123 MB | `c3dc70409857d745` | 本仓库 `llama-paraformer-requant` 离线再量化 |
| 说话人 CAM++（torch） | `~/Library/Caches/qwen3-asr-htp/speaker-models/campplus_cn_common.bin` | 27 MB | `3388cf5fd3493c9a` | ModelScope `iic/speech_campplus_sv_zh-cn_16k-common` |
| 说话人 CAM++（ONNX，定长 2 s） | 同目录 `campplus_t200.onnx` | 26.7 MB | `36c56a1b2c3d83b6` | `tools/export_campplus_onnx.py` 导出（parity 1.000000） |
| 车控夹具 | `…/paraformer-models/tingting/` | 32 条 | — | `tools/make_eval_pack.sh` + Tingting/Samantha |
| 独立音色夹具 | `…/paraformer-models/voice2/` | 32 条 | — | Flo + Daniel |
| SeACo 权重（研究用） | `…/seaco/model.pt` | 944 MB | — | ModelScope，未移植 |

## 3. 复现步骤

```sh
# 1) 宿主构建
cmake -S . -B build-x-asr -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-x-asr --target llama-paraformer llama-paraformer-api llama-paraformer-requant -j

# 2) Android 构建（Docker，与仓库既有流程一致）
docker run --rm --volume "$PWD:/workspace" --workdir /workspace --platform linux/amd64 -u 501:20 \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.8 \
  bash -c 'cmake --build /workspace/build-android-xasr --target llama-paraformer-api -j 6'

# 3) 部署
adb push build-android-xasr/bin/llama-paraformer-api /data/local/tmp/llama.cpp/bin/
adb push examples/paraformer/web /data/local/tmp/llama.cpp/
adb push ~/Library/Caches/qwen3-asr-htp/paraformer-models/paraformer-zh-q8_0.gguf /data/local/tmp/llama.cpp/

# 4) 启动（HTP 引擎 + CPU 引擎；测试台挂在 /）
adb shell 'cd /data/local/tmp/llama.cpp && \
  export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib \
         GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=1 && \
  nohup bin/llama-paraformer-api paraformer-zh-q8_0.gguf --device HTP0:0 \
    --port 8899 --engines 2 --threads 2 --web-dir web --warmup pf-api-warm.wav > pf-api.log 2>&1'
adb forward tcp:8899 tcp:8899

# 5) 打开测试台 http://127.0.0.1:8899/ ，或命令行验证
python3 examples/paraformer/tools/ws_client.py --url ws://127.0.0.1:8899/v1/realtime \
  --wav ~/Library/Caches/qwen3-asr-htp/paraformer-models/tingting/hvac_zh01.wav --paced
```

## 4. 测试台（`web/index.html`）

- 浏览器打开 `http://<host>:8899/`，点「开始录音」：麦克风 → 16 kHz 单声道 → 20 ms 分片 →
  WebSocket `input_audio_buffer.append`；画布实时滚动显示输入波形，下面是电平条。
- 「停止并提交」= `input_audio_buffer.commit`，页面显示 endpoint→final、服务端 queue/infer、
  语言标签与分段数；partial 文本实时出现在「识别结果」上方。
- 没有麦克风时用「用 wav 文件回放」：按 20 ms 实时节奏推同一段 wav，走的还是同一条 WS 路径。
- 服务端实测：`GET /` 返回 200（12 KB HTML），`ws_client` 在同一实例上跑通
  （设备上 endpoint→final 328 ms，queue 1 ms，infer 305 ms）。

## 5. 关键脚本

| 脚本 | 用途 |
|---|---|
| `tools/ws_client.py` | Realtime 协议客户端（20 ms 分片 / 整段），打印 delta、completed、服务端计时 |
| `tools/pause_punct.py` | 用 CIF 停顿插标点（命令字段 71.9%→78.1%） |
| `tools/export_campplus_onnx.py` | CAM++ → ONNX（27 MB，动态 T） |
| `tools/spk_gate_eval.py` | 声纹门控评估（FRR/FAR，本批实测 0/0，余量 0.46） |
| `tools/requant.cpp` | 离线再量化（Q8_0→Q4_K/Q5_0/Q6_K，保持未量化张量原类型） |
| `tools/lid_prototype.py` | 语言/说话人可分性探针（跨音色 LID 56%，说话人 98.4%） |
| `tools/mix_snr.py`、`tools/denoise_wiener.py` | 噪声混合与降噪参考实现（用于否决单通道降噪） |
| `tools/bench_pf_service.py`、`tools/profile_ops.py` | 服务并发压测、HTP 算子 profile |

## 6. 已知边界（不要误用）

1. 质量数字全部来自**合成音色**（Tingting/Samantha/Flo/Daniel）；换音色 CER 可从 0.6% 掉到 34%。
2. Realtime 的 partial 是"整段重解码"，间隔 ≥600 ms；不是流式模型。
3. 英文 OOV 名字的分词无法与模型对齐（GGUF 无 merge 表），需要 `a+b+c` 显式 token。
4. CAM++ 只导出**定长 2 s** 图（动态轴在部分长度上数值错误，实测 T=100 余弦 0.571、T=300 0.940）；
   音频按 2 s 分块、尾部均值补齐后平均 embedding，门控余量 0.25（本人最小 0.793 / 冒充最大 0.541）。
5. 设备端尚未集成声纹（缺 `onnxruntime_c_api.h`），门控目前只在本机 Python 侧验证。

## 7. 下一批（按优先级）

1. 把门控接到设备：补 `onnxruntime_c_api.h`（或用 sherpa-onnx 封装），在 API 里加 `--speaker-gate`，
   用 CPU 与 HTP 的 ASR 并行；同时把实测的 FRR/FAR 写进服务指标。
2. 测试台补"实时吞吐"面板（每段毫秒数分布、连续 N 段的 p50/p95），用于回归。
3. CIF 上 DSP（省 ~8 ms）；decoder/encoder 已无更多 runtime 杠杆。
4. 真实录音采集与评测（换掉 TTS 夹具）。
