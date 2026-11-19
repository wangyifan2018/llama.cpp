# 部署：OpenAI 兼容 ASR 服务

这是本项目的生产部署方式。前端（车机）按 OpenAI 音频接口调用，服务内部用 HTP 推理，
CPU 引擎与声纹/VAD 是可选件。

- 二进制：`llama-paraformer-api`
- 接口：`POST /v1/audio/transcriptions`、`GET /v1/models`，另有 `/health`、`/metrics`、
  `/v1/realtime`(WS) 与测试台 `/`
- 后端：**HTP 是默认且强制的后端**。启动脚本显式传 `--device HTP0:0`（`PF_DEVICE`），
  `--prefer-idle-htp` 让单路请求始终落在 HTP（CPU worker 只作第二路溢出）；
  **没有 HTP 引擎时服务以退出码 3 拒绝启动**，除非显式 `--allow-cpu-fallback`（或
  `PF_ALLOW_CPU_FALLBACK=1`），此时 `/health` 报 `status: degraded`、`/metrics` 报
  `cpu_fallback: true`。这样"悄悄跑在 CPU 上"不会再发生。`/health` 的 `htp_engines` 与启动日志 `N of M engines on HTP`
  可直接核对。`PF_DEVICE=auto` 保留二进制探测，供宿主使用。
- 精度档：热词规则 + 文本规则 + 标点，**属于服务的一部分**，不是可选优化

## 0. 冻结的默认方案（v1，2026-11-19）

以下十条是**生产默认**，改动需要重新过 §3 的验收并更新本节。依据都是本文件与
`PRODUCTION.md` 里记录的实测。

| # | 决定 | 依据 |
|---|---|---|
| 1 | 对外形态：OpenAI 音频接口（`/v1/audio/transcriptions`、`/v1/models`） | 调用方零学习成本；`response_format` 覆盖 json/text/verbose_json/srt |
| 2 | 后端：**HTP 强制**（`PF_DEVICE=HTP0:0`），无 HTP 引擎拒绝启动（退出码 3） | 曾出现"悄悄跑在 CPU 上"，单请求 512 ms 对 264–289 ms |
| 3 | 调度：`--prefer-idle-htp`，CPU worker 只作第二路溢出 | 共享队列让单路每隔一次落 CPU，P95 抖动 |
| 4 | 引擎数 2、线程 4（`PF_ENGINES=2 PF_THREADS=4`） | 4 线程比 2 线程快 35 ms，RSS 只差 1 MB |
| 5 | 实时：**服务端 VAD 断句**为默认，`silence_duration_ms=300` | 客户端自切端点会丢末字（16 条里 15 条）；250 ms 会把一句切成两段（11 段 vs 8 段） |
| 6 | `committed` 表示已接收，每个 `item_id` 对应一次 `completed` 或终态 `error` | 自动断句与显式提交使用同一队列、截止时间和编号 |
| 7 | 段尾静音裁到 150 ms 再推理（不 padding、不补齐整块） | 静音纯解码开销（约 7 ms/60 ms）；补齐不提升精度 |
| 8 | 精度档随服务部署：`PF_HOTWORDS` + `PF_TEXT_RULES` + 标点 | 同接口无档 65.6%，带档 87.5% |
| 9 | 声纹：**默认关闭**，只上报 `speaker_score/ok`，不拦截 | TTS 声纹不足以决定拦截；开启后 1–2 路无尾延迟代价（见 §4.2） |
| 10 | 进程管理：`/vendor` + init.rc（root）或 `/data/local/tmp` + supervisor（无 root） | 同一 launcher；`kill -9` 后约 10 s 恢复 |

明确不在本方案内（避免误读）：

- 真实座舱录音的精度结论。所有数字来自 TTS 夹具，换音色 CER 可从 0.6% 掉到 34%。
- 英文专有名词的模型级修复。当前靠 `cabin_text_rules.tsv` 的字面重写覆盖已知模板/店名。
- 逐帧文本预览（`--partials`）。默认关闭，它不是断句结果。
- 声纹拦截、说话人分离、单通道降噪（见 `denoise-options.md`）。

## 1. 两种落地形态

| 形态 | 适用 | 组成 |
|---|---|---|
| **生产（有 root）** | 量产/整车 | `/vendor/etc/init/paraformer-asr.rc` + `/vendor/bin/paraformer-asr` + `/vendor/etc/paraformer/paraformer-asr.conf`，由 init 拉起与重启 |
| 测试（无 root） | 现役测试机 | `/data/local/tmp/llama.cpp` 下同一套 launcher + supervisor 循环，`tools/install_local.sh` 安装 |

两者跑的是**同一个 launcher**（`deploy/paraformer-asr.sh`），差别只在谁能提供 init 级重启。
launcher 负责读配置、设环境、设置参数和连续快速重启退避，最后 `exec` 二进制。用 SIGTERM 启动服务的协作排空。整车镜像需要验证其 init 停止时限、SELinux 域和 DSP 权限。 [Android init 语义](https://android.googlesource.com/platform/system/core/+/refs/heads/main/init/README.md)

### 生产安装

```sh
# 1) 打包（不接触设备，只组装 /vendor 树 + manifest + 安装脚本）
PF_MODELS=$HOME/models/paraformer PF_VAD_FILE=$HOME/models/paraformer/silero_vad.onnx \
    examples/paraformer/tools/package_device.sh /tmp/pf-pkg

# 2) 设备（需 root 与可写 /vendor）
adb root && adb remount
adb push /tmp/pf-pkg/system/vendor/. /vendor/
adb shell restorecon -R /vendor/bin/paraformer-asr /vendor/bin/llama-paraformer-api \
    /vendor/lib64/paraformer /vendor/etc/paraformer /vendor/etc/init/paraformer-asr.rc
adb shell start paraformer_asr        # 或重启，rc 里 class main 会带起
adb shell 'cat /data/vendor/paraformer/paraformer-asr.log'
```

init 侧要点：非 oneshot（退出即重启）、`shutdown critical`（开机冷车时不拖垮启动）、
`user system` + `group system readproc`（`/dev/adsprpc-smd` 的现有属主）、
`on property:vendor.paraformer.restart=1` 供免 root 重启。

### 无 root 测试机安装

```sh
examples/paraformer/tools/install_local.sh                 # 只算 ASR，省内存
examples/paraformer/tools/install_local.sh --with-capabilities   # 加 VAD + 声纹
```

它做三件事：推 launcher/配置/控制台到 `/data/local/tmp/llama.cpp`；用
`deploy/paraformer-asr.local.conf`（路径全在 `/data/local/tmp`）启动；用
`setsid + while` 保活。实测 `kill -9` 后约 10 s 自动恢复（新 pid），`/health` 200。

## 2. 配置项（`paraformer-asr.conf`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `PF_BIN` / `PF_MODEL` / `PF_LIBDIR` | 按布局 | 二进制、权重、库目录 |
| `PF_LIBDIR_EXTRA` | 空 | 库分目录时追加（例如 sherpa/onnxruntime 单独一目录） |
| `PF_PORT` / `PF_BIND` | 8080 / 127.0.0.1 | 绑定回环，经 adb forward 或反向代理暴露 |
| `PF_ENGINES` | 2 | HTP + CPU 两引擎；无 root 测试机内存紧张时用 1 |
| `PF_THREADS` | **4** | 见 §4，2 线程会慢 35 ms |
| `PF_HOTWORDS` | 无 | **必须部署**，否则命令字段从 87.5% 掉到 65.6% |
| `PF_CONTEXT` / `PF_CONTEXT_MAX_LOGIT_GAP` | 空 | 实验短语表和声学 logit 差值限制，见 [Realtime 热词评测](HOTWORDS.md) |
| `PF_REALTIME_PUNCTUATE` | 0 | 设为 1，使用声学停顿给 Realtime 终稿添加分句标点 |
| `PF_TEXT_RULES` | 空 | 英文模板/专名重写（`never get to` → `Navigate to`） |
| `PF_VAD_MODEL` / `PF_SPEAKER_MODEL` + `PF_ENROLL_DIR` | 空 | 可选，各约 +60 MB |
| `PF_WEB_DIR` | 空 | 测试台目录；生产可不部署 |
| `PF_WARMUP` | 空 | 预热音频，缺失时服务回落 1.2 s 静音预热 |

## 3. 验收

```sh
python3 examples/paraformer/tools/smoke_production.py --port 8899 --wav <clip>   # 契约
python3 examples/paraformer/tools/eval_api.py --port 8899 --wav-dir <fixtures> \
        --out /tmp/api.tsv --punctuate                                          # 精度
python3 examples/x-asr/tools/eval_vehicle.py --pack tingting=<manifest> --run api=/tmp/api.tsv
```

SM6650 实测（HTP+CPU，2 引擎，4 线程，热词+文本规则+标点）：命令字段 **87.5%**，
device/state/value 90.5/90.5/100%，zh CER 0.60%，en WER 6.87%，wrong=0。

## 4. 延迟与资源（SM6650 实测）

| 项 | 值 |
|---|---|
| 单请求 wall p50 | **289 ms**（infer 277 ms，HTTP/WAV 约 12 ms） |
| 双路并发 wall p50 / p95 | 288 / 300 ms |
| 2 线程（旧默认） | infer **312 ms** |
| 4 线程（现默认） | infer **277 ms**，RSS 426 MB |
| 2 线程 RSS | 425 MB（**差约 1 MB**） |

四线程比两线程快 35 ms，内存只多约 1 MB：主机侧的 fbank、CIF、logits 回读是线程并行的，
两线程把它们卡住了。**这不是"用内存换速度"**，是之前没有把线程数配对。

其它资源事实：

- v73 的第二实例表现取决于固件和系统资源。这个 SM6650 固件能在隔离端口启动第二个实例，旧记录也有第二实例预热卡住的情况。复现脚本默认预留独立测试时段，已有服务可直接用 `accept.sh` 做回放。
- 设备整机只剩 ~1 GB 时，第二个 HTP 实例会被低内存回收（日志末尾是 `Killed`）。
  一个实例两引擎约 426 MB RSS，HTP 会话另有 3.1 GB vmem 预留。

## 4.2 声纹对尾延迟的影响（2026-11-19 实测）

同一条 2.3 s 命令、同一部署（HTP 优先、4 线程），只切 `PF_SPEAKER_MODEL`/`PF_ENROLL_DIR`：

| 配置 | 1 路 wall p50 / p95 | 2 路 wall p50 / p95 |
|---|---|---|
| 关闭声纹 | 273 / 307 ms | 289 / 378 ms（该轮有一次 453 ms 抖动） |
| **开启声纹** | **278 / 292 ms** | **291 / 313 ms** |

结论：**1–2 路没有可见代价**。原因是 `--prefer-idle-htp` 让 ASR 始终落在 HTP，声纹的
embedding（约 80 ms，只取前 1.5 s）在 CPU 上并行完成，被 280 ms 的推理覆盖。
开启后司机 0.921、乘客 0.336，判定正确。

3–4 路并发未在本轮重测；历史上（未启用 HTP 优先、2 线程）4 路 P95 从 700 升到 907 ms，
那条数字对应的是"CPU 引擎也在跑 ASR"的旧调度，不应直接套用到当前配置。

## 4.1 客户端自己已经有 VAD 时怎么接

如果前端已经有 VAD，它交给服务端的段落往往**末尾几乎没有静音**。这不是格式问题，是精度问题，
先看实测（同一批 16 条中文命令，设备 HTP）：

| 交给服务端的音频 | 与完整音频一致的条数 |
|---|---:|
| 完整（含尾部） | 16/16 |
| 末尾切掉 400 ms | **1/16** |
| 切 400 ms 后再补 300 ms 静音 | 0/16（补静音救不回来） |

切掉的部分是**真实语音**（这些 TTS 夹具本身末尾就是语音，尾部 RMS 0.07–0.11），所以丢的不是静音，
是最后一个字：`关闭主驾座椅加热` → `关闭主驾座椅将`，`调到一档` → `调到一`。

三种接法，按推荐顺序：

1. **让服务端断句（推荐）**：客户端不做 VAD，连续 `append`，服务端 `server_vad` 在声学停顿处断句。
   段落由检测器给出，天然带尾部静音，不会截断。代价是结果晚一个停顿时长（默认 500 ms）。
2. **客户端 VAD + 尾音（hangover）**：客户端仍自己断句，但在端点之后**多送 300–500 ms** 再 `commit`。
   多出来的静音几乎不增加推理时间（CIF 不会为空帧发字），却能保住最后一个字。
   这是前端 VAD 已经成熟、只想改一行的最小改动。
3. **客户端紧切 + 服务端标记**：保持现状，用响应头 `X-Tail` 识别问题段落：
   `hot` 表示这段以语音结尾（很可能被切掉尾巴），`silence` 表示结尾有静音。
   生产上可以统计 `hot` 比例，作为前端 VAD 尾音配置是否够的指标。

`X-Tail` 也出现在 `/v1/audio/transcriptions` 的每次响应里（`verbose_json` 另有同名字段），
判定窗口是最后 150 ms，RMS 阈值 0.02。

注意：服务端的 `--vad-model` 批量路径（>18 s 才切）**不能**修复紧切的段落——丢失的音频已经不在请求里了。

## 4.3 从零复现当前效果

前提：一台 SM6650 设备、`adb` 可见、`docker` 可用（交叉编译）、`PF_MODELS` 里放着
`tools/fetch_models.py` 校验过的权重。

```sh
export PF_MODELS=$HOME/models/paraformer
python3 examples/paraformer/tools/fetch_models.py --out "$PF_MODELS"   # 首次：取权重并校验 sha256
export PF_VAD_FILE=$PF_MODELS/silero_vad.onnx
export PF_SHERPA_SDK=/tmp/sherpa   # inc/c-api.h、lib/libonnxruntime.so、lib/libsherpa-onnx-c-api.so
examples/paraformer/tools/reproduce_deployment.sh                     # 构建、打包、验哈希、隔离回放
```

`reproduce_deployment.sh` 会输出独立运行目录，包含包、源版本、工具链摘要和验收原始结果。已有 Paraformer HTP 进程时，复测脚本退出并保留包。可用 `PF_PACKAGE_ONLY=1` 在主机只构建与打包。`accept.sh` 可直接验收已有服务，校验 32 条夹具哈希并保存 REST 1/2/4 路和 WS 单路、四路同步、错峰、长短混合结果。28/32 和零方向反转为 Tingting 夹具回归门槛。延迟在报告中记录 P95/P99，整车 SLO 由目标环境单独验收。

中文单路候选用 `PF_REALTIME_PUNCTUATE=1 PF_REPLAY_SINGLE=1` 复现。结果还包括手动提交、300 ms、180 ms 和 120 ms 四个档位的 Realtime 逐请求记录。 [单路中文对照与门槛](HOTWORDS.md#单路中文-realtime-候选)

| 项 | 期望 |
|---|---|
| 契约 | 普通烟测与自动断句、显式 flush、切换模式和截止时间检查通过 |
| 命令字段 | 87.5%（device 90.5%，value 100%，zh CER 0.60%，en WER 6.87%） |
| 1 路 | wall p50/p95 ≈ 280/287 ms |
| 2 路 | wall p50/p95 ≈ 292/306 ms |
| `/health` | `status ok`、`htp_engines ≥ 1`、`server_vad true`、`speaker_gate false` |

数字对不上时先看 §5 排障，再确认 `PF_MODELS` 里的权重 sha256 与 §2 的清单一致。

## 5. 排障

| 现象 | 检查 |
|---|---|
| 起不来，日志 `CANNOT LINK EXECUTABLE` | `PF_LIBDIR`/`PF_LIBDIR_EXTRA` 是否含 `libsherpa-onnx-c-api.so` 所在目录 |
| 端口无响应但进程在 | 是否绑到 127.0.0.1 且没建 adb forward；或有别的实例占着 HTP 会话 |
| 命令字段异常低 | `PF_HOTWORDS` 没部署（看启动日志的 `pf: N text rules`） |
| 进程被 `Killed` | 内存不足；降 `PF_ENGINES=1` 或关掉 VAD/声纹 |
| 只出第一段预览就没更新 | 旧版本只在 2.0–2.0 s 窗口内出预览，见 §6；现版本 0.8–6 s 且发送完整假设 |

## 6. 实时接口行为（`/v1/realtime`）

两种用法，**推荐第一种**：

**a) 连续流 + 服务端断句**（`session.update` 带 `turn_detection.type = "server_vad"`）：
客户端只推 `input_audio_buffer.append`，服务端 VAD 判定声学停顿后，入队即返回 `input_audio_buffer.committed`。客户端按 `item_id` 匹配一次终态和相应的计时。结果延迟包含停顿阈值
（默认 **300 ms**，`--vad-silence-ms` 或 `silence_duration_ms` 可配）+ 推理（约 280 ms）。
段落音频含尾部静音，因此不会出现客户端切太早导致的末尾丢字。

阈值与真实停顿的对应关系（实测，数字静音间隔）：250 ms 不断句，**400 ms 断句**，600 ms 断句。
检测器自身有窗口平滑，所以"配 300 ms" 的实际分界落在 250–400 ms 之间。

**不要低于 300 ms**：23 s 连续命令片段实测，阈值 250 ms 会切出 **11 段**（把"别关氛围灯/把顶灯关掉"
从句中切开），300 ms 得到 **8 段**、正好一句一段。想更快只能牺牲"一句一段"。

无停顿的长音频由检测器在 18 s 处强制断开；段尾多余的静音在送模型前会被裁到 150 ms
（静音只是纯解码开销，约 7 ms 每 60 ms）。

会话结束时（`commit` 或断开）剩余音频**直接推理、不补齐**：图是按精确形状构建的，
补齐只会多花算力并扰动 CIF 统计。

**b) 客户端自己切段**：`append` 到切点后 `commit`，服务端返回该段终稿。适合前端 VAD 已经可靠的场景。

预览（`x.pf.partial`）**默认关闭**（`--partials` 才开）：它是整段假设的刷新，与断句结果无关；
座舱需要的是"一段一个结果"，不是边说话边改字。启用时每次发完整假设，客户端整行替换，
只有在假设扩展时才额外发 append 式 `delta`。

终稿事件带 `speaker_score` / `speaker_ok`（启用声纹时）。
