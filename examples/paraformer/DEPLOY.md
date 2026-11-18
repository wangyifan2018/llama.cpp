# 部署：OpenAI 兼容 ASR 服务

这是本项目的生产部署方式。前端（车机）按 OpenAI 音频接口调用，服务内部用 HTP 推理，
CPU 引擎与声纹/VAD 是可选件。

- 二进制：`llama-paraformer-api`
- 接口：`POST /v1/audio/transcriptions`、`GET /v1/models`，另有 `/health`、`/metrics`、
  `/v1/realtime`(WS) 与测试台 `/`
- 后端：**默认 HTP**（检测到 HTP 设备时自动选 `HTP0:0`，无则回落 CPU；`--device CPU` 可强制）
- 精度档：热词规则 + 文本规则 + 标点，**属于服务的一部分**，不是可选优化

## 1. 两种落地形态

| 形态 | 适用 | 组成 |
|---|---|---|
| **生产（有 root）** | 量产/整车 | `/vendor/etc/init/paraformer-asr.rc` + `/vendor/bin/paraformer-asr` + `/vendor/etc/paraformer/paraformer-asr.conf`，由 init 拉起与重启 |
| 测试（无 root） | 现役测试机 | `/data/local/tmp/llama.cpp` 下同一套 launcher + supervisor 循环，`tools/install_local.sh` 安装 |

两者跑的是**同一个 launcher**（`deploy/paraformer-asr.sh`），差别只在谁能提供 init 级重启。
launcher 负责：读配置、设环境（`LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH`/`GGML_HEXAGON_*`）、
拼参数、崩溃退避（连续快速失败指数退避，最长 60 s），最后 `exec` 二进制，让 SIGTERM 直接进入
服务的优雅退出（先排空已接受请求再关监听）。

### 生产安装

```sh
# 1) 打包（不接触设备，只组装 /vendor 树 + manifest + 安装脚本）
PF_MODELS=$HOME/models/paraformer examples/paraformer/tools/package_device.sh /tmp/pf-pkg

# 2) 设备（需 root 与可写 /vendor）
adb root && adb remount
adb push /tmp/pf-pkg/system/vendor /vendor
adb shell restorecon -R /vendor/bin /vendor/lib64 /vendor/etc/paraformer
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

- 单个 HTP 会话**按设备独占**（v73）：服务运行时再起 CLI 会挂在初始化之后，不会报错；
  要跑 `reproduce.sh device` 先停服务。
- 设备整机只剩 ~1 GB 时，第二个 HTP 实例会被低内存回收（日志末尾是 `Killed`）。
  一个实例两引擎约 426 MB RSS，HTP 会话另有 3.1 GB vmem 预留。

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
客户端只推 `input_audio_buffer.append`，服务端 VAD 判定声学停顿后，每次发一对
`input_audio_buffer.committed` + `...transcription.completed`。结果延迟 = 停顿阈值
（默认 500 ms）+ 推理（约 280 ms）。段落音频含尾部静音，因此不会出现客户端切太早导致的末尾丢字。
停顿时长可配（`silence_duration_ms`），短于阈值的停顿不断句。

**b) 客户端自己切段**：`append` 到切点后 `commit`，服务端返回该段终稿。适合前端 VAD 已经可靠的场景。

预览（`x.pf.partial`）**默认关闭**（`--partials` 才开）：它是整段假设的刷新，与断句结果无关；
座舱需要的是"一段一个结果"，不是边说话边改字。启用时每次发完整假设，客户端整行替换，
只有在假设扩展时才额外发 append 式 `delta`。

终稿事件带 `speaker_score` / `speaker_ok`（启用声纹时）。
