# SM6650 两路 continuous batching 实测

2026-11-19，HONOR MTN-AN80，Android 16，SoC SM6650，Hexagon v73，2 HVX、0 HMX、2 MiB VTCM。实验基于 `155bfe7c7` 和本 example 的改动。两档均使用 [README](README.md) 中相同的 Q8_0 权重、HTP0:0 和上下文配置。设备上既有 Paraformer 服务保持运行。每个场景和档位各执行 3 轮，以下时间以秒为单位。

## 总耗时和 HTP decode

| 请求轨迹 | 总耗时中位数 串行 / 双路 | 总耗时降幅 | 吞吐提升 | Decode 累计中位数 串行 / 双路 | Decode 调用数 串行 / 双路 |
|---|---:|---:|---:|---:|---:|
| 同时到达，中 + 英 | 6.166 / 5.443 | 11.7% | 13.3% | 2.298 / 1.601 | 22 / 13 |
| 相隔 300 ms，中 + 德 | 6.395 / 5.622 | 12.1% | 13.7% | 2.502 / 1.796 | 24 / 15 |
| 长中文 + 英文，德语排队补位 | 15.194 / 13.377 | 12.0% | 13.6% | 5.335 / 3.634 | 50 / 27 |

Decode 计时覆盖提交与同步。两条短音频的 prefill 合计约 3.68 s，含长音频时约 9.4 s。这部分两档基本一致。两路档的真实配对 decode 在每轮分别有 9、9、23 次。21 个串行/双路配对样本全部在 EOG 停止，逐 token 一致。样本用于判断调度一致性，语料规模只覆盖这三条短命令及一条拼接语音。

## 从请求到达到终稿

| 请求轨迹 | P50 串行 / 双路 | P95 串行 / 双路 | P99 串行 / 双路 | 首条请求中位数 串行 / 双路 |
|---|---:|---:|---:|---:|
| 同时到达 | 4.538 / 5.245 | 6.174 / 5.459 | 6.177 / 5.463 | 2.944 / 5.087 |
| 相隔 300 ms | 4.319 / 5.181 | 6.106 / 5.365 | 6.109 / 5.377 | 2.957 / 5.046 |
| 第三条补位 | 11.739 / 12.312 | 14.897 / 13.085 | 14.898 / 13.089 | 8.545 / 13.032 |

各场景的 P95/P99 只来自 6 或 9 条样本，接近样本最大值。当前调度会在活跃序列两次 decoder 调用之间执行新请求的整段 prefill，因此首条终稿推迟，即使这一组请求更早全部完成。后续优先分别测音频前端、encoder、decoder prefill 的阻塞时间，在可中断的边界上限制一次追加的工作，再用首条和最慢请求的延迟做 A/B。输入应扩展到真实车内录音和更长的到达序列。

## 实际执行证据

`refill.2.1.jsonl` 中，请求 1 在 8,825.30 ms 完成。请求 2 在 8,825.42 ms 复用 slot 1，从 10,641.29 ms 开始与仍在 slot 0 的请求 0 配对。日志里的 `paired_decode` 在同一次两行 decoder 调用成功后记录。

单独执行的 `profile.log` 中，HTP0:0 执行 9 次第一层 `MUL_MAT_NX`，对应 `attn_norm-0` 的 `1024:2` 输入。输出投影实际执行 `1024:2 -> 151936:2` 的 `MUL_MAT`。音频侧 `IM2COL` 和 `a.conv_out.weight` 的 `MUL_MAT` 也在 HTP 上执行。CPU 日志包含生成 log-Mel 特征的图，greedy sampling 由主机完成。下面两条命令可复核 decoder 的实际执行形状。

```sh
rg 'HTP0:0 profile-op MUL_MAT_NX\|.*attn_norm-0.*x 1024:2 ->' profile.log
rg 'HTP0:0 profile-op MUL_MAT\|output\.weight.*x 1024:2 ->' profile.log
```

## 留档与复算

`$HOME/Library/Caches/qwen3-asr-htp/qwen3-asr-0.6b-batch2-sm6650-2026-11-19.tar.gz` 包含完整 WAV、trace、主机 smoke JSONL、设备 18 个计时进程的 JSONL 和 stderr、单独的 HTP profile、设备哈希验证及 `summary.json`。归档 SHA-256 为 `cd1df19d4890d251bf43eb5375d9ccf458f7eb40d6cb301006143938fe55a67e`。解压后可将 `fixtures` 直接交给 Android runner，或重算结果。

```sh
python3 examples/qwen3-asr-0.6b/compare.py /path/to/extracted/device-final 3
```
## 第二批：调度策略与首个终稿延迟（2026-11-19）

基线是第一批的 blocking 调度（两路档，同一二进制/权重/夹具）。本批加了三件事：把一次请求的
prefill 拆成可计时的三段、把“两次 decode 之间算谁的”变成可选策略、用同一 trace 做四档对照。

### 分阶段分解（每请求中位数，SM6650）

| 轨迹 | 策略 | log-Mel | 音频 encoder | decoder prefill | prefill 合计 |
|---|---|---:|---:|---:|---:|
| simultaneous | blocking | 7 | 789 | 1044 | 1840 |
| staggered | blocking | 6 | 792 | 1039 | 1837 |
| refill | blocking | 8 | 800 | 1045 | 1835 |
| simultaneous | idle | 6 | 789 | 1044 | 1842 |

结论：**log-Mel 可以忽略（约 7 ms）**，一次请求的阻塞几乎都由音频 encoder（约 790 ms，占 43%）
和 decoder prefill（约 1040 ms，占 57%）构成。音频 encoder 在 mtmd 里是**整块**的：一个 chunk
交进去，出来才返回，所以它是当前不可切分的地板。
### 四档对照（2 轮，capacity 2，串行对照做 token 校验）

| 轨迹 | 策略 | 总耗时 ms | 首条终稿 ms | P50 | P95 | P99 | decode 次数 | 配对次数 | token 一致 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| simultaneous | blocking | 5408 | 5055 | 5235 | 5417 | 5420 | 13 | 9 | ok |
| simultaneous | decode-first | 5417 (+8 / +10) | 5065 | 5244 | 5419 | 5420 | 13 | 9 | ok |
| simultaneous | interleaved | 5681 (+273 / −69) | 4986 | 5332 | 5696 | 5701 | 16 | 6 | ok |
| simultaneous | **idle** | 6162 (+754 / **−2087**) | **2968** | 4568 | 6163 | 6164 | 22 | 0 | ok |
| staggered | blocking | 5642 | 5071 | 5206 | 5347 | 5348 | 15 | 9 | ok |
| staggered | decode-first | 5512 (−131 / −114) | 4957 | 5073 | 5282 | 5306 | 15 | 9 | ok |
| staggered | interleaved | 5893 (+251 / −79) | 4992 | 5294 | 5602 | 5605 | 18 | 6 | ok |
| staggered | **idle** | 6380 (+738 / **−2131**) | **2940** | 4510 | 6083 | 6083 | 24 | 0 | ok |
| refill | blocking | 13313 | 9477 | 12974 | 13022 | 13029 | 27 | 23 | ok |
| refill | decode-first | 13331 (+18 / +15) | 9492 | 12970 | 13060 | 13070 | 27 | 23 | ok |
| refill | interleaved | 13777 (+464 / +339) | 9816 | 12766 | 13527 | 13566 | 33 | 17 | ok |
| refill | **idle** | 15182 (+1869 / −917) | **8560** | 11752 | 14894 | 14903 | 50 | 0 | ok |

括号内是相对 blocking 的 (总耗时 / 首条终稿)。
### 结论

1. **首条终稿延迟是被“新请求的整段 prefill”顶起来的**，不是解码慢。blocking 下新请求的 1.84 s
   prefill 插在活跃序列两次 decode 之间，首条约 5.0 s；`idle` 让新请求等活跃序列结束再开始，
   首条降到 **2.94–2.97 s（−41%）**，P50 也从 5.24 s 降到 4.57 s。
2. **代价是吞吐**：`idle` 的配对次数为 0（完全没有 continuous batching），总耗时 +14%，
   P95/P99 反而变差（6163 对 5417）。
3. **`interleaved`（每轮只推进一个 prefill chunk）只省 70–80 ms**，因为最大的一块是 790 ms 的
   音频 encoder，chunk 粒度上它依然不可分；它把 decoder prefill 摊开，但对首条帮助有限，
   refill 轨迹下总耗时还多 464 ms。
4. **`decode-first` 基本中性**（staggered 上 −131 ms，其余 ±10 ms）：只是把一次 decode 提到
   prefill 之前，改变不了 1.84 s 的阻塞长度。
5. 四档 **逐 token 与串行对照完全一致**；补位（refill 中第三条与活跃序列配对）在 blocking /
   decode-first / interleaved 下都发生（配对次数 17–23），idle 下按定义不发生。

### 适用区间与取舍

- 适用：2 slot、2–3 条请求、约 2 s 音频、SM6650（v73、2 HVX、无 HMX）。超过 3 条请求、
  长音频或 4 slot 未测。
- 产品取舍：要“第一声尽快有答案”（唤醒后立刻回）就用 `idle`，接受总耗时 +14% 与 P95 变差；
  要吞吐就留在 `blocking`；`interleaved` 只有在音频 encoder 可切分之后才有意义。
- 下一步的结构性杠杆是 **把音频 encoder 切成可中断的 chunk**：那 790 ms 是目前唯一挡在
  “首条更快”前面的东西，文本 prefill 已经可以摊开。
### 留档与复算

`~/Library/Caches/qwen3-asr-htp/qwen3-asr-0.6b-batch2-schedule-sm6650-2026-11-19.tar.gz`（796 KB，
SHA-256 `25ef540d8b263566e0aa84c612260b0be93eb6d31c392b016f0d8e029a59b3c6`）包含四档目录、
每条 trace 的 capacity 1/2 JSONL 与 stderr、`schedule-report.json`、设备指纹与 `host.sha256`。

```sh
mkdir -p /tmp/qwen-b2 && tar -xzf <archive> -C /tmp/qwen-b2
python3 examples/qwen3-asr-0.6b/compare_schedules.py --dir /tmp/qwen-b2   # 重算上面两张表
```

设备段（换 `QWEN_SCHEDULE` 即换档，`QWEN_BUDGET_MS` 只对 interleaved 生效，-1 表示每轮一个 chunk）：

```sh
ADB=$ANDROID_SDK/platform-tools/adb QWEN_SCHEDULE=idle QWEN_BUDGET_MS=-1 \
  examples/qwen3-asr-0.6b/run_device.sh build-android-xasr \
    "$CACHE/Qwen3-ASR-0.6B-Q8_0.gguf" "$CACHE/mmproj-Qwen3-ASR-0.6B-Q8_0-convout-q8.gguf" \
    "$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
    /tmp/qwen-b1/fixtures /tmp/qwen-b2/idle 2
```

## 单请求优化：量化、打包与 HTP 算子剖析（2026-11-20）

基线：一条 2.2 s 中文命令，`blocking`、capacity 2、单请求，SM6650。

| 权重 / 打包 | 终稿 ms | 首 token ms | 音频 encoder | decoder prefill | decode 合计 |
|---|---:|---:|---:|---:|---:|
| Q8_0 / eager | 2980 | 1853 | 792 | 1043 | 1024 |
| Q8_0 / packed | 2892 | 1766 | 786 | 969 | 1036 |
| Q5_K / eager | 2958 | 2037 | 790 | 1235 | 835 |
| Q4_K / eager | 2766 | 1942 | 791 | 1142 | **744** |
| **Q4_K / packed** | **2710** | 1884 | 779 | 1096 | 743 |
| Q8_0 + Q4_K projector | 2949 | 1839 | 786 | 1040 | 1023 |

结论：

1. **量化只赚解码**：Q4_K 把 decode 从 1024 ms 降到 744 ms（−27%，解码是权重带宽受限），
   但 decoder prefill 反而 +99 ms（GEMM 变慢）。净收益 −214 ms（−7%）。这跟 Paraformer 上
   看到的“4-bit 只在 M=1 有利”是同一个规律。
2. **projector 量化无效**（786 对 792 ms）：requant 只量化了 19 个张量，其余仍是 F16，音频 encoder
   是计算受限而不是带宽受限。
3. **packed 打包**把 decoder prefill 从 1043 降到 969 ms（−74 ms），首 token 提前 87 ms。

### HTP 算子剖析（`GGML_HEXAGON_PROFILE=1`，一次完整运行）

| 算子 | 次数 | 累计 ms | 占比 |
|---|---:|---:|---:|
| OPBATCH（每次图的派发） | 24 | 5026 | 50.1% |
| MUL_MAT | 1740 | 2564 | 25.5% |
| SET_ROWS | **1234** | **750** | 7.5% |
| MUL_MAT_NX | 1158 | 620 | 6.2% |
| MUL_MAT+ADD | 1214 | 490 | 4.9% |
| IM2COL | 6 | 290 | 2.9% |

两条关键结论：

- **派发成本按图区分**：两次 prefill 的 `OPBATCH` 分别是 **0.76 s 和 0.96 s**，而 22 次 decode 的
  `OPBATCH` 是 **50–100 ms 每次**。单请求 2.7 s 的构成因此是：音频 encoder 图约 0.78 s、prefill 图约
  0.96 s、10 个 decode 步约 0.7–1.0 s——**解码基本是派发受限，不是算力受限**。
- **`SET_ROWS` 是 KV cache 的逐行写入，不是音频 embedding**（先前记错，此处更正）：形状直方图显示
  560 次 `1:1024 x 1024:1 x 1:2097152` 与 560 次 `1024:1 x 1:1 x 1024:2048`，正好是 28 层 x 2（K/V）
  x 20 个 decode 步；另有 56 次 `1024:54`，是两次 prefill 的 54 行 K/V 写入。也就是每层每 token
  一次散布写，单行约 0.6 ms、54 行一次约 10 ms。

### 下一步（按收益排序）

1. **降低每 token 的派发开销**（当前 50–100 ms/步）：单请求最大的一块。方向是同一形状的 decode 图
   跨步复用；**不是** `GGML_HEXAGON_OPBATCH`——512/2048/8192 三档实测终稿 2967/2986/2992 ms，
   与默认 2980 ms 无差别（±1%），说明开销在每次 `llama_process` 的固定往返，不在缓冲区批数。
2. **把 prefill 的 KV 写入变连续**：54 行是连续 idxs，用一次连续拷贝代替 56 次散布写，目标是压
   0.96 s 的 prefill 图。
3. 量化维持 Q4_K（−27% decode），但要用 F16 源重量化，避免 q8→q4 的二次损失。

### 编码器与 attention 的结构性发现（2026-11-20 补充）

按算子名把 profile 分成三段（同一次运行）：

| 段 | 算子数 | 累计 ms | 占比 |
|---|---:|---:|---:|
| 派发 OPBATCH | 28 | 5318 | 51.8% |
| decoder 侧（MUL_MAT/MUL_MAT_NX/SET_ROWS…） | 10690 | 3752 | 35.3% |
| **encoder 侧** | 426 | **1366** | 12.9% |

encoder 的 1366 ms 里，**三层 conv2d 占 654 ms**：`a.conv2d.1` 82 ms、`a.conv2d.2` **224 ms**、
`a.conv2d.3` 59 ms，再加 6 次 `IM2COL` 290 ms；`a.conv_out`（896→1024，q8_0）只有 17 ms。
也就是说 encoder 的时间几乎全在第一层卷积块，不是 attention，也不是投影。

试过的图结构变体都无效（同一条音频）：

| 变体 | encoder ms | 终稿 ms |
|---|---:|---:|
| 默认（im2col + MUL_MAT） | 792 | 2980 |
| `MTMD_QWEN3A_CONV_WEIGHT_FIRST=1`（权重优先布局） | 793 | 2977 |
| `MTMD_QWEN3A_CONV=gather` | 796 | 2983 |

结论：**改图结构解不了 encoder**，瓶颈是 DSP 上的 conv 核（`hvx-flat`、2 MB VTCM、f16×f16→f32，
实测约 44 GFLOP/s，约为该 SoC f16 峰值的四分之一）。真要提速，要么写融合 conv 核，要么把 conv 权重
降到 q8 以减少带宽与 VTCM 压力——两者都在 DSP 侧，不在图拓扑。

### attention：flash 在这颗芯片上是正收益

第一批假设“无 HMX 时 flash 会退化成多个小算子”，实测相反：

| 配置 | 终稿 ms | 首 token | decoder prefill | decode 合计 |
|---|---:|---:|---:|---:|
| 默认（flash 关） | 2980 | 1853 | 1043 | 1024 |
| **flash 开** | **2501** | **1557** | **761** | 864 |
| Q4_K（flash 关） | 2766 | 1942 | 1142 | 744 |
| **flash 开 + Q4_K** | **2348** | 1689 | 896 | **581** |

flash 开把 prefill 砍掉 282 ms；再叠 Q4_K（decode −27%），单请求从 2.98 s 降到 **2.35 s（−21%）**。
`QWEN_FLASH_ATTN=0` 可回到旧档。

### OPBATCH 是什么，静态图/补齐有没有用（2026-11-20）

`OPBATCH` 是 Hexagon 后端把**一张图的所有计算算子打包成一条 FastRPC 消息**交给 DSP 执行，
profile 打出的 `n-ops` 与 `usec` 就是这次打包执行在 DSP 上的时间。它**不是**因为图尺寸变化而重发：
后端按 `graph->uid` 缓存了 op 节点（`GGML_HEXAGON_TIMING=1` 里 `cached 1`），CPU 侧 enqueue 一张
734 op 的 decode 图只要 **2.6 ms**，process 之后的 synchronize 只有 **0.01 ms**，而每步 `llama_process`
是 100–115 ms，profile 里对应步的 `OPBATCH` 是 50–100 ms——时间在 DSP 执行里。

所以“静态图 + 补齐”对这版栈没有收益：

| 设想 | 实际情况 | 结论 |
|---|---|---|
| decode 用固定形状图，省重建 | 同一 trace 跑两轮总耗时 2980 / 2987 ms（<1%）；op 节点已缓存 | 已经是静态的，没有可省的 |
| encoder 补齐到固定长度 | encoder 耗时随帧数线性：2.65 s 音频 778 ms，8.3 s 音频 **3190 ms**（4.1×） | 短音频被补齐后要付最长音频的钱，**负收益** |
| 固定形状省调度重规划 | enqueue 2.6 ms，占每步 100 ms 的 2.6% | 可忽略 |

长音频的对照（同一条命令重复 3 次拼成 8.3 s，Q8_0 + flash）：encode 3190 ms、decoder prefill 2017 ms、
终稿 7526 ms——两个阶段都线性于音频长度，这也说明“补齐到定长”只会让短请求变慢。

### 量化精度：q8 源二次量化到 Q4_K 的对照

| 夹具 | Q8_0 | Q4_K | 一致 |
|---|---|---|---|
| zh | 请导航到上海虹桥机场。 | 同左 | ✓ |
| en | Set the cabin temperature to twenty-two degrees. | 同左 | ✓ |
| de | Navigiere bitte zum Berliner Hauptbahnhof. | 同左 | ✓ |

三条夹具逐字一致（注意这是从 Q8_0 再量化，不是 F16 源；F16 源需要原始 checkpoint，本机没有）。

### 单请求优化收尾：最终配置与“没用的方向”（2026-11-20）

最终配置：**Q4_K 权重 + flash attention 开 + 4 线程**，一条 2.65 s 中文命令 **2348 ms**
（基线 Q8_0/flash 关是 2980 ms，−21%）。分阶段：encoder 779、decoder prefill 896、decode 581。

逐项结论（全部在 SM6650、同一 trace 同一天测得）：

| 方向 | 结果 | 判定 |
|---|---|---|
| decode 图跨步复用 | op 节点按 graph uid 命中（cached 1），enqueue 2.6 ms，sync 0.01 ms，每步 100–115 ms | 无 CPU 侧可省，时间在 DSP 执行 |
| 静态图 + 补齐 | 两轮同 trace 差 <1%；encoder 随帧数线性（2.65 s→778 ms，8.3 s→3190 ms） | 静态图无收益，补齐是负收益 |
| prefill KV 连续写 | flash 开后独立 K/V SET_ROWS 从 profile 消失（原 750 ms），换成 FLASH_ATTN_EXT 430 ms | 已被 flash 覆盖，无需再做 |
| OPBATCH 512/2048/8192 | 2967/2986/2992 ms | 无效 |
| 宿主线程 2/4 | 2351/2355 ms | 无差别（DSP 受限） |
| packed 打包 | flash 关时 prefill 1043→969 ms | 小幅正收益，最终配置未启用 |
| conv 图变体 | weight-first 793 ms / gather 796 ms vs 792 ms | 无效 |
| Q4_K（q8 源再量化） | decode 1024→744 ms，三条夹具文本一致 | 采用 |

剩下的两个真实瓶颈都不在可改的层：

1. encoder 的三层 conv2d（654 ms，占 encoder 84%）：hvx-flat、f16×f16→f32、2 MB VTCM，实测约
   44 GFLOP/s（该 SoC f16 峰值约四分之一）。提速需要 DSP 侧融合 conv 核或 conv 权重 q8。
2. decode 每 token 的 DSP 执行（Q4_K 后约 58 ms）：28 层一次前向，无法再摊薄，除非一次提交多个
   token（speculative decoding）或多个请求（continuous batching）。

复现命令：

    QWEN_EXTRA_ENV='QWEN_FLASH_ATTN=1' QWEN_SCHEDULE=blocking \
      examples/qwen3-asr-0.6b/run_device.sh build-android-xasr \
        $CACHE/Qwen3-ASR-0.6B-q4k.gguf \
        $CACHE/mmproj-Qwen3-ASR-0.6B-Q8_0-convout-q8.gguf \
        $NDK/.../libc++_shared.so /tmp/qwen-solo /tmp/qwen-q/final 1
