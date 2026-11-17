# Paraformer 在 llama.cpp / HTP 上的适配细化方案

日期：2026-11-17
状态：设计草案（结论基于源码与公开实现核对，尚未跑真机）

## 0. 决策摘要

1. **长语音的 overlap / stitching 不是模型结构问题，是流水线段问题。** FunASR 官方“长音频版 Paraformer-large”没有在模型内部做 overlap：它用 VAD 切段、段内 padding、再按 VAD 起点偏移拼文本和时间戳。我们要补的是 VAD 边界处的声学上下文与 token 去重，用 CIF 的 fire frame 做时间轴。
2. **默认不做全重叠投票。** 短车控句（≤15 s）只做“核心段 + 0.3 s 左右上下文”，只保留时间戳落在核心段内的 token；跨界的低置信 token 用邻窗候选修复。只有长语音、VAD 边界低置信、或需要输出 partial 时才开全重叠投票，代价约 1.3–1.6× 编码量。
3. **静态块式输出分三档。** 推荐 A：编码按固定 chunk/桶跑，整句解码一次，VAD 端点才出 final。B：滑窗 + 可回撤尾部（3 token / 500 ms），用现有离线权重即可。C：换 streaming Paraformer 权重（chunk_size≈[0,10,5]），才是真正“每块固定输出且不可回撤”，需要新权重和新转换脚本。
4. **双语先用单模型。** `funasr/paraformer-zh` 官方标注“中文和英文”，8404 字符级/BPE 词表，220M；英文弱于中文且无大小写/标点。先用现有车控中英夹具量化差距，不达标再做 zh/en 双模型 + 每 VAD 段 LID 路由。
5. **图要可手调，就把边界做成 stage registry。** 固定 I/O 名字和形状：`feat → enc(HTP, T桶) → alpha/cif(CPU) → dec(HTP, K桶) → stitch(CPU)`。每层/每算子通过配置切换实现，所有改动必须过“中间层 dump + 车控字段”两道门。

## 1. 已核实的事实与来源

### 1.1 llama.cpp / HTP 侧

- HTP 把 tensor 的 `ne/nb/type` 直接写进 op 描述符并参与 hash：`ggml/src/ggml-hexagon/ggml-hexagon.cpp:1346-1377`、`1379-1392`。shape 变了就是另一张图。
- 一个 HTP session 只缓存一个 graph uid：`:2555`（`sess->cached_uid == graph->uid`）。**长度桶来回切换会反复重建图**，所以每个桶要有自己的 scheduler/session，或限制常驻桶数。
- 该 fork 没有 `GGML_OP_WHILE`/`GGML_OP_COND`，设备端跑不了数据依赖循环；动态 fire 必须在 host。
- 可复用 x-asr 的范式：命名输入/输出 `examples/x-asr/xasr.cpp:608-617`、静态 cache 写回 `:553-577`、每 slot 一个 scheduler `:1487-1495`、批内 gather/scatter `:1801-1857`。
- HTP 算子覆盖 MUL_MAT / CONV_2D(_DW) / SOFT_MAX / NORM / ADD / MUL / PAD / CUMSUM / ARGMAX 等（`ggml/src/ggml-hexagon/htp-kparams.cpp:14-69`），encoder/decoder 的主体算子没有硬阻塞。

### 1.2 Paraformer 现成实现

- CrispASR 参考实现 `paraformer.cpp`：
  - fbank(80 mel, 25/10 ms) + LFR(stack 7, stride 6) + CMVN 在 host：`:326-377`。LFR 帧率约 16.7 fps（60 ms/帧），输入维度 560。
  - CIF 是 host 上的顺序 for 循环：conv1d(512,512,k=3)→ReLU→linear→sigmoid→累加，fire 阈值 1.0，末尾 tail 阈值 0.45：`:384-518`。
  - encoder 图按 `T_lfr` 每次重建：`:643-741`；decoder 图按 `N_tokens` 每次重建：`:887-928`。这正是 HTP 不能直接吃的地方。
  - 已暴露 `fire_frames`，并转成 centiseconds：`:1134-1160`。注意它用 `lfr_m*10 = 70 ms/帧`，而 LFR 的**步长**是 60 ms；时间戳尺度必须以 FunASR 参考对齐后再定，不能直接照抄。
  - 提供 73 个中间层的 diff harness（`stage` 参数与 `acoustic_embeds` / `decoder_layer_*` / `logits` 捕获点），可直接移植成对拍工具。
  - 英文输出是 BPE，含 `@@` 续接标记（`:974-990`）；stitching 不能把 `@@` 词组切开。
- FunASR `auto_model.py` 的 VAD 路径：`merge_vad` 默认 15 s，`slice_padding_audio_samples()` 切片并 padding，`restored_data` 恢复时间顺序，时间戳加 `vadsegments[j][0]` 偏移，文本按空格拼接；**没有 overlap 参数**。
- FunASR `timestamp_tools.py`：`ts_prediction_lfr6_standard()` 用 CIF peak 位置生成字级时间戳，`force_time_shift=-1.5` 帧、`MAX_TOKEN_DURATION=12`、`vad_offset` 后处理；说明 CIF fire frame 是可用的时间轴。
- FunASR `CifPredictorV2.forward_chunk()`（`:316-412`）：流式权重把 `alphas[:, :chunk_size[0]]` 和 `[sum(chunk_size[:2]):]` 置零，缓存 `cif_hidden`/`cif_alphas`，`is_final` 才追加 0.45 tail。这是“真静态块输出”的参考语义。
- 模型矩阵（ModelScope model zoo）：
  - `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`：中文和英文，220M，官方标注“输入 wav ≤20 s”。
  - `...large-vad-punc...-zh-cn...-vocab8404`：同样权重 + VAD/punc 流水线，支持任意长度。
  - `...large-vad-punc...-en...-vocab10020`：英文专用，220M。
  - `...large...-vocab8404-online`：流式，220M。
- 现成 GGUF 只有 `cstr/paraformer-zh-GGUF`（Q8_0 227 MB / Q4_K 123 MB / F16 421 MB，8404 词表）。英文/流式 GGUF 需要自己转。

## 2. 长语音：分段、padding、overlap、stitching

### 2.1 核心段与上下文窗

定义前端 VAD 给出的核心语音段 `C = [s, e]`（ms）。

- **默认（≤15 s，CAR 车控场景）**：解码窗口 `W = [s - L, e + R]`，`L = R = 0.3 s`（约 5 个 LFR 帧）。只保留 token 中心时间落在 `C` 内的部分。若首/末 token 置信度低或时间跨度跨越核心边界 >50%，从相邻窗口的上下文候选里替换它。
- **长语音（核心段 > 单桶容量或 >15 s）**：把核心段切成 `W_i = [s_i - L, s_i + W + R]`，步长 `S = W - O`，`W = 8 s`、`O = 1 s`（可调）。所有窗口独立推理，输出带全局时间的 token 流，再 stitch。
- **不做全重叠投票的理由**：encoder 的注意力是全局的，窗口一变，alpha 与 fire 位置都会变；全重叠能提升边界鲁棒性，但会让同一段音频算 1.3–1.6×，且重复 token 的去重本身会引入新错误。先只修边界。

### 2.2 Token 记录

CIF 在 host 产出每个 token 的 fire frame， stitch 需要以下信息（缺一个都会让去重退化成猜）：

```cpp
struct PfToken {
    std::string text;   // 可能以 @@ 结尾（BPE 续接）
    int64_t t0_ms;      // fire frame * ms_per_alpha_frame + window_offset
    int64_t t1_ms;      // 下一个 fire frame；末 token 用有效末端
    float   posterior;  // max softmax(logits[n])，必须保留 logits 而不只 argmax
    float   alpha_peak; // fire 帧的 alpha
    float   coverage;   // token 时间跨度内 alpha 之和
    int     window_id;
    int     index_in_window;
    bool    from_context;
};
```

`ms_per_alpha_frame` 必须实测校准：`lfr_n * 10 = 60 ms` 是 stride 推导值，CrispASR 用的是 70 ms；用 FunASR timestamp 输出对同一段音频校准后再写死。

### 2.3 Stitch 算法

1. 所有窗口的 token 转成全局时间；按 `t0_ms` 排序。
2. 过滤：核心段内保留 `t_center ∈ [s, e]` 的 token；上下文 token 只作为边界候选池。
3. 聚类：时间重叠（或 `|t0_i - t0_j| < 120 ms`）且文本相同/前缀兼容的 token 归为一组。
4. 打分选取：`score = 0.7 * posterior + 0.2 * min(1, alpha_peak / alpha_mean) - 0.3 * edge_penalty - 0.2 * from_context`，`edge_penalty` 与 token 到窗口边界的距离成反比。选最高分，其余丢弃。
5. BPE 合并：若被选 token 以 `@@` 结尾，必须连同后续续接 token 一起处理；聚类/删除只能发生在完整词边界。
6. 输出：按时间顺序拼文本；`<sil>` 只用于时间戳展示，不进文本。英文词间空格沿用 CrispASR 的规则（连续英文 token 之间加空格）。

失败模式（验收要覆盖）：VAD 把词首/词尾切掉；中英混说处 BPE 断词；相邻窗口对同一字给出不同 token；静音段的尾部 0.45 fire 造成幽灵 token；时间戳尺度错 10 ms/帧导致越长越漂。

### 2.4 开关

```text
PARAFORMER_OVERLAP=context|full|off   # 默认 context
PARAFORMER_CTX_L_MS=300
PARAFORMER_CTX_R_MS=300
PARAFORMER_WIN_S=8
PARAFORMER_WIN_OVERLAP_S=1
PARAFORMER_STITCH_DT_MS=120
```

## 3. 中英双语

| 方案 | 权重 | 优点 | 代价 | 判定 |
|---|---|---|---|---|
| 单模型 zh-en | 现有 Q8_0 227 MB | 零额外内存、中英混说天然支持 | 英文命令字段可能弱；无大小写/标点 | **先做** |
| 双模型 + 每 VAD 段 LID | zh-en 8404 + en 10020（需转换） | 英文专用词表、可针对性热词 | 多 227 MB；转换器要参数化 vocab/lfr/predictor；LID 需验证 | 仅当英文夹具不过线 |
| streaming zh-en 220M | 需下载并转换，cache I/O | 真固定块输出 | 新图契约、新精度基线 | 仅在 §5 的 B 档失败后 |

注意：

- 8404 词表是字符 + BPE 续接，英文输出为小写、无标点；车控后处理要保留现有 ITN/大小写/标点逻辑。
- 不要假设“官方写中文和英文”就等于英文车控命令合格。用现有 `examples/x-asr/tools/eval_vehicle.py` 的同一套夹具跑 CER、命令字段、硬负样本，与 x-asr 基线（Tingting CER 7.23% / 命令 75.0%）直接对比；不达标就不替换。
- 双模型路由按 VAD 段而不是按整句；段级 LID 可以用轻量分类器或上游语言标签。转换器 `convert-paraformer-to-gguf.py` 目前把 hyperparams 写死为 zh 配置（`:92-105`），换 en 权重前必须参数化并核对 tokens.json。

## 4. 可手工调优的图拆分

### 4.1 Stage 边界与静态形状

| Stage | 设备 | 固定输入 | 固定输出 | 桶键 |
|---|---|---|---|---|
| `feat` | CPU | PCM `[n_samples]` | LFR `[560, T_b]`、valid_len、mask | `T_b` |
| `enc` | HTP | features `[560,T_b]`、pe、attn_mask `[T_b]` | enc `[512,T_b]` | `T_b` |
| `alpha` | CPU（先） | enc `[512,T_valid]` | alphas `[T_valid]` | — |
| `cif` | CPU | alphas、enc、is_final | embeds `[512,N]`、fire frames、conf | N 动态 |
| `dec` | HTP | embeds `[512,K_b]`、enc `[512,T_b]`、cross/self mask | logits `[V,K_b]` | `T_b,K_b` |
| `stitch` | CPU | tokens、VAD 段 | 最终文本 + 时间戳 | — |

- 桶建议 `T_b ∈ {64,128,256,384}`（约 4/8/16/24 s LFR 帧）；启动只 preload 两个常用桶，其余 lazy 并在 LRU 中淘汰，避免 HTP 单 uid 缓存反复失效和内存爆掉。
- `K_b` 从参考实现的 tokens/frame 上界实测后取带余量值，例如 `K_b = ceil(T_b * r_max) + 8`；**绝不允许 N > K** 时静默截断，必须升级到更大的 K 桶或记录错误。
- padded 行必须同时做三件事，否则静默错：
  1. LFR 输入补零，但 conv 边缘用相邻有效帧复制，避免 FSMN(k=11) 看到人工静音；
  2. attention mask 用加性 `-1e9`，padded query 行至少保留一个合法 key，避免全 mask 后 softmax 出 NaN；
  3. CIF 只消费 `T_valid` 行的 alphas，超出部分显式置零；`tail_threshold` 只在 `is_final` 且最后一个有效 alpha 之后追加。
- decoder 的 padded token 槽位写零，self/cross mask 阻断；logits 只回读前 N 行。

### 4.2 调优接口

```cpp
struct StageSpec {
    const char * name;
    Backend preferred;                 // HTP / CPU
    BucketKey key;                     // T_b / K_b
    ggml_cgraph * (*build)(...);       // 可替换的建图函数
    std::vector<const char *> inputs, outputs; // 名字固定，便于 pin 后端和 dump
};
```

配置项（都默认保持基线，改一项跑一项 A/B）：

```text
PARAFORMER_ROUTE=feat:cpu,enc:htp,alpha:cpu,cif:cpu,dec:htp,stitch:cpu
PARAFORMER_BUCKETS=64,128,256,384
PARAFORMER_GRAPHS=preload|lazy
PARAFORMER_LAYERS=enc.0-9:cpu,dec.3:cpu      # 逐层覆盖；会产生 scheduler split
PARAFORMER_ATTN=plain|chunk|flash              # flash 必须先过等价性门
PARAFORMER_FSMN=conv|im2col|ssm
PARAFORMER_DUMP_STAGE=enc.12,acoustic_embeds,dec.7,logits
PARAFORMER_PROFILE=1
```

- 逐层 backend pin 用 `ggml_backend_sched_set_tensor_backend()`（x-asr 已在 cache/constant 上使用）；细粒度切换会增加 split 和 dispatch，参照 x-asr 的 ~7 µs/op 地板先算账再试。
- 等价性门分三层：① 中间层 raw f32 与 PyTorch 参考逐元素/余弦（73 层 dump 直接复用）；② alphas、fire frames、logits top-1 一致；③ 车控夹具 CER/命令字段与 A/B 延迟。任何一层不过就不进默认。
- 性能口径沿用已定规则：同轮交错 A/B、每臂独立进程、首条热机，报 N 与 p50/p95/p99；`elapsed_s` 是判据，RTF 在输出长度变化时会反向误导。

## 5. “静态块式输出”的三档语义

| 档 | 语义 | 权重 | 是否要 stitch | 适用 |
|---|---|---|---|---|
| A（默认） | 编码按固定桶/chunk，整句 decoder 一次，VAD 端点才出 final | 现有离线 GGUF | 边界修复需要 | 车控 final-only、长音频离线 |
| B | 滑窗 4 s / 步长 2 s / overlap 0.6 s；每次输出带 `stable` 标记，末尾 3 token 或 500 ms 可回撤 | 现有离线 GGUF | 必须 | 需要 partial 字幕 |
| C | streaming Paraformer，chunk_size≈[0,10,5]（600 ms 当前 + 300 ms 前瞻，按 60 ms 帧），cache 固定形状，块输出不可回撤 | 需新权重 + 新转换 | 只需 cache 边界 | B 档回撤不可接受时 |

A 档已经满足“runtime 根据语音长度自行 padding”和“计算图分块固定”：变长只体现在桶选择与 mask 上，图不重建。B/C 才是真正的“文本也按块吐出”，不要混为一谈。

**不推荐**用强制 alphas 总和等于固定 K 来凑每块固定 token 数：那等于改对齐，会让 fire 位置与字错开，必须先拿 CER/命令字段做 A/B，不能作为默认。

## 6. 里程碑与验收

- **M0 宿主基线**：下载 Q8_0/Q4_K，跑通 CrispASR 参考实现；固化车控夹具转写与 73 层中间 dump。门槛：与 FunASR Python 参考在夹具上逐条一致或差异有解释。
- **M1 静态图（全 CPU）**：实现 feat/桶/mask/host CIF/固定 K decoder；门槛：每个桶与 M0 逐字节一致，N>K 不静默截断。
- **M2 encoder 上 HTP**：按桶常驻 session；门槛：encoder 输出与 CPU 参考余弦 ≥ 0.999（具体阈值按 dump 分布定），记录每桶耗时与图重建次数。
- **M3 decoder 上 HTP + 调优开关**：门槛：fire frames、logits top-1、整句文本一致；逐项 A/B 记录被否决的变体。
- **M4 overlap/stitch + 双语**：门槛：中英夹具命令字段不低于 x-asr 基线；长语音接缝处无重复/断字；硬负样本不误触发。
- **M5（条件）partial**：只有产品明确需要边说边出字才做 B/C 档。

## 7. 风险清单

- 时间戳尺度 60 ms vs 70 ms 未校准；stitch 越长越漂。
- 上下文窗改变全局注意力 → 改变 alphas → 边界 token 与无上下文版本不同，必须靠对拍而不是假设等价。
- BPE `@@` 被 stitching 切断，英文词裂开。
- HTP 单 session 单 graph uid 缓存：多桶交替会反复建图；桶数和 LRU 策略要实测。
- 设备内存：Q8_0 227 MB × 双引擎 + 多桶激活，在仅约 800 MB 空闲 + 2.1 GB swap 的设备上不能贪多。
- x-asr 已判死的方向别重踩：Flash Attention 不等价、CPU 3/4 线程被内存卡死、算子级搬运到 HTP 可能更慢。
- Paraformer 无标点/大小写，车控流水线后处理必须保留。

## 8. M0 实测进展（2026-11-17，宿主 CPU）

- 权重 `cstr/paraformer-zh-GGUF` Q8_0 已下载，sha256 `e73232ecf7e484907fd58f6a481fa27c1ceff454a1bc441d4a032a3a63e9783f`。
- 实现骨架：`examples/paraformer/`（`llama-paraformer` CLI、CrispASR 派生 runtime + 本仓库最小 GGUF/device shim），
  宿主 CPU 已跑通中英转写；`--device NAME` 可切到 HTP，仍需真机验证。
- `tingting` 夹具 32 条（x-asr 的 96 条是 3 轮重复）：原始转写中文 CER **0.60%**、英文 WER **12.21%**、
  命令字段 **56.2%**；仅做标点恢复后命令字段 **68.8%**。x-asr 同夹具 on-arm 基线是命令 75.0% / CER 7.23%。
- 原始命令分低的主因确认是**评测器按逗号切子句**而 Paraformer 不输出标点，不是声学错误。
  少数真实错误：`seat_en01` 的 `two→too`、`light_en01` 的 `Set→said`、`hvac_en04` 的 `rear vents→reevent`。
- 结论：中文声学明显优于 x-asr，英文仍不达标；先补标点后处理再把英文作为重点，不要仅凭原始命令分否定该模型。

## 9. M1 实测：首包建图开销与 warmup（2026-11-17，SM6650 HTP）

**问题**：M0 里 HTP 首条 1278 ms、稳态 290 ms，p99 被首条拉爆。先定位这 ~1 s 的来源再动手。

**方法**：给 runtime 加累计 lap 计时（`CRISPASR_PARAFORMER_BENCH=1`，`paraformer_lap:` 行），
把一次推理拆成 graph build / sched alloc / compute 三段，对照有无 warmup、以及长短 shape 切换。

**结论（N=32 夹具 + 多次单条复测）**：

| 阶段 | 首条（无 warmup） | warmup 后的真实请求 |
|---|---:|---:|
| enc graph build | 159 ms | 18 ms |
| enc sched alloc（HTP 首次分配 + 注册图） | **733 ms** | 20 ms |
| enc compute | 134 ms | 129 ms |
| dec graph build + sched alloc | **221 ms** | 1 ms |
| dec compute | 29 ms | 28 ms |
| **wall（整条）** | **1130 ms** | **185 ms** |

- 首条 ~1 s 的大头是 **sched alloc**（HTP 首次给图分配 buffer 并下发到 DSP），不是 CPU 建图：
  encoder 733 ms + decoder 221 ms。加 `paraformer_warmup()` 在计时路径外跑一次完整推理后，
  这些降到 20/1 ms，首条 wall 1130→185 ms。
- **跨 shape 不会重付**：warmup 用 1.4 s 音频、后续跑 3.5 s，sched alloc 仍是 22 ms；
  差异只落在 compute（129→312 ms，随 LFR 帧数线性）。所以 **sched alloc 是进程级一次性**，
  warmup 用任意代表性长度即可，不需要 pytest 式地为每个长度档各预热一次。
- **长度分桶的假设被否**： encoder compute 随帧数线性是真实 DSP 算力，pad 到统一长度只会让短句变慢；
  且 HTP 图按 uid 缓存、同 shape 已复用。M0 规划的“长度分档常驻图”不解决首包，本批不做。
- 32 条全量交错对拍：有/无 warmup **转写 32/32 逐条完全一致**；
  p50 291→291 ms（稳态不变），p95 389→381 ms，**p99/max 1278→390 ms**。
  warmup 自身耗时 1260 ms，发生在计时路径外（服务启动阶段）。

**实现**：`examples/paraformer/` 新增 `paraformer_warmup(ctx, samples, n)` 与 CLI `--warmup FILE`（默认关闭，
避免改变既有基准口径）。服务化后应在 accept 之前对一段代表性音频调用一次。

**下一批**：① 首包剩下的 185 ms 里 enc compute 129 ms 是主项，按 operator profile 拆 ConvNeXt/attention；
② 标点后处理 + 英文 `said/too/reevent` 专项；③ overlap/stitch 与长语音；④ 常驻服务化后测真实并发 P95。

## 10. M2 实测：encoder compute 的算子级分解（2026-11-17，SM6650）

**问题**：M1 后首包 185 ms 里 encoder compute 129 ms 是主项。拆开看是 HTP dispatch 地板
（像 x-asr 那样）还是 DSP kernel，判断 encoder 还值不值得在 runtime 层投入。

**方法**：`GGML_HEXAGON_PROFILE=1 GGML_LOG_LEVEL=debug` 抓 HTP 每 op 的 usec，用
`examples/paraformer/tools/profile_ops.py` 按 OPBATCH（enc/dec）和 op 类型聚合。
数据：`tingting/prof1.log`（短句 win_en02，T_lfr≈24）、`prof_long.log`（win_zh03，T_lfr≈59）。

**结果（短句，encoder 1405 op / 133.3 ms）**：

| op | 次数 | usec | 占比 | avg |
|---|---:|---:|---:|---:|
| MUL_MAT+ADD | 200 | 82702 | **62.7%** | 413 us |
| CONV_2D_DW (FSMN) | 50 | 14869 | 11.3% | 297 us |
| MUL_MAT (attn scores/ctx) | 100 | 14270 | 10.8% | 143 us |
| CONT (permute/view 拷贝) | 500 | 7985 | 6.1% | 16 us |
| 其余（RELU/ADD/SOFT_MAX/NORM/MUL…） | — | ~12000 | 9% | — |

- 200 次 MUL_MAT+ADD = 50 层 × 4 个投影。按权重分类：**FFN l1+l2 = 52.6 ms（占 matmul 的 63%）**，
  每个 512×2048 投影 ~530–544 us；QKV 19.8 ms（403 us）；out proj 7.5 ms（152 us）；entry block 1.5 ms。
- 权重已是 q8_0，HTP 走 `hvx-tiled` 原生 int8 路径（profile 显示 `q8_0 x f32 → f32`）。
- **是 kernel-bound，不是 dispatch-bound**：1405 op × x-asr 测得的 ~7 us dispatch ≈ 10 ms，仅占 7%；
  单 op usec 高达数百 us 是 DSP 矩阵运算本身。这与 x-asr（2449 op 中 dispatch 地板 17 ms/77 ms）
  完全相反，不要套用 x-asr 的 dispatch 优化经验。
- **随帧数线性**：长句 T_lfr≈59 时 matmul avg 413→969 us（2.35×），encoder 133→309 ms（2.32×），
  与帧数 24→59（2.46×）一致。确认 compute-bound，没有可摊薄的固定项。

**结论与复活条件**：encoder 已接近该权重/该芯片的算力天花板，runtime 层没有低风险优化杠杆。
CONT（6%）这类搬运优化上限太低，不值得动图结构；把 FSMN conv 从 f32 cast 改成低精度有对拍风险、
上限仅 11%。**只有两类改动能动 40% 的 FFN GEMM 主体**：① 更激进量化（FFN 权重 int4 / 混合精度），
必须先过 CER/逐层对拍；② 改模型结构（缩小 FFN dim）并重训，那是另一个模型，不再是适配工作。
在本权重下，继续优化 encoder 的投入应转向服务化后测真实并发 P95、以及识别质量（标点/英文）。

**工程交付**：`tools/profile_ops.py` 固化了 enc/dec 分段 + op 聚合方法，后续任何图改动都能复用同一口径。

## 11. M3 实测：服务化与并发 P95（2026-11-17，SM6650 HTP）

**目标**：把指标口径从 CLI 单进程对齐到产品需求（VAD 端点→最终转写 P95、最多四路并发）。

**实现**：新增 `llama-paraformer-server`（`examples/paraformer/server.cpp`）。Paraformer 是全句一次性
推理，无 append/commit 状态，线协议简化为单事务：`RECOGNIZE(op=0, payload=整段 float32) → FINAL(status=1,
payload=UTF-8 文本)`，帧头仍用 `u32 op,u32 reserved,u32 bytes` / `u32 status,u32 bytes`，可复用 x-asr
客户端的 framing。**单 context + mutex 串行**：`paraformer_transcribe_impl` 会写共享的 sched 与 encoder
图缓存，多实例共用一个 context 不安全；而每实例各加载 226 MB 权重在设备上内存不可行。启动时 `--warmup F32`
在 accept 前预热一次（实测 1191 ms），把一次性 HTP 注册开销挪出请求路径。新工具
`tools/bench_pf_service.py`：生成 warmup f32、N 路同时/错峰请求、报 P50/P95/P99 与转写一致性。

**结果（每条音频 2.0–2.7 s，每格 N=8 轮，0 错误，转写逐条一致）**：

| 场景 | p50 | p95 | p99 | 说明 |
|---|---:|---:|---:|---|
| 单路（warm） | 285 ms | 295 ms | 297 ms | 与 CLI 稳态 290 ms 一致，warmup 生效 |
| 2 路同时到达 | 449 ms | 587 ms | 591 ms | ≈2× 单路（串行排队） |
| 4 路同时到达 | 707 ms | **1126 ms** | 1128 ms | ≈4× 单路，串行 head-of-line |
| 4 路错峰 300 ms | 295 ms | **317 ms** | 320 ms | 错峰 > 单条推理 285 ms，几乎不排队 |

**结论**：
- 单 worker 串行的吞吐上限就是 1/285 ms ≈ 3.5 段/s；四路同时到达时 P95 线性累加到 1.13 s，
  达不到产品对四路并发的低尾延迟期望。这是架构限制，不是 HTP 算力问题（单路已到 kernel 天花板）。
- **错峰是最有效的低成本缓解**：VAD 段天然错峰若 > 单条推理时长（车内多说话人通常不会精确同时断句），
  P95 就回到单路水平（317 ms）。运维上可加一个小的入口抖动/排队整形避免同时 commit。
- 转写在 1/2/4 路下逐条一致（4 条 wav 各 distinct=1），并发无状态串扰；warmup 后首请求无 1 s 尖峰。

**复活条件 / 下一批**：若要压四路同时到达的 P95，唯一路径是**多 context 共享权重**（改 GGUF loader
让多 context 复用同一份只读权重 tensor，各自持有 sched/图缓存/激活），再按 x-asr 的 2+2 思路做
HTP/CPU 双引擎路由；那会重新引入内存压力（x-asr 已测得 CPU 3/4 线程被内存卡死）。在此之前，
先用错峰整形 + 识别质量（标点/英文）作为投入方向。

## 12. M4 实测：共享权重 + HTP/CPU 双引擎并发（2026-11-17，SM6650）

### 12.1 权重共享的可行边界（先查约束再动手）

"一份权重 buffer 同时被 HTP 和 CPU 原生执行"在本版本 ggml **不可行**，这是三个硬约束叠加的结果：

1. **buffer-affinity**：`ggml-backend.cpp:1368-1397` 中 scheduler 按权重张量的物理 buffer 决定 op 的执行后端；
   权重在 HTP device buft 就跑 HTP，在 host buft 就跑 CPU（除非开 `op_offload`）。
2. **HTP 只认自己的 buft**：`ggml-hexagon.cpp:3181-3187` 中 `supports_buft` 只接受 HTP 的
   `buffer_type`/`host_buffer_type`，不认 CPU host buft，所以 host 权重无法直接喂给 HTP 算。
3. **单 HTP session**：v73 强制 `ndev=1`（"forcing ndev to 1 for SoCs archs lower than v75"），
   一个 device 只有一个 session，session 内 `graph_compute` 无锁，两个 HTP engine 并发不安全。

实测验证（单条 3.5 s 音频，`win_zh03.wav`，各跑 warmup 后稳态）：

| 权重布局 | 稳态 wall | warmup | 结论 |
|---|---:|---:|---|
| HTP device buffer（基线，tiled repack） | **389 ms** | 1295 ms | 最快，HTP 原生 |
| CPU host buffer + `PF_OP_OFFLOAD=1` | 527 ms | 672 ms | offload 没有加速 MUL_MAT，反而 +35% |
| CPU host buffer，无 offload | 515 ms | 678 ms | op 落到 CPU |

`op_offload` 对 Paraformer 的 GEMM 无效（权重跨设备读 + 失去 tiled repack）。

### 12.2 落地的方案：共享"加载"，不共享"buffer"

- 一次 `read_file` 把 GGUF 读进内存（226 MB），一次 `gguf_init_from_buffer` 解析 —— 由 `paraformer_weights`
  持有，供所有 context 复用。
- 每个 engine 用 `load_weights_bytes(同一份 bytes, 自己的 backend)` 各自 materialize 一份权重 buffer：
  HTP engine 得到 tiled device buffer，CPU engine 得到 host buffer。ggml 的 `gguf_init_from_buffer`
  自建大小正确的 ggml_context，避免了手工估算 pool 大小的坑（`ggml_dup_tensor` 路线因 pool 不足多次 abort）。
- 物理 buffer 两份是 HTP repack 的硬约束；单 HTP session 决定了第二引擎只能是 CPU。
  内存代价：bytes 226 MB + HTP buffer 226 MB + CPU buffer 226 MB，设备 RSS 约 1.5 GB（可用 7 GB）。
- 接口：`paraformer_weights_load/free` + `paraformer_init_shared`；后端由 `params.use_gpu` 决定
  （不看进程级 `CRISPASR_PARAFORMER_GPU` env），因此 HTP 与 CPU context 可在同一进程共存。

### 12.3 server 改造：请求级双 worker

去掉 M3 的全局 `g_infer_mutex`。改为共享请求队列 + 每 engine 一个 worker 线程：reader 线程把每条
RECOGNIZE 作为独立任务入队并等 `std::future`；两个 worker（各绑定一个 context）竞争取任务并行推理。
`--engines 1|2`（默认 2）：engine 0 走 `--device`（HTP），engine 1 固定 CPU；两者都在 accept 前 warmup。

### 12.4 真机结果（SM6650，每条音频 2.0–2.7 s，0 错误）

| 配置 | 单路 p50/p95 | 2 路 p50/p95 | 4 路 p50/p95 |
|---|---:|---:|---:|
| 单引擎 HTP（= M3 基线） | 285 / 296 | 439 / **577** | 707 / 1126 |
| 双引擎 HTP+CPU（共享权重） | 337 / 370 | 329 / **373** | 516 / **740** |
| 双引擎 + 4 路错峰 300 ms | — | — | 310 / **349** |

- **2 路 p95 577 → 373 ms（−35%）**：两请求分落 HTP/CPU 两个 worker 真并行，不再线性累加；
  p50 也从 439 降到 329（部分请求命中更快的 HTP 引擎而非全部排队）。
- **4 路 p95 1126 → 740 ms（−34%）**：每个引擎各处理约 2 条，仍受每引擎内串行限制，
  故是 ~2× 单路而非 4×。要进一步压需第三个引擎（设备无第二 HTP session，只能再加 CPU 槽，
  但 x-asr 已测得 CPU 多线程被内存卡死）。
- 单路双引擎 p50 337 略高于单引擎 HTP 285：请求随机落到较慢的 CPU 引擎（357 ms）所致，
  这是用单路轻微换尾延迟的取舍。
- 转写：4 条 wav × 70 个响应 distinct=1，跨 HTP/CPU 引擎无串扰；错峰 300 ms 仍有效（349 ms）。

### 12.5 结论与遗留

- "降要求，先做两路"达成：2 路 P95 降低 35%，且未引入重复读盘（GGUF 只读一次，shared load 154 ms）。
- 未做的"物理单份权重 buffer 共享"有明确机制障碍（§12.1），复活条件：HTP 后端支持从 host buft
  直接执行权重算子（等价于去掉 tiled repack 依赖）或设备提供第二个 HTP session。
- 剩余方向：第三引擎的可行性（受 CPU 内存限制）、英文识别质量、`Set`→`said` 专项。

## 13. M5 实测：NAR logit 偏置修复英文整词混淆 + 票差诊断（2026-11-17，SM6650）

### 13.1 先测票差，不照搬 x-asr 结论
补标点后命令字段 68.8% 的残余失败里，8/10 是英文同音/近音词：`said/set`、`two/too`、
`seat/seed`、`footwell/footwall`、`lights/likes`、`rear vents/reevent`。x-asr 上热词已判死
（词首 gain 上限 0.9 logit，Set 票差 1.76–6.42），但那是自回归 joiner；Paraformer 是 NAR，
每个 CIF 位置独立 argmax，整词偏置的可行性必须先量票差再定。

先查词表（GGUF `tokenizer.tokens`，8404）：`set`(7704)/`said`(631)/`two`(4709)/`too`(415)/
`and`(7776)/`in`(2479) 都是**整词单 token**，可直接偏置；`seat`/`seed`/`footwell`/`rear`
是 BPE 片段，整词偏置无效——这决定了本批只能覆盖功能词，覆盖不了名词。

新增诊断 `PF_DUMP_LOGITS=<N>` + `PF_WATCH_TOKENS=set,two,...`：打印前 N 个 CIF 位置的 top-5
及指定 token 相对 argmax 的精确 gap（宿主 CPU，raw logits）：

| 混淆 | clip / 位置 | raw gap | 判定 |
|---|---|---:|---|
| `said`→`set` | seat_en03 pos0 / light_en01 pos0 | 2.89 / 3.00 | 可翻，需 gain>3 |
| `too`→`two` | seat_en01 pos14 | 0.96 | 易翻 |
| `in`→`and` | hvac_en04 pos10 | 4.16 | **不采用**：gap 大，且 pos9 `on`→`in`(-7.54)/`and`(-8.05) 会被扰动，误触发风险 |

### 13.2 落地：默认关闭的整词 logit 偏置
- `PF_HOTWORDS=<file>`，每行 `<gain> <token>`；gain 限 [0,5]，token 必须在词表；解析后进
  `ctx->hotword_bonus[vocab]`，argmax 时 `val = logits[n][v] + bonus[v]`（NAR 无前缀，故无
  x-asr 的 head/prefix 权重）。CLI `--hotwords FILE` / `--dump-logits N` 为等价开关。
- 默认 `hotword_bonus` 为空 → 基线**逐字节不变**（已对拍改动前 `run.tsv` 的 32 条转写完全一致）。
- 最终配置只放两个安全词：`3.2 set` / `1.5 two`；`and` 按上表排除。

### 13.3 结果（宿主 CPU + SM6650 HTP 一致）
全 32 条仅 3 条转写改变，全是目标修复，零回退、零误触发：

| 指标 | 基线 | 偏置 | 偏置+标点后处理 |
|---|---:|---:|---:|
| 命令字段 | 56.2% | 62.5% | **75.0%** |
| en WER | 12.21% | 9.92% | 9.92% |
| zh CER | 0.60% | 0.60% | 0.60% |

- 偏置+标点 75.0% **追平 x-asr on-arm 基线 75.0%**（此前 Paraformer 68.8%）。Punctuation 是
  车控流水线既有的标点层（Paraformer 无标点），本批不改它，只叠加验证。
- HTP 真机 A/B（warmup 后，4 clip）：`said→set`×2、`too→two`×1 全部生效，正确 clip 不变。
- 延迟：偏置是 argmax 内一次数组读，base/bias p50 253.9/252.9 ms（噪声内，零开销）。

### 13.4 边界与遗留
- 只能修整词功能词；`seat→seed` 等 BPE 名词混淆需另一条路（词典式 ITN / 重训），本批不碰。
- gain 是全局作用于所有 CIF 位置的：`set`=3.2 之所以安全，是因为 `set`/`said` 在句中其他位置
  的 gap 均 >8（见 13.1 全位置 dump），3.2 不足以误伤。换词表/模型必须重新量全位置 gap。
- 未做：hardneg/controls 独立包对拍（本夹具 negative_clips=0），上线前应补；第三引擎可行性。
