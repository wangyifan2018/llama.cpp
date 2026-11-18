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
- 只能修整词功能词；`seat→seed` 等 BPE 名词混淆由 §14 的前缀条件偏置处理。
- gain 是全局作用于所有 CIF 位置的：`set`=3.2 之所以安全，是因为 `set`/`said` 在句中其他位置
  的 gap 均 >8（见 13.1 全位置 dump），3.2 不足以误伤。换词表/模型必须重新量全位置 gap。
- 未做：hardneg/controls 独立包对拍（本夹具 negative_clips=0），上线前应补；第三引擎可行性。

## 14. M6 实测：BPE 词尾的条件偏置（2026-11-17，SM6650）

### 14.1 什么能修、什么不能修——由 BPE token 数决定
§13 之后残余 8 条失败里 5 条是英文名词 BPE 混淆。查词表 + dump 得出一条硬判据：

**只有当正确词与错误词的 BPE token 数相同、且只差最后一个 token 时，logit 偏置才能修。**

| 错误→正确 | BPE 路径 | token 数 | 词尾 gap | 可修 |
|---|---|---:|---:|---|
| `seed`→`seat` | `se@@ ed` → `se@@ at` | 2→2 | at vs ed = 3.26–3.46 | **是** |
| `footwall`→`footwell` | `t@@ w@@ all`(3) → `t@@ well`(2) | 3→需删 1 | well vs all = 3.86 | 否（需删 token） |
| `likes`→`lights` | `li@@ kes`(2) → `li@@ ght s`(3) | 2→需加 1 | ght vs kes = 9.82 | 否（需加 token + gap 大） |

`footwell`/`lights` 不是 logits 能解决的——需要 token 插入/删除，属另一类机制（重训或
token-level 编辑，与 NAR argmax 无关）。这是为什么文本层 ITN 也危险：它会在 `footwwell`
这种 BPE 拼接处制造新错误（实测：直接给 `w@@>well` 加成得到 `footwwell`，因为多出的 `w@@`
仍在）。因此本批**不改文本层**，只在解码层做 token 数相同的替换。

### 14.2 前缀条件偏置（prefix-conditional gain）
全局给 `at` 加成会污染所有 `at`，故新增第二种热词格式：`<gain> <prefix>><token>`，
仅当**上一个已选 token == prefix** 时才给当前位置的 target 加成。NAR 循环里维护 `prev_token`
即可，无需自回归。`se@@` 在词表里是单 token 且几乎只出现在 `seat`，规则极窄。

### 14.3 结果（宿主 CPU + SM6650 HTP 一致）
在 §13 的 `3.2 set / 1.5 two` 之上加 `3.8 se@@>at`，全 32 条仅 2 条转写改变
（seat_en03 `seed→seat`、seat_en04 `seed→seat`），零回退、零误触发：

| 指标 | §13 (M5) + 标点 | §14 (M6) + 标点 |
|---|---:|---:|
| 命令字段 | 75.0% (24/32) | **78.1% (25/32)** |
| device / state / clause | 78.6% | 83.3% |
| en WER | 9.92% | 8.40% |
| zh CER | 0.60% | 0.60% |

命令字段只 +1 条：seat_en03 的 value（`level one`）此前因 device 字段解析失败被连带丢弃，
修 seat 后整条命令通过；seat_en04 因 `massaician` 粘连仍失败。HTP 真机 A/B：两条 `seed→seat`
均生效，`footwall` clip 不受影响。延迟开销仍在噪声内。

### 14.4 边界与遗留
- 前缀条件偏置依赖"正确/错误 BPE token 数相同"，这是一条可复用的判据：先 dump 两条 BPE 路径
  和词尾 gap，再决定是否加成；token 数不同就不要碰。
- `se@@>at` 安全是因为 `se@@` 几乎唯一对应 seat；换词表要重新确认前缀的歧义度。
- 残余失败（hardneg 未验）：`footwall/likes/reevent/and` 名词粘连、`driversee peater`/
  `massaician` 双错粘连、中文「掉/调」同音、标点插错位。这些要么需重训 / token 编辑，
  要么属后处理层，且 hardneg 陈述句易误触发，不在解码偏置范围。上线前应补独立包对拍。

## 16. M7 实测：图与 arena 复用（2026-11-17，SM6650 + 宿主 CPU）

**问题**：M1 把首包 1 s 的建图/分配挪到 warmup 之后，但每个请求仍在重建图。原代码有一条注释
`#215e UAF fix: always rebuild (sched gallocr regrow frees cached buffers)`：每次请求
`assign(256 MB, 0)` 一块 arena、重建 encoder 图（设备 20 ms）、再 reset+alloc；decoder 图完全
没缓存（形状随 token 数变，设备约 10 ms）。稳态 306 ms 里约 31 ms 是这部分，且 256 MB × 2 块
arena 常驻。

**根因**：encoder 与 decoder 共用一个 scheduler，给 decoder 分配图会让 encoder 的缓冲区失效，
当时用“每次重建”换安全。正确做法是一个 stage 一个 scheduler，各自分配一次后常驻。

**改动**：
1. 新增 `pf_stage_cache{ sched, ctx, gf, keys, meta, hits, misses }`，encoder / decoder 各一份；
   命中条件是 `T_lfr` 与 `(T_lfr, N_tokens)` 精确相等，命中时只上传输入并计算。
2. 两个 scheduler 分开创建，互不失效；只在 miss 时 reset+alloc。
3. arena 从 256 MB 每次 `assign` 改为 32 MB 惰性 `resize`（只建一次）；tensor 元数据实测只占
   1–2 MB（`no_alloc=true`，激活在 scheduler 缓冲里）。
4. 新增 `paraformer_cache_stats()` 与 `PF_CACHE_STATS=1`，lap 增加 `enc graph cached` / `enc graph miss`。

**结果（设备，同一段 2.65 s 音频连跑三次，warmup 后）**：

| 指标 | 改动前 | 改动后（cache hit） |
|---|---:|---:|
| enc graph build | 20 ms | 3.6 ms（含 lap 开销） |
| enc sched alloc | 2.6 ms | 0.03 ms |
| enc compute | 235 ms | 238.6 ms |
| 单请求 wall | 306 ms | 280.4 ms（−8.4%） |
| 峰值 RSS (VmHWM) | ~700 MB | 252 MB |

**32 条车控夹具（每条长度不同，命中率低）**：设备 p50/p95/p99 = 280 / 367 / 373 ms（改前
290 / 380），32/32 转写与记录的 `run.tsv` 逐条一致；cache 统计 `enc hit=3 miss=29,
dec hit=1 miss=31`。

**边界与遗留**：
- 精确按长度缓存，对“长度每次都不同”的流量命中率很低（夹具 3/32），平均只有 3–4%；同长度重复
  （同一句话、同一 VAD 段重放）才有 8% 以上。
- 提高命中率要做 T 分桶 + 注意力 mask + FSMN 边缘复制，padding 会带来与帧数成正比的额外计算
  （约 5 ms/帧）。按当前 235 ms/44 帧，分桶最多净赚约 10 ms，优先级低于量化。
- arena 从 512 MB 降到 64 MB 是本批最大的副作用收益：双引擎服务从约 1.4 GB 降到约 500 MB，
  为第三引擎或标点模型腾出内存。
- decoder 按 `(T, N)` 缓存，长度变化即 miss；N 分桶会改变尾部 token 的 logits（FSMN 居中卷积），
  必须过转写对拍才能上，本批不做。

## 17. M8 实测：服务化加固（2026-11-17，SM6650）

**问题**：M3/M4 的服务端只有一条无界 FIFO 和两个 worker：没有队列上限、没有超时、没有
取消、没有自 warmup，也分不出“排队”和“推理”。4 路 P95 740 ms 里到底排了多少，之前只能猜。

**改动（`server.cpp`）**：
1. `--queue N`（默认 32）：队列满时立即回 `busy: queue full`，不让连接挂着等。
2. `--timeout-ms T`（默认 15000）：请求在队列里等超过 T 就丢弃并回 `queue timeout`，把引擎
   留给新请求；推理一旦开始不可中断，所以超时只作用于排队阶段。
3. **客户端断开即取消**：worker 动手前用 `recv(MSG_PEEK|MSG_DONTWAIT)` 检查对端是否已关闭，
   关闭则跳过推理（回 `client closed`）。
4. **自 warmup**：没给 `--warmup` 时用 1.2 s 静音在 accept 前预热一次（形状不影响一次性开销），
   `--no-warmup` 可关。
5. `PF_TIMING=1` 打 `queue_ms / infer_ms / engine`，客户端指标可分解。

**真机结果（2 引擎，4 路齐发，N=32，逐条一致）**：

| 指标 | 值 |
|---|---|
| 客户端 p50 / p95 / p99 | 494 / 705 / 760 ms |
| 服务端 queue p50 / p95 | 134 / 362 ms |
| 服务端 infer p50 / p95 | 301 / 390 ms |

对比 §12 的 516 / 740：客户端 P95 降约 5%，并且第一次能把尾延迟拆成“排队 362 + 推理 390”。
宿主 CPU 上的功能验证：`--timeout-ms 1` 时排队请求返回 `queue timeout`、先到的正常返回；
`--queue 1` 时 2/4 立即收到 `busy: queue full`（0.3 ms 返回）。

**边界**：超时只覆盖排队，正在跑的推理不可中断；要打断推理得在 `paraformer_transcribe` 里加
取消点，本批不做。断开检测依赖 TCP 半关闭，客户端进程被 kill -9 时也要等内核发 RST 才生效。

## 18. M9 实测：质量鲁棒性、标点口径与热词（2026-11-17）

### 18.1 标点层：仓库里没有，朴素的规则插入会变差

M0 里“仅标点后处理 68.8%”依赖的那层代码并不在仓库里（`eval_vehicle` 只按标点切子句）。
本批把它补成可复现的 `examples/paraformer/tools/pf_clauses.py`（在子句起始词前插一个逗号），
结果**变差**：

| 口径 | command | device | state | value |
|---|---:|---:|---:|---:|
| raw | 56.2% | 69.0% | 64.3% | 77.8% |
| 规则插入逗号 | 46.9% | 69.0% | 59.5% | 50.0% |

32 条里 7 条被改动，其中 2 条（`空调别关，把后排出风关掉`、`别关氛围灯，把顶灯关掉`）真的
是两句，规则修对了；另外 5 条被切错，例如 `把左前车窗打开一半` 被切成 `把左前车窗，打开一半`。
结论：**“把 X 打开”这类单句里也出现起始词，靠词表插入不可行**。要拿到 68.8% 那档数字，
需要真正的标点模型（FunASR CT-Transformer 一类）或语义感知的切分；而更便宜的方向是让下游
语义解析器支持无标点输入（按设备词出现位置切），本批两种都没做，只留下这个反例与脚本。

### 18.2 热词开关对照（同一 32 条，raw 口径）

| 口径 | command | device | value | zh CER | en WER |
|---|---:|---:|---:|---:|---:|
| 偏置关 | 56.2% | 69.0% | 77.8% | 0.60% | 12.21% |
| 偏置开（`hotwords_vehicle.txt`） | **65.6%** | 73.8% | 88.9% | 0.60% | **8.40%** |

三项规则（`3.2 set` / `1.5 two` / `3.8 se@@>at`）在无标点口径下带来 +9.4 点命令字段、
en WER −3.8 点，中文零变化、零回退。这与 §13 的“62.5%（仅 set/two）”一致，多出来的是
`se@@>at`。

### 18.3 座舱噪声敏感性（降噪收益上限）

用 `tools/mix_snr.py` 按电平精确混合座舱噪声（噪声功率 = 语音功率 / 10^(SNR/20)），
同一 32 条、同一权重：

| SNR | zh CER | en WER | command | device | value |
|---|---:|---:|---:|---:|---:|
| 干净 | 0.60% | 12.21% | 56.2% | 69.0% | 77.8% |
| 15 dB | 0.00% | 11.45% | 56.2% | 66.7% | 77.8% |
| 10 dB | 0.60% | 16.79% | 50.0% | 59.5% | 66.7% |
| 5 dB | 7.83% | 29.01% | 31.2% | 45.2% | 44.4% |

读法：这个模型对 15 dB 以内的噪声几乎免疫（中文 CER 仍为 0），10 dB 开始掉英文，5 dB 全面塌陷。
也就是说**前端降噪只要能把这 5-10 dB 找回来，就能守住命令字段**；反过来，不降噪时
5 dB 场景下命令字段只剩 31%，任何解码层优化都补不回来。这是“AI 降噪”在本项目里的量化依据。

### 18.4 硬负样本（陈述句误触发）

16 条陈述句（`vehicle_hard_negatives.tsv`，Tingting 合成，8 中 8 英）：

- **false_trig 0/16**，wrong 0，zh CER 0.00%，en WER 15.56%；
- **near_miss 12/16**：句子里同时出现设备词和状态词（`座椅上有一本书`、`空调滤芯该换了`），
  下游语义解析必须区分“陈述”与“命令”，否则这 12 条就是潜在误触发池。

### 18.5 边界

- 噪声是合成混合，不是真实车内录音；SNR 按全段 RMS 计算，未做语音活动对齐。
- 硬负样本是合成音色，未覆盖真实说话人和方言。
- 标点实验只改了文本层，未接模型；hotword 只覆盖整词/等长 BPE 词尾。

## 19. M10 实测：说话人鲁棒性、热词覆盖边界、量化受阻（2026-11-17）

### 19.1 换说话人比任何解码调优都重要

同一 32 条文本、同一权重、同一评测器，只换合成音色（中文 Flo、英文 Daniel(en_GB)）：

| 音色 | 偏置 | zh CER | en WER | command | device | value |
|---|---|---:|---:|---:|---:|---:|
| Tingting + Samantha | 关 | 0.60% | 12.21% | 56.2% | 69.0% | 77.8% |
| Tingting + Samantha | 开 | 0.60% | 8.40% | 65.6% | 73.8% | 88.9% |
| Flo + Daniel | 关 | **34.34%** | **22.90%** | **18.8%** | 38.1% | 27.8% |
| Flo + Daniel | 开 | 34.34% | 18.32% | 21.9% | 42.9% | 27.8% |

- 只换中文音色，CER 从 0.6% 掉到 34.3%，命令字段从 56.2% 掉到 18.8%；热词偏置对这个退化
  几乎没有作用（中文 CER 不变，命令只 +3 点，全部来自英文）。
- 英文换 en_GB 口音，WER 12.2% → 22.9%，偏置仍能修 −4.6 点（说明偏置修的是“模型知道但没选”
  的混淆，不是口音导致的声学错误）。
- 结论：**当前夹具上的 0.6% CER 是 Tingting 这一把音色的成绩，不能当作模型能力上界**。
  任何“上线”判断都必须用真实座舱录音或至少多音色；此前 x-asr 在同一把音色上也测到
  Flo 55%/Tingting 8% 的差异，方向一致。

### 19.2 热词机制的覆盖边界（复核 §14.1 判据）

按 §14.1 的判据（正确/错误词 BPE token 数相同、只差末 token）复核当前偏置表的覆盖：

| 残余错误（Tingting，偏置后） | BPE 路径 | 能否用现有机制修 |
|---|---|---|
| `footwall` → `footwell` | 3 → 2 | 否（要删 token） |
| `likes` → `lights` | 2 → 3 | 否（要加 token，且 gap 9.82） |
| `reevent` → `rear vents` | 1 → 2 | 否（要插 token） |
| `said` → `set` | 1 → 1 | 已修 |
| `too` → `two` | 1 → 1 | 已修 |
| `seed` → `seat` | 2 → 2，末 token 不同 | 已修（前缀条件） |

也就是说，偏置这一类**解码期 logit 干预**已经吃干净了：剩下 3 类都需要 token 插入/删除，
属于“词表级 / 上下文偏置”问题，不是增益能解决的。要做“热词模型”，方向应是
（a）上下文感知的 shallow fusion（给每个 VAD 段一个候选词表，按 prefix trie 约束解码），
或（b）重训/微调带 domain biasing 的 decoder；两者都需要新权重或训练数据，超出本仓库范围。

### 19.3 Q4_K 量化 A/B：本机无外网，未执行

`curl https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main/paraformer-zh-q4_k.gguf`
在本轮连续两次 75 s 超时（`Failed to connect to huggingface.co port 443`），仓库内也没有
`llama-quantize` 可用的转换路径（Paraformer 是自定义 GGUF 布局，转换器写死 zh 超参）。
因此 Q4_K 只留下计划与预期上界：

- 预期内存：227 MB → 123 MB/引擎，双引擎 454 → 246 MB（配合 §16 的 arena 缩减，三引擎可行）。
- 预期延迟上界：encoder 里 MUL_MAT+ADD 占 62.7%，Q4_K 权重字节减半，若 DSP 走 int4 路径
  且仍 compute-bound，理论上限约 −20~30% encoder 时间（235 ms → 165~190 ms）。
- 验收门：先跑 32 条 CER/WER/command 与逐条转写对拍，退化就停；再报 p50/p95 与 VmHWM。
- 复现命令（有网时）：
  ```sh
  curl -L -o paraformer-zh-q4_k.gguf \
    https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main/paraformer-zh-q4_k.gguf
  ./build-x-asr/bin/llama-paraformer paraformer-zh-q4_k.gguf fixtures-tingting/*.wav
  ```

## 20. M11 实测：前端降噪到底值不值（2026-11-17，Tingting 32 条）

**先修正 §18.3 的一个事实错误**：当时用的“噪声”文件 `cabin-asr-probe/zh-noise-10db.wav`
其实是一段**带噪语音**（模型把它转写成 `请导航到上海虹桥机场`），所以那一节测到的是
“语音叠语音（babble）”，不是稳态座舱噪声。两种干扰必须分开看，下面重做。

方法：`tools/mix_snr.py` 按电平混合（噪声功率 = 语音功率 / 10^(SNR/20)），
`tools/denoise_wiener.py` 是单通道维纳降噪原型（512 点 STFT / 10 ms 帧移，最小统计噪声估计 +
判决引导先验 SNR，增益下限 0.08，重叠相加）。降噪前后跑同一权重、同一 32 条，
并用干净参考算真实 SNR 增益。

**稳态粉红噪声（1/f + 低频路噪）**：

| SNR | 原始 zhCER/enWER/command | 降噪后 | 实测 SNR 增益 |
|---|---|---|---:|
| 15 dB | 0.60% / 10.69% / 56.2% | 0.60% / 10.69% / 56.2% | +1.69 dB |
| 10 dB | 0.00% / 12.98% / 53.1% | 0.00% / 12.98% / 53.1% | +1.48 dB |
| 5 dB | 0.00% / 14.50% / 53.1% | 0.00% / 17.56% / 50.0% | +0.96 dB |

**白噪声（全带平坦）**：

| SNR | 原始 zhCER/enWER/command | 降噪后 | 实测 SNR 增益 |
|---|---|---|---:|
| 15 dB | 2.41% / 22.90% / 40.6% | 1.20% / 24.43% / 40.6% | +5.55 dB |
| 10 dB | 3.01% / 32.06% / 40.6% | 3.01% / 37.40% / 34.4% | +4.36 dB |
| 5 dB | 4.22% / 55.73% / 31.2% | 4.82% / 58.02% / 31.2% | +3.70 dB |

**结论（与我上一轮的判断相反，按数据改口）**：

1. **SNR 增益不等于识别增益。** 白噪声上降噪实打实拿到 +3.7~+5.6 dB，但命令字段在 15/5 dB
   一点没变、10 dB 反而掉 6 点；粉红噪声上几乎没增益、5 dB 还掉 3 点。
2. **这个东西怕的是“说话人干扰”，不是稳态噪声。** 稳态噪声即使在 5 dB 也只让 CER 从 0.6% 到 0
   或 4.22%（白噪声）；而 §18.3 的语音叠语音在 5 dB 直接把 CER 打到 7.83%、命令掉到 31.2%。
   单通道维纳滤波对另一路说话人本来就无能为力。
3. 所以**普通 DSP 降噪不值得进 runtime**：不写进 C++，不设默认开关。要提升噪声鲁棒性，
   正确的靶子是 babble 与混响，工具是**多麦/波束成形**或**目标说话人提取（learned TSE）**，
   并且验收必须用命令字段/CER，不能只看 SNR 或 PESQ。
4. 评估链路已固化：`mix_snr.py`（按电平混）+ `denoise_wiener.py`（参考降噪器 + 真实 SNR 增益），
   任何新降噪模型都可以用同一口径对照，避免“SNR 提升但业务没变”的伪收益。

**边界**：噪声是合成的（粉红/白），没有真实车内录音；SNR 用全段 RMS，未做 VAD 对齐；
只测了一个降噪器（维纳 + 最小统计），没有测学习型降噪。

## 21. M12 实测：跨 token 数的域内纠错（span rewrite）

**问题**：§14.1 的判据说明 logit 增益只能修“token 数相同”的混淆。残余英文错误
`footwall->footwell`（5 token -> 4）和 `likes->lights`（2 -> 3）都需要增删 token，
任何按位置的增益都做不到。

**做法**：解码后增加一遍 span 改写（默认关闭，配置在 `hotwords_vehicle_replace.txt`）：

```text
R <margin> <wrong1+wrong2+...> => <right1+right2+...>
```

命中条件：argmax 结果里出现与 wrong 完全一致的连续 token 串，且右侧序列的“每位置平均 logit”
不比左侧低出 `margin` 以上；命中后用右侧序列替换该 span（长度可变）。token 用文本比较而不是
id 比较（词表里有重复字符串）。没有 R 行时输出与改前逐字节一致（32/32 已对拍）。

**实测（Tingting 32 条，设备与宿主一致）**：

| 配置 | command | device | value | zh CER | en WER | 改写次数 |
|---|---:|---:|---:|---:|---:|---:|
| 仅增益（`hotwords_vehicle.txt`） | 65.6% | 73.8% | 88.9% | 0.60% | 8.40% | 0 |
| 增益 + R margin 2.0 | 68.8% | 76.2% | 94.4% | 0.60% | 7.63% | 1 |
| 增益 + R margin 5.0 | **71.9%** | 78.6% | 94.4% | 0.60% | **6.87%** | 2 |

- `margin 5.0` 会同时改写 `footwall->footwell`（实测票差 1.13）和 `likes->lights`（票差 4.59）。
- 硬负样本 16 条：误触发仍为 **0/16**，zh CER 0%，en WER 15.56% -> 13.33%（增益顺带修了陈述句里的
  `set/two`），near_miss 12 -> 13。
- 延迟：宿主单条均值 91.8 -> 94.9 ms（一次运行的差，属噪声量级；改写本身是一次线性扫描）。

**边界（必须写在配置旁边）**：

1. 这是一张**域内强制纠错表**，不是软偏置。margin 5.0 意味着“即使模型对错词更自信也改”，
   只有在该领域正确词唯一时才安全（车内说 `footwall`、`ambient likes` 无意义）。
2. 规则要逐条量票差再写，票差大的必须显式接受风险；换词表/换模型后票差会变，必须重测。
3. `reevent->rear vents` 没做：词表里没有 `vent`/`vents` 的可用切分，右侧序列写不出来；
   这类需要“插入一个词表里不存在的词”的错误，只能靠上下文偏置解码或重训。
4. 长语音路径（`out_tokens`）暂不启用改写，避免改写后的 token 时间戳错位。


## 22. 语义层补分隔的两种做法都不成立（2026-11-17）

§18.1 的规约插入失败后，又试了第二种更贴近语义的做法：在“第二个设备词出现的位置”切子句
（`--unpunctuated`，在 `eval_vehicle.py` 里实现后已回退，不入库）。结果同样变差：

| 口径 | command | device | state | value |
|---|---:|---:|---:|---:|
| raw（不补分隔） | 56.2% | 69.0% | 64.3% | 77.8% |
| 按起始词插逗号（§18.1） | 46.9% | 69.0% | 59.5% | 50.0% |
| 按设备词切分（本节） | 21.9% | 40.5% | 33.3% | 33.3% |

原因相同：`把左前车窗打开一半` 只有一个设备词但会被切成 `把左前` + `车窗打开一半`，
整条命令失效。**分隔与解析必须联合求解**：把所有设备词出现位置作为候选切点，用动态规划找
“能解析成完整命令的子句数最多”的切分，而不是先切后解析。这属于语义层的工作，不应塞进 ASR；
本仓库不改 `eval_vehicle.py` 的默认口径，等语义层实现 DP 分段后再用同一 32 条对照。

结论：**Paraformer 不输出标点这件事，短期只能靠下游语义层吸收**；在语义层完成联合分段之前，
所有“命令字段”数字都必须注明用的是哪种口径（raw / 补分隔）。

## 23. M13 实测：三引擎（2026-11-17，SM6650）

**动机**：§16 把每进程的 arena 从 512 MB 压到 64 MB 后，内存不再是一引擎一份 227 MB 的死结；
§12 的双引擎（HTP+CPU）在 4 路齐发时 P95 还有 705 ms。本批把 `--engines` 放开到 4
（1 个设备引擎 + 最多 3 个 CPU 引擎），实测三引擎是否值得。

**改动**：`server.cpp` 的引擎规格从“固定 HTP + 可选 1 个 CPU”改成“设备引擎 + (n-1) 个 CPU”，
上限 4。其余调度不变（共享 FIFO、谁空闲谁取）。

**结果（4 路齐发，N=32，2.0–2.7 s 片段，逐条一致）**：

| 引擎数 | 客户端 p50 | p95 | p99 | 服务端 queue p50/p95 | infer p50/p95 | 进程 RSS |
|---|---:|---:|---:|---:|---:|---:|
| 2（§17） | 494 | 705 | 760 | 134 / 362 | 301 / 390 | ~500 MB |
| 3 | **447** | **626** | 681 | 1 / 315 | 363 / 503 | 933 MB |

- P95 降 11%（705 -> 626），中位数降 10%；排队几乎消失（p50 1 ms），代价是单条推理变慢
  （infer p50 301 -> 363），因为三个引擎共用同一组 CPU 核，HTP 引擎的 host 侧也被挤。
- 内存 933 MB 已经超过“约 800 MB 空闲”，实际在吃 swap；本批只跑了 8 轮，没有做长时间热压测，
  **上线前必须补 30 分钟以上的热与 swap 观察**。
- 转写 32/32 一致，无错误返回。

**边界与后续**：
- 三引擎的收益来自“排队消失”，不是算力变多；如果 VAD 段本来就错峰（§11 的 300 ms 错峰），
  双引擎已足够，不必多花 430 MB。
- 真正想再降 P95，应换 Q4_K（123 MB/引擎）再做三引擎，内存回到 ~600 MB；该实验因本机无外网
  仍未做（§19.3）。
- 4 引擎（1+3 CPU）本批没测，预计 CPU 争抢会吃掉收益。

## 24. M14 实测：Q4_K 量化（2026-11-17，离线再量化）

**通路**：huggingface 不可达，ModelScope 可达但没有 `cstr/paraformer-zh-GGUF`（只有官方
PyTorch 权重 880 MB；仓库转换器只输出 F16，做不出 Q4_K）。于是新增
`examples/paraformer/tools/requant.cpp`：用 ggml 自己的 `ggml_quantize_chunk` 把现有 Q8_0
GGUF 离线再量化，未量化的张量**保持原类型原字节**（1-D 的 cmvn/norm 是 F32，入口投影 560 宽
不是 256 的倍数，留 F16）。

```sh
cmake --build build-x-asr --target llama-paraformer-requant -j
./build-x-asr/bin/llama-paraformer-requant paraformer-zh-q8_0.gguf paraformer-zh-q4_k.gguf Q4_K
# requant: 958 tensors, 283 quantized q4_K, 675 kept F16, 226.3 MB -> 122.6 MB
```

**质量（32 条，宿主 CPU）**：逐条只有 4 条不同，两条更好（`a->the`、`too->two`）、两条更差
（`dome->dme`、`massaician->massasian`）；聚合 **command 59.4% vs 56.2%**、en WER 11.45% vs
12.21%、zh CER 都是 0.60%。属于量化噪声，不是系统性退化。

**延迟与内存（设备）**：

| 权重 | 引擎 | p50 | p95 | 进程峰值 RSS |
|---|---|---:|---:|---:|
| Q8_0 | HTP | **277** | **370** | 252 MB |
| Q4_K | HTP | 324 | 443 | **146 MB** |
| Q8_0 | CPU | 334 | 468 | — |
| Q4_K | CPU | 387 | 509 | — |

**结论：Q4_K 在这颗 DSP 上是“省内存、费时间”。** HTP 上慢 17%（277 -> 324），CPU 上慢 16%，
但内存降 42%。Q8_0 走的是 hvx-tiled int8 原生路径，Q4_K 没有同等的 int4 快路，收益只剩体积。

**与三引擎组合**（4 路齐发，N=32）：

| 权重 | 引擎数 | p50 | p95 | RSS |
|---|---:|---:|---:|---:|
| Q8_0 | 2 | 494 | 705 | ~500 MB |
| Q8_0 | 3 | **447** | **626** | 933 MB |
| Q4_K | 3 | 490 | 723 | 615 MB |

Q4_K 的单条变慢正好抵掉第三引擎的排队收益（490/723 对 494/705），所以**默认仍是 Q8_0 双引擎**；
只有在内存/热受限、必须多引擎时才用 Q4_K + 3 引擎（同等尾延迟下多 115 MB 缓冲）。

**边界**：Q4_K 是从 Q8_0 再量化的（多一次舍入）；没有测 Q5_K/Q6_K；没有做长时间热压测；
HTP 上 Q4_K 是否走了部分回退未从日志确认（只能从“HTP 324 vs CPU 387”判断仍有 DSP 参与）。

## 25. M15 实测：每请求时间线（2026-11-17，SM6650）

把 feat / CIF / decoder / logits 回读都打上 lap 点后，稳态一条 2.70 s 音频（warmup 后）：

| 阶段 | 累计 | 增量 | 占比 |
|---|---:|---:|---:|
| feat（fbank+LFR+CMVN） | 3.09 | 3.09 | 1.0% |
| enc graph（cache hit）+ sched | 3.34 | 0.25 | 0.1% |
| encoder compute | 244.83 | 241.5 | **81.2%** |
| CIF（host conv1d+linear+sigmoid） | 254.22 | 9.4 | 3.2% |
| decoder compute | 296.95 | 42.4 | **14.3%** |
| logits 回读（8404 x N f32） | 297.28 | 0.33 | 0.1% |
| 合计 wall | 297.7 | | |

两个结论：

1. **logits 回读可以排除**：只有 0.33 ms，之前怀疑的 800 KB 跨设备拷贝不是瓶颈，不需要把
   argmax 搬到 DSP。
2. 除 encoder 外只剩两块：**decoder 42.4 ms** 和 **CIF 9.4 ms**。decoder 的 16 个 block 在
   N≈26、T≈45 下本来就要 ~0.87 GMAC 的 FFN，加上 cross-attention，42 ms 与“compute-bound”
   一致，没有明显浪费。
3. CIF 的 9.4 ms 是 host 上的 conv1d(512x512,k=3) + linear，约 35 MMAC 走标量路径；把它并进
   encoder 图（alpha 在 DSP 上算，只回传 T 个 float）理论上能省 8 ms 左右（2.7%），
   是本轮之后剩下的唯一低风险小项。

至此，单请求 297 ms 里 encoder 占 241 ms，与 §10 的结论一致：**除非换权重（量化/更小模型），
runtime 层已经没有超过 5% 的杠杆**。

## 26. M16 实测：为什么 4-bit 在这颗 DSP 上更慢（2026-11-17）

用 `GGML_HEXAGON_PROFILE=1` 抓同一条 2.70 s 音频（warmup 后）的单算子耗时，按权重类型聚合
（encoder + decoder，共 498 个权重矩阵乘）：

| 格式 | 权重位宽 | 文件大小 | 498 个 matmul 总时间 | 单个均值 | 相对 Q8_0 |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 4.5 | 122.6 MB | 398.1 ms | 799.5 us | **+25%** |
| Q4_K | 4.5 | 122.6 MB | 421.2 ms | 845.9 us | **+32%** |
| Q5_0 | 5.5 | 148.5 MB | 379.4 ms | 761.8 us | +19% |
| Q6_K | 6.6 | 176.1 MB | 380.2 ms | 763.5 us | +19% |
| Q8_0 | 8.5 | 226.3 MB | 319.4 ms | **641.3 us** | 基线 |

同一份 profile 里，**非权重算子完全不变**：attention 的 `MUL_MAT`（无权重）在三种格式下都是
33.5 ms、`CONV_2D_DW` 都是 24.8 ms、`CONT` 都是 12.4 ms、`SOFT_MAX`/`RELU`/`NORM` 逐项一致。
也就是说，慢的部分精确地落在 498 个“带权重的 matmul”上，其他一概不动。

**机制**：profile 的 dtype 字段显示权重在设备上保持原生量化类型参与计算（`q8_0 x f32`、
`q4_K x f32`、`q4_0 x f32`），不是先整体反量化成 f16。于是：

1. q8_0 的每个元素就是一个 int8，kernel 可以直接做 int8 点积，没有解包步骤；
2. 4-bit 每个字节装两个值，kernel 必须先做移位/掩码解包，再乘 4-bit 子块的 scale（K-quant 还有
   6-bit 的 scale/min），解包开销随 M 行数摊薄；
3. 编码器每个 matmul 的 M 只有 45 帧，解包摊不薄，所以 4-bit 省下的带宽抵不过解包；
   decoder 里 M 更小，惩罚更重（`MUL_MAT+ADD` 224 -> 284 us，cross-attn 的 `MUL_MAT_NX`
   936 -> 1262 us，+35%）。
4. 5-bit/6-bit 反而比 4-bit 快（+19% vs +25/32%）：它们的解包是“补一个 bit 平面到 int8”，
   比 4-bit 的双值拆分更便宜，说明瓶颈确实是解包而不是带宽。

**结论**：在这颗 v73 上，**Q8_0 就是延迟最优解**，位宽再低只会更慢；没有 runtime 技巧能把这个
惩罚找回来（换 tile 布局、换 repack 都改不了 kernel 的解包成本）。4-bit 只是内存杠杆：

| 目的 | 选择 | 代价 |
|---|---|---|
| 最低延迟（默认） | Q8_0 | 226 MB / 641 us |
| 省内存但要质量与速度平衡 | **Q5_0** | −34% 内存，+19% 延迟 |
| 内存最紧 | Q4_K | −46% 内存，+32% 延迟 |

Q4_0 与 Q4_K 速度接近（+25% vs +32%），但 Q4_K 的精度结构更好，内存受限时选 Q4_K。

**新测试集（TTS，未见文本）**：另生成 24 条座舱命令（`tools/holdout_texts.tsv`，12 中 12 英，
覆盖车窗/空调/座椅/车灯/导航/消息/音乐/雨刮/后备箱，含否定与数字），Tingting + Samantha。三种权重
在上面逐条差异都在噪声内：

| 权重 | zh CER | en WER |
|---|---:|---:|
| Q8_0 | 3.25% (4/123) | 13.48% (12/89) |
| Q4_K | 3.25% | 12.36% |
| Q4_0 | 3.25% | 13.48% |
| Q5_0 | 4.07% | 13.48% |

注意 holdout 的 CER 3.25% 明显高于调参集的 0.60%：**调参集是乐观的**，新句子/新说法会掉，
这与 §19 的“换音色掉到 34%”是同一类结论，量化反而不是主要误差来源。

**边界**：只测了一条 2.70 s 音频的 profile（encoder 主项稳定，但 decoder 的百分比会随 N 变）；
没有测 imatrix 辅助量化；holdout 仍是合成音色，没有真实座舱录音。

### 26.1 还试过哪些旋钮（都不行）

| 旋钮 | 结果 | 说明 |
|---|---|---|
| `GGML_HEXAGON_MM_SELECT=1` | 无变化 | 该值只影响 HMX 优先，v73 没有 HMX，2 与 1 等价 |
| `GGML_HEXAGON_NHVX=1` | p50 279 -> **469 ms**（+68%） | 少一半 HVX，符合预期，默认 2 是对的 |
| 换 tile repack | 不适用 | q4_K/q4_0 已有 tiled repack，慢在解包不在布局 |
| 混合精度（FFN 用 Q8_0、其余 4-bit） | 不建议 | 4-bit 只占 37% 的 matmul 时间，省 26 MB 却要 +5% 总时间 |

结论：**这颗 DSP 上没有能救 4-bit 的旋钮**，kernel 的解包成本是编译期决定的；要么接受 Q8_0 的
内存，要么用 Q5_0 换内存并接受 +19%。

唯一还有理论上限的方向是**让 M 变大**（两条流共用一个 encoder 图，M 45 -> 90），把每个权重 tile
的加载摊薄到两倍帧数上。这条要改 SANM encoder 的 batch 维与 FSMN/CIF，属于结构改动，本批不做；
按 x-asr 固定车道探测的经验（B=2 省 21%、B=4 省 31%），Paraformer 的 encoder 是 compute-bound，
预期收益低于 x-asr，建议先用一个小实验（固定两条流、只测 encoder）验证再决定。

## 27. M17 实测：既要组批又要 Q4，先看 M 的标度律（2026-11-17）

两个问题各自做完实验后，结论都落在同一条标度律上。

### 27.1 组批（把两条流塞进一个 encoder 图）没有收益

不需要先写批处理图：只要量出"权重 matmul 的时间随 M（帧数）怎么变"，就能判断组批有没有空间。
同一份 profile 里按 tensor 形状把 498 个权重 matmul 按 M 分组（q8_0，设备 HTP）：

| M（帧） | 单个 matmul 均值 |
|---:|---:|
| 5 | 92.2 us |
| 13 | 231.8 us |
| 20 | 337.3 us |
| 41 | 533.2 us |
| 45 | 746.9 us |
| 167 | 2711.4 us |

最小二乘：**16.31 us/帧，截距 −24 us，R2 = 0.9962**。截距为零说明**没有固定开销可摊薄**；
把两条 45 帧的流合成一次 M=90 的 op，预测 1444 us，而两次 M=45 串行是 1420 us，**组批反而慢 0.6%**。
q4_0 同样（斜率 20.75 us/帧，截距 −45 us），组批预测 1822 us 对串行 1777 us。

为什么 x-asr 组批能省 21–31% 而这里不能：x-asr 的 encoder 是 dispatch-bound（2449 个算子、
每个算子只有几 us，固定调度成本占 77 ms 里的 17 ms），组批摊薄的是**每个算子的固定成本**；
Paraformer 每个 matmul 平均 641 us，dispatch 只占约 1%，没有固定成本可摊。这也是为什么
M 从 13 到 167 严格线性、截距为负。

结论：**这条方向判死，不再写批处理图。**

### 27.2 Q4 为什么在别的模型上更快，在这里不可能

Q4 的惩罚随 M 变化（同一个 profile 口径）：

| M | q8_0 | q4_0 | Q4 惩罚 |
|---:|---:|---:|---:|
| 5 | 92.2 us | 99.8 us | +8% |
| 7 | 129.6 | 127.2 | −2% |
| 13 | 231.8 | 279.5 | +21% |
| 45 | 746.9 | 934.9 | +25% |
| 167 | 2711.4 | 3434.9 | +27% |

- **越短越接近打平，越长越亏**：惩罚是"每元素解包"的量，随 M 线性累加；只有在 M≈5–7 时才近似打平。
- 别的模型（自回归解码）Q4 更快，是因为它们的 matmul 是 **M=1 的矩阵-向量乘**：每个权重只被
  用一次，时间由权重字节数决定（带宽 bound），4-bit 把字节砍一半就几乎把时间砍一半。
- Paraformer 是非自回归：一次 matmul 覆盖全部帧，M=13–167，算术强度约 47 MAC/byte，是
  **compute-bound**；4-bit 只减少字节、不减少乘加，反而多出解包指令，所以必然更慢。
- 方向上唯一能让 Q4 变快的是把 matmul 拆成 M=1 的逐帧乘——那会把算子数放大 45–167 倍并丢掉
  跨帧的 SIMD 归约，整体只会更慢。**改图救不了 Q4。**

由此给出判据：**权重 matmul 的 M 决定 Q4 是赚是亏**。M≈1（自回归解码）用 Q4；M≥10（整句编码）
用 Q8_0。Paraformer 属于后者，Q8_0 是延迟最优，Q4 只在"内存或带宽是硬约束"的部署里才值得。

**边界**：标度律只覆盖 q8_0/q4_0 与这条 2.7 s/10 s 夹具；没有测 M=1 的极端（模型里不存在）；
组批结论来自标度律外推，没有真的写批处理图验证——如果将来改了 kernel 让固定成本变高，这条要重测。

## 28. M18 实测：段级上下文偏置（PF_CONTEXT）

**问题**：热词增益只能修"词表里已经存在、且 token 数相同"的混淆；POI/联系人这类**未登录实体**
要么整词不在输出里，要么第一字对、后面错。需要按"这一趟行程可能出现的词"来偏置。

**实现**：`PF_CONTEXT` 指向一个词表文件，每行 `<gain> <phrase>`。短语按模型词表做
**最长匹配分词**（中文按字、英文按词表片段，也可以写成 `a+b+c` 显式指定 token），存进前缀树。
解码时维护"当前活跃的树节点"：某个短语已经匹配到第 k 个 token，就给它第 k+1 个 token 的 logit
加 `gain`；短语首 token 由 `PF_CONTEXT_START` 单独控制（**绝对 logit 值**，默认 0）。
没有 `PF_CONTEXT` 时行为与之前逐字节一致。

**实测（10 条 TTS 命名片段，Tingting）**：

| 配置 | 名字正确的片段 |
|---|---:|
| 无上下文 | 6/10（王子涵✗、李牧尘✗、put on airport✗、a doll✗） |
| gain 4.0 | 6/10（李牧尘→李沐尘，仍差一字） |
| **gain 8.0** | **8/10**（王梓涵✓、李沐宸✓；英文两条仍错） |

**回归检查**：32 条车控夹具 + 热词 + span rewrite，开上下文后 command **71.9%（不变）**、
zh CER 0.60%（不变）、en WER 6.87%（不变）；延迟无变化（每个位置多一次小表遍历）。

**踩到的两个坑（都写进代码注释）**：

1. **首 token 偏置必须小且是绝对值**。最初把 `PF_CONTEXT_START` 当倍率用（3.0 × gain 4 = 12 logits），
   结果整句变成 `张陆张ad ad徐p…`，命令字段 0%。改成"根节点按 max 合并 + 绝对值 ≤ 2"后恢复正常。
2. **英文 OOV 名字修不了**：`pudong` 在模型里被切成 `put|on`，与词表里的 `pudong` 分词完全不同，
   续接偏置无从发力。这类要用 `R` 改写规则显式给出错误/正确的 token 序列。

**边界**：只在 TTS 上验证；真名/方言名的分词与发音差异更大，需要真实数据；短语表要按行程/用户
动态生成（POI 与联系人本来就会变），不能写死在代码里。

## 29. M19：分词器真相与词表驱动的编码器

**先查事实**：把 GGUF 的 21 个 KV 全部 dump 出来，只有 `tokenizer.tokens`（8404 个字符串），
**没有 merges、没有 scores**；从 ModelScope 下载的原始 `tokens.json` 也只是一个扁平列表。
也就是说这个模型的 BPE **合并表根本不在任何可获得的文件里**，无法做"真正的" BPE 编码。

**实现**（`pf_tokenize_phrase`）：词表驱动的动态规划分词
- 词中间的位置优先尝试 `X@@` 续接形式，词尾用不带 `@@` 的普通 token；
- 目标：**piece 数最少**，并列时取更长的 piece；
- 中文按字切分（每个字就是一个 token），**精确**；
- 英文只能近似——模型自己的切分由不可见的合并顺序决定。

实测三种目标的差别（同一批词）：

| 词 | 模型实际输出 | piece 数最少 | 排名和最小（已否决） |
|---|---|---|---|
| lights | `li@@\|gh@@\|ts` | `light\|s`（对不上） | **`li@@\|gh@@\|ts`（对上了）** |
| footwell | 未观测 | `fo@@\|o@@\|t@@\|well` | `f@@\|oo@@\|tw@@\|el\|l` |
| 王梓涵 | `王\|梓\|涵` | `王\|梓\|涵` | 同 |

排名和最小虽然对上了 lights，但给 footwell 生成了 `f@@|oo@@|tw@@|el|l`，用作上下文时把整句
改成 `footwwall`（多出一个 w）——**比不生效更糟**。所以最终选 piece 数最少，并在注释里记下这次否决。

**结论与用法**：
1. `PF_CONTEXT` / `PF_HOTWORDS` 现在可以直接写中文短语（无需手写 token）；
2. 英文短语若要精确匹配模型的切分，仍要用 `a+b+c` 显式形式；
3. 想彻底解决，需要在转换 GGUF 时把 FunASR 的 `seg_dict`/`tokens.json` 之外的合并信息一并写入
   （现模型没有），或者按 SeACo 那类自带上下文偏置的权重重做——两条都是模型侧工作。

**回归**：32 条车控夹具 command 71.9%、zh CER 0.60%、en WER 6.87%，与未加编码器时一致；
10 条命名片段里中文 6/6 正确（王子涵→王梓涵、李牧尘→李沐宸），英文 2/4（OOV 名仍需 R 规则）。

## 30. 混说夹具重调热词：旧 profile 本来是负收益（2026-11-18，host CPU）

**为什么重调**：§13-14 的热词是在 16 条纯英文夹具上量出来的，那批夹具用 Tingting 合成，
句子结构全是 `Set the ...` 的英文单语模式。产品要的是**中英混说**，而混说的 token 序列
和纯英文不同——上一轮导出 15 句混说的 token 后就能看出来，`Set`/`said` 在混说里根本不出现。
继续用旧 profile 等于在一个不代表目标说话人的分布上调参。

**先修测量工具，再调参**。规则是：混说的字符 CER 不是合格指标（`导航到 Beijing 机场` 输出
`导航到北京机场` 语义正确却算 7 个字符错），所以新写了 `tools/eval_mixed.py`，直接复用
`examples/x-asr/tools/eval_vehicle.py` 的字段解析器，按 (device, position, state, value) 打分。
**解析器必须先能给自己满分**——用 15 句参考文本自检，发现并修掉三个仪器缺陷：

| 缺陷 | 现象 | 根因 |
|---|---|---|
| 英文多词模式永不命中 | `打开 front left window` 判成缺 position | zh 路径在 `re.sub(r"\s+","",text)` 后的串上匹配，`front left` 这种带空格的 pattern 不可能命中 |
| 英文动词全变 `?` | `turn on the 内循环` 判成 state 未知 | 同上，空格被删掉后 `\bturn on\b` 也失效 |
| 数字读不出 | `twenty one 度` / `twenty two degrees` | en 路径没有"数词+单位"规则；`22 degrees` 两条路径都不认 |

修法是**不改共享解析器**：两条路径都跑，取"子句数 → 已解析字段数"更优的一条（en 路径保留空格，
自然赢下前两个）；第三个是把英文数词归一成阿拉伯数字、`degrees` 归一成 `度`，
这也和 Whisper/SenseVoice 的 ITN 口径一致，不归一会冤枉自己。
自检从 1 失败到 **0/10 失败**，此时基线混说 command = 80%。

**A/B 结论（15 句混说 + 16 条纯英文 + 16 条纯中文，host CPU）**：

| 臂 | 混说 command | 英文 WER | 中文 |
|---|---:|---:|---|
| 无热词 | 80.0% | 12.21% | 基准 |
| 旧 `hotwords_vehicle.txt`（§13+§14） | **70.0%** | 8.40% | 逐字节不变 |
| 本轮（§15） | **100.0%** | **8.40%** | 逐字节不变 |

**旧 profile 在混说上是负收益**——不是没效果，是**有害**。根因是 §13 那条 `1.5 two`：
它修的是纯英文 `level two`，但在混说句 `set the 主驾 temperature to 二十一度` 里，
`to` 后面那个位置的 `two` 只差 0.96，g 在 1.0 以上就会插进去，输出变成
`temperature two二十一度`，下游读成 `to | two | 21` = **11 度**。这不是精度下降，是把指令改错了。
把 g 降到 0.6 不插入了，但纯英文的 `level two` 也就修不回来了（WER 回到 9.16%）。

**冲突用前缀条件拆开**：两个位置的**前一个 token 不同**——`level two` 前是 `l`，
冲突位置前是 `to`。改成 `1.5 l>two` 后两者同时正确，这条是 §15 相对 §13 的主要收获：
**同一个目标 token 在不同上下文里需要的 gain 可以相反，全局增益是错的对象。**

**`rear right` 不是缺词**：混说里 `打开 rear right reading light` 吐成 `打开 we are right ...`，
但纯英文 `turn on the rear right reading light` 是**完美**的（`re@@|ar`）。词表里确实没有 `rear`
整词，但有 `re@@`+`ar`，所以这是**中英切换边界上第一个英文词变弱**，不是词表缺词。
上一轮把它记成"词表缺词、偏置修不了"是错的。全局加 `2.6 re@@` 能修，但输出成 `reare`
（`re@@|are` 拼出 reare）——单点增益改不了"三个 token 对两个"的形状。
正确工具是 §14 已实现的 **span rewrite**（`R <margin> <wrong> => <right>`）：
`re@@+ar+right` 正好也是 3 个 token，可以原位替换。margin 是声学守卫，
实测 deficit 2.39，`margin 1.5` 被拒、`2.6` 生效。**先量 deficit 再定 margin**，
不能用"能改对"当判据。

**`footwell` 是 §14 判错的**：§14 说 `footwall`(3 token) 与 `footwell`(2 token) token 数不同、
偏置修不了，且文本 ITN 会造出 `footwwell`。这条对**整词**成立，对**词尾**不成立——
模型已经走对到 `f@@ oo@@ t@@`，只在最后一跳选了 `al` 而不是 `well`，而 `well` 是单个词表 token、
只差 2.89，且前一 token 正好是 `t@@`。`3.0 t@@>well` 直接把 `footal` 修成 `footwell`。
教训：**"token 数不同所以修不了"要先问"不同发生在哪一段"**，前缀相同的部分完全可以用条件偏置接住。

**验收口径（本轮确立）**：
1. 混说门禁看 **command 字段**，不看字符 CER；
2. 纯英文/纯中文两套夹具都要跑，且中文转写必须**逐字节不变**（本轮 16/16）；
3. 热词规则按 `R` / `<prefix>><token>` / `<token>` 三级优先，能用条件就不用全局；
4. 新规则进来先测 deficit 再定 gain/margin，**gain 上限 5.0、margin 是距离不是概率**；
5. 夹具里凡是用 Tingting 合成的，manifest 可能是 CRLF——`\r` 会把行首覆盖掉导致全 100% 错，先 `tr -d '\r'`。

**剩余的英文错误（8.40% 那 6 条）**：`Set`→`a`/`said`、`heater`→`see peater`/`peater`、
`massage`→`massaician`、`vents`→`reevent`、`lights`→`likes`、`footwell`→`footwall`。
全部是**词表缺形**或**多 token 粘连**，`R` 规则的 margin 也接不住（deficit 超一大截）。
这些只能靠重训、或上真正的语义后处理层。
