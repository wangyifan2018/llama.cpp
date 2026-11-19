# Realtime 座舱热词目标

以 `/v1/realtime` 的中文终稿为主线。先提高车窗、空调、座椅、灯光的动作、方向、位置、数值和否定准确率，再提高地点、联系人、品牌等实体的召回。英文和中英混说保留回归。用同一 OpenAI 兼容 Server 和四路真实到达节奏验收，统计错误动作、误改及 P95/P99。

## 已测基线

SM6650 现役服务使用 Q8_0、HTP 优先、两个引擎、每引擎四线程和 300 ms Silero 自动断句。确认设备二进制与审查时 `build-android-xasr` 的 SHA-256 一致。设备热词文件对应 `0b2989564^`，主分支又增加了七个中文条件增益。设备实验继续使用已部署文件，未替换其服务。

| 评测 | 结果 |
|---|---|
| 32 条 Tingting，Realtime 自动断句 | 中文 12/16，英文 8/16，出现 1 个中文方向错误，P95/P99 735/744 ms |
| 同 32 条，REST WAV + 标点 | 中文 16/16，英文 12/16，零方向错误 |
| 18 条独立热词回放，Realtime 自动断句 | 中文 6/7，英文 0/6，反例 0/6，P95/P99 655/659 ms |
| 同 18 条，主分支词表和隔离真机候选，Realtime 自动断句 | 中文 6/7，英文 0/6，反例 0/6，P95/P99 655/664 ms |

指标从夹具 PCM 结束到客户端收到最后一条终稿，发送步长为 20 ms，包含断句等待。两个入口的分数不能互换。四个中文差异与句中分隔有关，另外四个英文差异涉及声学词形。热词样本使用系统合成音色，只用于候选淘汰与回归。

## 中文车控分句

`PF_REALTIME_PUNCTUATE=1` 复用 REST 的词时间戳和声学停顿分句规则，只调整 Realtime 终稿。SM6650 同一二进制、热词和四路音频下的对照如下。长短音频混合采用客户端提交，其他准确率与延迟行使用 300 ms 服务端自动断句。

| 指标 | 关闭分句标点 | 开启分句标点 |
|---|---:|---:|
| 中文车控整句 | 12/16 | 14/16 |
| 中文方向错误 | 1 | 0 |
| 英文车控整句 | 8/16 | 8/16 |
| 32 条四路错峰，P95/P99 | 737/744 ms | 727/745 ms |
| 四路长短混合，P95/P99 | 1193/1209 ms | 1189/1206 ms |

复现段落“空调别关，把后排出风关掉”恢复分隔，消除了后排出风的方向错误。“把左前，车窗打开一半”出现了新的位置分隔错误，“关掉按摩座，椅通风保持开启”仍把座椅词切开。候选保留为可选开关。下一次验收需要保护车内位置和设备词的边界，并加入独立中文说话人的复合指令和非指令负例。

在预留设备上分别以 `PF_REALTIME_PUNCTUATE=0` 和 `PF_REALTIME_PUNCTUATE=1` 启动同一包。设置不同的 `PF_RUN`，两次都运行本页的热词回放，32 条自动断句结果在 `tingting-score.json` 和 `tingting.jsonl`。开启时可直接用 `PF_REALTIME_PUNCTUATE=1 tools/reproduce_deployment.sh` 执行部署验收，额外输出 `realtime-auto-score.json`。

SeACo-Paraformer 是可选的参考模型。同一 18 条音频在主机 CPU 上无上下文得到中文 6/7、英文 0/6。空格分词和逐行完整短语词表两种输入均得到中文 7/7、英文 1/6，6 条反例中无本表定义的实体引入。空格分词前后的主机中位推理耗时约 449/451 ms。这个结果只衡量模型对拍，不能推断 SM6650、HTP 或 Realtime 尾延迟。 [SeACo-Paraformer 论文](https://arxiv.org/abs/2308.03266) · [上游热词参数](https://github.com/modelscope/FunASR/blob/main/docs/python_api.md)

同一主机 CPU 上，在当前 Q8_0 Realtime 手动提交配置中仅增加已有 `PF_CONTEXT=context_vehicle.txt`，中文从 6/7 到 7/7，英文保持 0/6，反例均为 0/6。用非实时发送做该音频回归，提交到终稿 P95/P99 从 84/86 到 88/116 ms。18 条样本的 P99 受单次抖动影响。把这条轻量方案留作对照，采用四路 SM6650 回放再决定是否发布。

单独启动英语 Q8_0 checkpoint，在同一 Realtime 手动提交入口测 6 条英文正例和 3 条反例，命中 1/6，误引入 0/3。现有试验中，英文 checkpoint 和 SeACo 都没有解决大部分英文专名。记录两者为参照，继续验证限定领域和声学证据的重打分。

## 单路中文 Realtime 候选

CIF 的 fire time 是解码时间锚点。把它直接用作分句边界，会把没有实际停顿的“左前车窗”断开，也会把“关掉按摩，座椅……”的真实停顿错挂到“座椅”词内。候选在时间锚点之间检查 20 ms PCM 窗口，要求连续 120 ms 的 RMS 都低于 0.012，再把落在车内设备或位置词内部的候选分界移到词首。中文 Realtime 终稿使用这个规则，REST 继续使用原有时间戳规则。`PF_REALTIME_PUNCTUATE=1` 启用终稿分句。固定 RMS 门槛还需要在真实车内噪声和麦克风增益下复核。

同一批 CIF 性能改动把卷积权重从 `[输出][输入][tap]` 重排为 `[tap][输入][输出]`，使输出通道的累加可连续向量化，并保留每个通道的累加顺序。Mac CPU 对拍覆盖 77 条音频，包含两组音色、近音反例和 10 秒混合夹具，转写、token 序列和时间戳逐条一致。CIF 阶段中位从 7.94 ms 降至 1.08 ms。SM6650 单路手动提交时，HTP 路径中的主机 CIF 中位从 8.17 ms 降至 5.13 ms，HTP encoder 仍约为 230 ms。

在 SM6650 上用 Q8_0、HTP 优先、一 HTP 加一 CPU、每引擎四线程，采用 20 ms 帧做单路 Realtime 对照。基线源码为 `2e7ccd1d4`，两组都开启 `PF_REALTIME_PUNCTUATE=1`。手动提交从 `commit` 计时，自动断句从 WAV 发送结束计时。下表用回放端逐段收到最后终稿的时间计算百分位。

| 单路工作负载 | 中文整句和方向错误，之前 → 候选 | P50/P95/P99，之前 → 候选 |
|---|---|---|
| 前端 VAD 后提交 | 16/16、0 → 16/16、0 | 327/387/434 → 327/380/402 ms |
| 服务端 300 ms 自动断句 | 14/16、0 → 16/16、0 | 657/730/763 → 658/731/761 ms |
| 服务端 180 ms 自动断句 | 15/16、0 → 16/16、0 | 500/724/1102 → 493/569/578 ms |
| 服务端 120 ms 自动断句 | 15/16、0 → 16/16、0 | 436/499/507 → 424/499/506 ms |

已有前端 VAD 时，用 `input_audio_buffer.commit` 提交完整语段。SM6650 最终回放中它的 P95 为 380 ms，300 ms 自动断句为 731 ms。120 ms 与 180 ms 的行来自补齐“风量”词边界之前的阈值扫描，两者都有 5/16 条音频产生多个终稿，300 ms 为 1/16，手动提交为 0/16。表中整句分数基于同一音频的各终稿拼接。较短阈值仍留作实验，避免客户端把带否定或转折的后半句当作独立命令执行。16 条样本的尾延迟会受个别请求影响，手动提交的两次候选 P95 分别为 406 和 380 ms。180 ms 基线中单条 HTP 推理抖到 938 ms。两条原分句错误已修复，手动与 300 ms 档中文 CER 为 0.60%，两个短阈值档为 1.20%。

```sh
# Existing test service with PF_REALTIME_PUNCTUATE=1.
PF_RUN=/tmp/pf-single examples/paraformer/tools/replay_single_realtime.sh \
    8899 "$PF_MODELS/tingting" manual 300 180 120

# Isolated Android build and replay on a reserved SM6650 device.
PF_REALTIME_PUNCTUATE=1 PF_REPLAY_SINGLE=1 examples/paraformer/tools/reproduce_deployment.sh
```

夹具 WAV 和清单由 `tools/tingting-fixtures.sha256` 校验。单路验收保存终稿、`item_id`、回执和终稿时刻、后端 `infer_ms`、结果字段和方向错误。隔离部署同时把手动提交和 300 ms 自动断句的 16/16、零方向错作为 Tingting 回归门槛。整车产品精度和时延门槛仍由客户 VAD 与真实座舱录音验收。 [编译和设备依赖](DEPLOY.md#43-从零复现当前效果)

这批设备包的模型 SHA-256 为 `e73232ecf7e484907fd58f6a481fa27c1ceff454a1bc441d4a032a3a63e9783f`，Silero VAD 为 `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f`。工具链为 `ghcr.io/snapdragon-toolchain/arm64-android@sha256:e39e5e9c0957550a92c0de756b7556d81196c7c7f7328ed80de252bfb4878adf`。Android 基线二进制 SHA-256 为 `785cdc663bf01c7153148133845bc989436bcb709cc0645b36850281292de650`，阈值扫描版为 `768a0293d0d320cdd1e4725bcdb3dffb4a81be0d20883cce3f923b05a1bc0598`，最终版为 `b384b360df028443b429369a104b6aa05dd7a5885a4993d50949ce2b0634dffb`。原始回放和设备逐文件校验记录在上文归档下的 `single.QVdIFR`，其中 `final-bundle/package/SHA256SUMS` 固定其余依赖和热词、文本表、预热 WAV。

在两次设备配置里加入 `export CRISPASR_PARAFORMER_BENCH=1` 可复现 CIF 分项计时。让手动提交作为启动后的第一组回放，在设备日志中取两个引擎预热记录之后的 16 条 `paraformer_bench: cif_predict`。Meijia 的同文与未见句式回放可用归档的 `meijia-verified-fixtures.tar.gz`，其 SHA-256 为 `8d8e997665013404c05840cf114e606e696053b4f4f586ff4ef9fe9d0426aa3a`。档内有 28 条被测 WAV、两份清单和逐文件哈希。

Meijia 同文回放补上“风量”后，手动提交从 15/16 提到 16/16，300 ms 自动断句从 14/16 提到 15/16。12 条未见句式在最终版上分别为 11/12 和 10/12，均为零方向错误。复合灯光命令的“和”偶尔转写为“焊”，自动断句还漏掉过“按摩”。旧的 `voice2` 夹具两种模式均为 4/16，音色来源和发音质量有待独立确认，原始结果一并留档。

回放 Meijia 时使用 `tools/eval_realtime.py --unpaced` 单路跑档内的 `meijia/manifest.tsv` 和 `meijia-heldout/manifest.tsv`。分别添加 `--server-vad --silence-ms 300` 和使用默认手动提交。用 `examples/x-asr/tools/eval_vehicle.py` 打分，`meijia-heldout` 另外指定 `--semantics examples/paraformer/tools/vehicle_semantics_expanded.tsv`。两份未见句式结果均用准确率评估，时延以按 20 ms 节奏发送的 Tingting 回放为准。

## 短语增强和近音反例

真机配对实验保持同一个 HTP + CPU 配置，在 `/v1/realtime` 用 20 ms PCM 包和服务器自动断句，每组最多四路，相邻流错开 220 ms。每个配置单独启动，保留结果、`item_id`、后端和时间。用 Tingting/Samantha 回放原来的 18 条，再用 Flo 中文和 Daniel 英文合成同一批文字，检查对音色变化的敏感度。第三组另有 12 条近音反例，例如同音人名、同一个地名的汽车站和火车站，以及 Lucky Coffee 和 Luckin Coffee。

| 四路错峰自动断句 | 关闭短语 trie | 原有短语 trie | 最大 logit 差值 4 | 最大 logit 差值 6 |
|---|---|---|---|---|
| 原音色中文实体、英文实体 | 6/7、0/6 | 7/7、0/6 | 6/7、0/6 | 7/7、0/6 |
| 换音色中文实体、英文实体 | 2/7、1/6 | 5/7、1/6 | 2/7、1/6 | 4/7、1/6 |
| 中文近音反例误引入 | 0/6 | 2/6 | 0/6 | 2/6 |
| 原音色 P95/P99 | 674/683 ms | 667/675 ms | 663/679 ms | 656/674 ms |
| 换音色 P95/P99 | 634/663 ms | 636/676 ms | 647/681 ms | 690/708 ms |
| 32 条车控整句、方向错误 | 20/32、1 | 20/32、1 | 20/32、1 | 未跑 |

候选新增 `PF_CONTEXT_MAX_LOGIT_GAP`。短语续词必须在限定的原始 logit 差值内才可获得短语增益，默认从声学输出自然进入短语，同一帧的最佳原始 token 作为参照。该方案只复用已读回的 logits。`PF_CONTEXT` 未配置时继续使用发布词表和文字规则，未提供 margin 时保持原有 trie 行为。全局 trie 和差值 6 都把“王子涵”和“李沐晨”改成了另一个联系人。差值 4 的两个音色都等于关闭短语表的命中数。两组正例各 18 条，P99 只用于这轮对照。全局短语增强继续留在实验配置。

另外测了 X-ASR INT8 480 ms 的宿主 CPU 对照。用模型自带的双语 BPE 和四路径 `modified_beam_search`，对热词增益 2、4、6 做扫描。原音色中文 1/7、英文 1/6，换音色中文 3/7、英文 1/6，所有档位与无热词对照相同，12 条近音反例均无目标实体误引入。早期 `cjkchar+bpe` 对照将中文逐字切分，和模型词表不匹配，记录在独立目录。该 X-ASR 配置没有进入实时链路。 [Sherpa-ONNX 热词接口](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)

## 会话热词

Realtime 支持在 `session.audio.input.transcription.keywords` 设置中文热词。每次配置最多 24 个短语，每个短语 2 至 16 个模型词表内的汉字。`[]` 关闭会话短语增益，省略字段保留当前设置。热词 trie 在会话配置时建立，入队时按 `item_id` 持有不可变快照。HTP 与 CPU 工作线程读取同一快照，模型继续复用现有 logits，不增加模型推理。第一个字由原转写起头，后续字必须在最佳原始 logit 的 6.0 差值内才获得增益。

用同一条音频并发开两条 WebSocket，只有带“王梓涵”的会话输出“给王梓涵打电话”。另一条同时返回“给王子涵打电话”。收到已接收回执后立即清空词表，已入队语段仍用旧词表，下一段恢复原转写。该检查同时覆盖 CPU、HTP 和服务端自动断句。嵌套配置的 24 kHz、非法 VAD 参数和超限词表均返回错误，之前配置保持有效。

```sh
python3 examples/paraformer/tools/check_realtime_session.py \
    --url ws://127.0.0.1:8899/v1/realtime --wav "$PF_OUT/hotword-tts/entity_zh03.wav" --expect-vad

python3 examples/paraformer/tools/eval_realtime.py \
    --url ws://127.0.0.1:8899/v1/realtime --manifest "$PF_OUT/hotword-tts/manifest.tsv" \
    --server-vad --unpaced --keyword 王梓涵 --keyword 虹桥火车站 \
    --out "$PF_OUT/scoped.tsv" --details "$PF_OUT/scoped.jsonl"
```

## 对照与放行

1. 先锁定 `PF_HOTWORDS`/`PF_TEXT_RULES`，验证中文复合指令在 Realtime 终稿保留分句边界。用相同 WAV 对比 client commit 与 server VAD，分别统计动作、方向、位置、数值和否定。保留每段的 `item_id`、后端和端到端耗时。
2. 将地点、联系人与车控词分开，通过当前会话的业务上下文只激活相关候选。复用同一遍解码 logits，按短语累积声学证据。候选近音且无法区分时保留原转写，记录原始与候选结果。用近音负例检测是否误选联系人或地点。
3. 以 SeACo 热词分支做中文参考。若单遍重打分收益仍不足，测按需二级复核。额外模型只处理高价值且低置信度的语段，记录触发比例、模型常驻内存，以及 HTP 忙时的四路短请求尾延迟。
4. 验收按调参集、独立合成音色和真实目标车录音划分。同样词表要覆盖大小不同的上下文、近音误报和跨域误改。以每类实体召回、反例误引入率、下游错误动作和多余动作作为精度门禁。延迟对照使用相同到达节奏、智能座舱并行负载和 HTP 配置，比较 P95/P99 与截止时间超时数。产品门槛由目标车验收确认。

```sh
PF_OUT=/tmp/pf-hotword-replay
XASR_EVAL_TEXTS=examples/paraformer/tools/hotword_commands.tsv ZH_VOICE=Tingting EN_VOICE=Samantha \
    sh examples/x-asr/tools/make_eval_pack.sh "$PF_OUT/hotword-tts"
XASR_EVAL_TEXTS=examples/paraformer/tools/hotword_commands.tsv ZH_VOICE='Flo (中文（中国大陆）)' EN_VOICE=Daniel \
    sh examples/x-asr/tools/make_eval_pack.sh "$PF_OUT/hotword-voices"
XASR_EVAL_TEXTS=examples/paraformer/tools/hotword_near_commands.tsv ZH_VOICE=Tingting EN_VOICE=Samantha \
    sh examples/x-asr/tools/make_eval_pack.sh "$PF_OUT/hotword-near"

# Configure a reserved test service for each arm and replay the same packs.
PF_RUN="$PF_OUT/off" examples/paraformer/tools/replay_realtime_hotwords.sh \
    8899 "$PF_OUT" "$PF_MODELS/tingting"
```

测试包内置 `context_vehicle.txt`。在隔离测试服务的 `PF_CONF` 中先清空 `PF_CONTEXT` 跑对照，再用下面配置重启同一个测试服务，将 `PF_RUN` 改成 `"$PF_OUT/guarded"` 重跑。第二个差值档把 4 改成 6，原有 trie 对照省去 `PF_CONTEXT_MAX_LOGIT_GAP`。保持 HTP 优先、一个 HTP 和一个 CPU 引擎、每引擎四线程、300 ms 自动断句。重生成后的哈希若不匹配，使用归档 WAV 或单独记为新音色基线。 [部署和隔离回放](DEPLOY.md#43-从零复现当前效果)

```sh
PF_CONTEXT=/vendor/etc/paraformer/context_vehicle.txt
PF_CONTEXT_MAX_LOGIT_GAP=4
```

X-ASR 对照使用 X-ASR 480 ms INT8 模型目录中的 `encoder.int8.onnx`、`decoder.onnx`、`joiner.int8.onnx`、`tokens.txt` 和 `bpe.model`。热词文件从 `hotword_annotations.tsv` 的 `required` 列取值，按 `|` 分隔后逐行保存。

```sh
python3 -m pip install sherpa-onnx==2.0.1 sentencepiece==0.2.2 numpy==2.0.2
PF_REPO=$PWD
(cd "$XASR_MODEL" && sha256sum -c "$PF_REPO/examples/paraformer/tools/xasr-hotword-model.sha256")
python3 examples/paraformer/tools/eval_xasr_hotwords.py --model "$XASR_MODEL" \
    --manifest "$PF_OUT/hotword-tts/manifest.tsv" --out "$PF_OUT/xasr-off.jsonl"
python3 examples/paraformer/tools/eval_xasr_hotwords.py --model "$XASR_MODEL" \
    --manifest "$PF_OUT/hotword-tts/manifest.tsv" --hotwords "$PF_OUT/xasr-hotwords.txt" \
    --score 2 --out "$PF_OUT/xasr-on.jsonl"
python3 examples/paraformer/tools/score_realtime_hotwords.py \
    --annotations examples/paraformer/tools/hotword_annotations.tsv --run "$PF_OUT/xasr-on.jsonl" \
    --out "$PF_OUT/xasr-score.json"
```

真机原始事件、夹具 WAV 哈希、模型和词表哈希、SeACo 对拍脚本与结果保存在 `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/mainline-5c06d3074.WOjAJd`。
其中 `realtime-fixtures.tar.gz` 保存本轮 80 个 WAV 和四份清单。SHA-256 为 `52b75cd028ed6c7b8812c7b897c939f1f6fbe648fa2cc0756040f248b9049ef7`。解包后将其中的 `tingting` 目录作为回放脚本的第三个参数，可跳过依赖本机音色版本的合成步骤。
