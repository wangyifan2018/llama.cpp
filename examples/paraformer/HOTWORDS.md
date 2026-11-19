# Realtime 座舱热词目标

以 `/v1/realtime` 的自动断句终稿为主线。目标是在独立说话人和未见句式上提高车控词、地点、联系人、品牌及中英混说实体的召回，同时控制误改、错误动作和 P95/P99。继续用同一 OpenAI 兼容 Server，模型或解码策略由同一套录音和请求流量决定。

## 已测基线

SM6650 现役服务使用 Q8_0、HTP 优先、两个引擎、每引擎四线程和 300 ms Silero 自动断句。确认设备二进制与审查时 `build-android-xasr` 的 SHA-256 一致。设备热词文件对应 `0b2989564^`，主分支又增加了七个中文条件增益。设备实验继续使用已部署文件，未替换其服务。

| 评测 | 结果 |
|---|---|
| 32 条 Tingting，Realtime 自动断句 | 20/32 整句正确，出现 1 个方向错误，P95/P99 735/744 ms |
| 同 32 条，REST WAV + 标点 | 28/32，零方向错误 |
| 18 条独立热词回放，Realtime 自动断句 | 中文 6/7，英文 0/6，反例 0/6，P95/P99 655/659 ms |
| 同 18 条，主分支词表和隔离真机候选，Realtime 自动断句 | 中文 6/7，英文 0/6，反例 0/6，P95/P99 655/664 ms |

指标从夹具 PCM 结束到客户端收到最后一条终稿，发送步长为 20 ms，包含断句等待。两个入口的分数不能互换。四个中文差异与句中分隔有关，另外四个英文差异涉及声学词形。热词样本使用系统合成音色，只用于候选淘汰与回归。

SeACo-Paraformer 是可选的参考模型。同一 18 条音频在主机 CPU 上无上下文得到中文 6/7、英文 0/6。空格分词和逐行完整短语词表两种输入均得到中文 7/7、英文 1/6，6 条反例中无本表定义的实体引入。空格分词前后的主机中位推理耗时约 449/451 ms。这个结果只衡量模型对拍，不能推断 SM6650、HTP 或 Realtime 尾延迟。 [SeACo-Paraformer 论文](https://arxiv.org/abs/2308.03266) · [上游热词参数](https://github.com/modelscope/FunASR/blob/main/docs/python_api.md)

同一主机 CPU 上，在当前 Q8_0 Realtime 手动提交配置中仅增加已有 `PF_CONTEXT=context_vehicle.txt`，中文从 6/7 到 7/7，英文保持 0/6，反例均为 0/6。用非实时发送做该音频回归，提交到终稿 P95/P99 从 84/86 到 88/116 ms。18 条样本的 P99 受单次抖动影响。把这条轻量方案留作对照，采用四路 SM6650 回放再决定是否发布。

单独启动英语 Q8_0 checkpoint，在同一 Realtime 手动提交入口测 6 条英文正例和 3 条反例，命中 1/6，误引入 0/3。现有试验中，英文 checkpoint 和 SeACo 都没有解决大部分英文专名。记录两者为参照，继续验证限定领域和声学证据的重打分。

## 短语增强和近音反例

真机配对实验保持同一个 HTP + CPU 配置，在 `/v1/realtime` 用 20 ms PCM 包和服务器自动断句，每组最多四路，相邻流错开 220 ms。每个配置单独启动，保留结果、`item_id`、后端和时间。用 Tingting/Samantha 回放原来的 18 条，再用 Flo 中文和 Daniel 英文合成同一批文字，检查对音色变化的敏感度。第三组另有 12 条近音反例，例如同音人名、同一个地名的汽车站和火车站，以及 Lucky Coffee 和 Luckin Coffee。

| 四路错峰自动断句 | 关闭短语 trie | 原有短语 trie |
|---|---|---|
| 原音色中文实体、英文实体 | 6/7、0/6 | 7/7、0/6 |
| 非原音色中文实体、英文实体 | 2/7、1/6 | 5/7、1/6 |
| 原音色 P95/P99 | 658/662 ms | 667/675 ms |
| 非原音色 P95/P99 | 636/679 ms | 636/676 ms |
| 32 条车控整句、方向错误 | 20/32、1 | 20/32、1 |

候选新增 `PF_CONTEXT_MAX_LOGIT_GAP`。短语续词必须在限定的原始 logit 差值内才可获得短语增益，默认从声学输出自然进入短语，同一帧的最佳原始 token 作为参照。该方案只复用已读回的 logits。`PF_CONTEXT` 未配置时继续使用发布词表和文字规则；未提供 margin 时保持原有 trie 行为。`4` 是待对照的实验值，配置未启用。

另外测了 X-ASR INT8 480 ms 的宿主 CPU 对照。用模型自带的双语 BPE 和四路径 `modified_beam_search`，对热词增益 2、4、6 做扫描。原音色中文 1/7、英文 1/6，换音色中文 3/7、英文 1/6，所有档位与无热词对照相同，12 条近音反例均无目标实体误引入。早期 `cjkchar+bpe` 对照将中文逐字切分，和模型词表不匹配，记录在独立目录。该 X-ASR 配置没有进入实时链路。 [Sherpa-ONNX 热词接口](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)

## 对照与放行

1. 先锁定已有 `PF_HOTWORDS`/`PF_TEXT_RULES`，按相同 WAV 对比 client commit 与 server VAD，拆出标点和末尾裁剪的影响。保留每段的 `item_id`、后端和端到端耗时。
2. 先测现有 `PF_CONTEXT` 和复用同一遍解码 logits 的短语重打分。只在领域候选和声学证据都满足时改写，保留原文本。口语替代表达、否定、位置、数值和非指令负例要分别报数。
3. 用 SeACo 热词分支及英文候选做参考。若单遍重打分收益仍不足，测按需二级复核。额外模型只处理高价值且低置信度的语段，记录触发比例、模型常驻内存，以及 HTP 忙时的四路短请求尾延迟。
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

测试包内置 `context_vehicle.txt`。在隔离测试服务的 `PF_CONF` 中先清空 `PF_CONTEXT` 跑对照，再用下面配置重启同一个测试服务，将 `PF_RUN` 改成 `"$PF_OUT/guarded"` 重跑。原有 trie 对照用同一路径，省去 `PF_CONTEXT_MAX_LOGIT_GAP`。保持 HTP 优先、一个 HTP 和一个 CPU 引擎、每引擎四线程、300 ms 自动断句。重生成后的哈希若不匹配，使用归档 WAV 或单独记为新音色基线。 [部署和隔离回放](DEPLOY.md#43-从零复现当前效果)

```sh
PF_CONTEXT=/vendor/etc/paraformer/context_vehicle.txt
PF_CONTEXT_MAX_LOGIT_GAP=4
```

X-ASR 对照使用 X-ASR 480 ms INT8 模型目录中的 `encoder.int8.onnx`、`decoder.onnx`、`joiner.int8.onnx`、`tokens.txt` 和 `bpe.model`。热词文件从 `hotword_annotations.tsv` 的 `required` 列取值，按 `|` 分隔后逐行保存。

```sh
python3 -m pip install sherpa-onnx==2.0.1 sentencepiece==0.2.2 numpy==2.0.2
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
