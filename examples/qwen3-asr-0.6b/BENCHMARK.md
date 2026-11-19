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
