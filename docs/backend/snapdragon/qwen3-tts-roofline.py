#!/usr/bin/env python3
"""Qwen3-TTS-0.6B 在 v73(SM6650) 上的每帧预算表。

两本账分开算，因为它们由不同模块主导：
  bytes -> predictor 73%    MAC -> vocoder 82%
wall = max(LM 时间, vocoder 时间)，两者并行（LM 上 HTP，vocoder 上 CPU）。

改下面的 HTP_BW / HTP_MAC / CPU_MAC 三个数就能变成真机预测。
"""
H, QD, KVD, FF, L = 1024, 16 * 128, 8 * 128, 3072, 28
per_layer = H * QD + 2 * (H * KVD) + QD * H + 3 * H * FF      # 15.73 M
TALKER = per_layer * L + H * 3072                              # 28 层 + codec_head
PRED_AR = per_layer * 5 + H * H + H * 2048                     # 5 层 + proj + 1 head (每个位置)
PRED_NAR = 5 * (H * KVD + H * QD // 2 + 3 * H * FF) + H * 2048 * 15   # 单位置塌成 FFN 塔 + 15 头
Q = 1.03                                                       # per-block scale

def codec_macs_per_s(f=12.5):
    """按官方 decoder_config 的速率链算 code2wav 的 MAC/s。"""
    m = 8 * (4 * 512 * 1024 + 3 * 512 * 1024) * f              # pre_transformer
    r = f
    for _ in (0, 1):                                           # 2x upsample
        m += r * 4 * 1024 * 1024
        r *= 2
        m += r * (7 * 1024 + 1024 * 4096 + 4096 * 1024)        # ConvNeXt
    m += r * 7 * 1024 * 1536                                   # DAC entry
    for s, cin, cout in ((8, 1536, 768), (5, 768, 384), (4, 384, 192), (3, 192, 96)):
        m += r * 2 * s * cin * cout                            # ConvTranspose (k=2s)
        r *= s
        m += 3 * (r * 7 * cout * cout + r * cout * cout)        # 3 个残差单元
    return m + r * 7 * 96

CODEC = codec_macs_per_s()                                     # 31.0 GMAC/s
CODEC_MB = 118e6 * 1.03 / 1e6                                  # 权重 121 MB/chunk

# -------- v73 假设（实测后替换） --------
HTP_BW, HTP_MAC, CPU_MAC = 6e9, 30e9, 45e9   # GB/s, MAC/s, MAC/s

def lm(passes, pbits, tbits):
    p = PRED_AR * passes if passes > 1 else PRED_NAR   # 16 pass: 每位置一份权重; 1 pass: NAR 塔
    b = (TALKER * tbits / 8 + p * pbits / 8) * Q
    return b / 1e6, TALKER + p                        # LM MB/frame, MAC/frame

def wall(passes, pbits, tbits, voc_scale=1.0, chunk=4):
    lm_mb, mac = lm(passes, pbits, tbits)
    mb = lm_mb + 15 + CODEC_MB * voc_scale / chunk              # +KV +vocoder 权重摊薄
    t_lm = max(mb * 1e6 / HTP_BW, mac / HTP_MAC) * 1e3
    t_voc = CODEC * voc_scale * 0.08 / CPU_MAC * 1e3
    return lm_mb, mb, t_lm, t_voc, max(t_lm, t_voc)

print(f"codec: {CODEC/1e9:.1f} GMAC/s = {CODEC*0.08/1e9:.2f} GMAC/frame, {CODEC_MB:.0f} MB/chunk\n")
print(f"{'config':34s} {'LM MB':>6s} {'tot MB':>6s} {'LM ms':>6s} {'voc ms':>7s} {'wall':>6s} {'RTF':>5s}")
rows = [
    ("A 现状 16 pass W8",                  16, 8, 8, 1.00),
    ("B 16 pass W4",                       16, 4, 4, 1.00),
    ("C 1-pass, W8",                        1, 8, 8, 1.00),
    ("D 1-pass, 6bit talker",               1, 8, 6, 1.00),
    ("E 1-pass, 4bit talker",               1, 8, 4, 1.00),
    ("E + vocoder -33% (P2b)",              1, 8, 4, 0.67),
    ("F 8 book + 1-pass W4 (降级档)",        8, 4, 4, 1.00),
]
for name, p, pb, tb, vs in rows:
    lmb, mb, tlm, tvoc, w = wall(p, pb, tb, vs)
    print(f"{name:34s} {lmb:6.0f} {mb:6.0f} {tlm:6.0f} {tvoc:7.0f} {w:6.0f} {w/80:5.2f}")
print("\n注: LM 上 HTP(6 GB/s, 30 GMAC/s)，vocoder 上 CPU(45 GMAC/s)，两者并行。")
print("    RTF<1 要求 wall < 80 ms；只修 LM 时 wall 被 vocoder 的 82 ms 钉住。")
