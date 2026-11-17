#!/usr/bin/env python3
"""Single-channel Wiener denoiser prototype for the cabin front end.

Minimum-statistics noise estimate plus a decision-directed a priori SNR. This is
the reference the C++ stage is checked against, and it is what decides whether a
denoiser is worth porting at all.

    python3 denoise_wiener.py --in noisy/ --out clean/ [--floor 0.08]
"""

import argparse
import os
import wave

import numpy as np

N_FFT = 512      # 32 ms at 16 kHz
HOP = 160        # 10 ms
ALPHA = 0.98     # decision-directed smoothing
MIN_WIN = 100    # frames in the minimum-statistics window (1 s)


def read(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2, path
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return x.astype(np.float64) / 32768.0


def write(path, x):
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def frames_of(x, n=N_FFT, hop=HOP):
    count = 1 + max(0, (len(x) - n) // hop)
    pad = max(0, (count - 1) * hop + n - len(x))
    xp = np.concatenate([x, np.zeros(pad)])
    view = np.lib.stride_tricks.sliding_window_view(xp, n)[::hop]
    return view[:count]


def denoise(x, floor=0.08, over=1.0):
    win = np.hanning(N_FFT + 1)[:-1]
    fr = frames_of(x) * win
    spec = np.fft.rfft(fr, axis=1)
    power = np.abs(spec) ** 2

    # minimum statistics: per bin minimum over a trailing window
    padded = np.pad(power, ((MIN_WIN, 0), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, MIN_WIN, axis=0)
    noise = windows.min(axis=2) * over
    noise = np.maximum(noise, 1e-12)

    gain = np.ones_like(power)
    prev_gain = np.ones(power.shape[1])
    prev_gamma = np.ones(power.shape[1])
    for i in range(power.shape[0]):
        gamma = power[i] / noise[i]
        xi = ALPHA * (prev_gain ** 2) * prev_gamma + (1 - ALPHA) * np.maximum(gamma - 1.0, 0.0)
        g = xi / (1.0 + xi)
        g = np.maximum(g, floor)
        gain[i] = g
        prev_gain = g
        prev_gamma = gamma

    out_fr = np.fft.irfft(spec * gain, n=N_FFT, axis=1) * win
    out = np.zeros(len(x))
    norm = np.zeros(len(x))
    for i in range(out_fr.shape[0]):
        s = i * HOP
        out[s:s + N_FFT] += out_fr[i]
        norm[s:s + N_FFT] += win ** 2
    norm[norm < 1e-8] = 1.0
    return out / norm


def snr_db(clean, test):
    n = min(len(clean), len(test))
    c, t = clean[:n], test[:n]
    e = np.sum(c ** 2)
    d = np.sum((t - c) ** 2)
    return 10.0 * np.log10(max(e, 1e-12) / max(d, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--clean", help="reference dir, to report SNR gain")
    ap.add_argument("--floor", type=float, default=0.08)
    ap.add_argument("--over", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    gains = []
    for name in sorted(os.listdir(args.src)):
        if not name.endswith(".wav"):
            continue
        noisy = read(os.path.join(args.src, name))
        out = denoise(noisy, floor=args.floor, over=args.over)
        write(os.path.join(args.dst, name), out)
        if args.clean:
            ref = read(os.path.join(args.clean, name))
            gains.append(snr_db(ref, noisy + 0) - snr_db(ref, out))
    if gains:
        print("SNR gain: mean %.2f dB, min %.2f, max %.2f" % (float(np.mean(gains)), min(gains), max(gains)))
    print("wrote %s" % args.dst)


if __name__ == "__main__":
    main()
