#!/usr/bin/env python3
"""Speaker-gate prototype on the cabin fixtures.

Kaldi-style 80-dim log-mel -> CAM++ ONNX embedding -> per-trip enrolment and a
cosine threshold. Reports separability (driver vs passenger) and the per-segment
cost of running the gate on the CPU while ASR stays on HTP.

    python3 spk_gate_eval.py --onnx campplus.onnx --driver DIR --passenger DIR
"""

import argparse
import glob
import os
import time
import wave

import numpy as np

SR, N_FFT, WIN, HOP, N_MEL = 16000, 512, 400, 160, 80


def mel_banks():
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    lo, hi = hz2mel(20.0), hz2mel(SR / 2.0)
    mels = np.linspace(lo, hi, N_MEL + 2)
    bins = np.floor((N_FFT + 1) * mel2hz(mels) / SR).astype(int)
    fb = np.zeros((N_MEL, N_FFT // 2 + 1))
    for m in range(1, N_MEL + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(1, c - l)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(1, r - c)
    return fb


FB = mel_banks()          # 80 x 257
HAMMING = np.hamming(WIN)


def fbank80(path):
    with wave.open(path, "rb") as f:
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    # kaldi defaults: pre-emphasis 0.97, hamming window, 25/10 ms
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    n = 1 + max(0, (len(x) - WIN) // HOP)
    if n <= 0:
        return np.zeros((1, N_MEL))
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * HAMMING[None, :]
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    return np.log(np.maximum(spec @ FB.T, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--driver", required=True, help="dir with the enrolled speaker's wavs")
    ap.add_argument("--passenger", required=True, help="dir with impostor wavs")
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    frames = sess.get_inputs()[0].shape[1]
    if not isinstance(frames, int):
        raise SystemExit("the ONNX graph must have a fixed frame dimension (see export_campplus_onnx.py)")
    print("onnx graph input: %s (fixed T=%d, %.1f s)" % (sess.get_inputs()[0].shape, frames, frames / 100.0))

    def embed(path):
        # Chunk into fixed-length windows, pad the tail with the chunk mean,
        # then average the per-chunk embeddings.
        feats = fbank80(path).astype(np.float32)
        chunks = []
        for i in range(0, max(1, feats.shape[0]), frames):
            c = feats[i:i + frames]
            if c.shape[0] < frames:
                pad = np.repeat(c.mean(axis=0, keepdims=True), frames - c.shape[0], axis=0)
                c = np.concatenate([c, pad], axis=0)
            chunks.append(c[None, :, :])
        embs = [sess.run(None, {"feats": c})[0].ravel() for c in chunks]
        embs = [e / (np.linalg.norm(e) + 1e-9) for e in embs]
        return np.mean(embs, axis=0)

    drv = sorted(glob.glob(os.path.join(args.driver, "*zh*.wav")))
    pax = sorted(glob.glob(os.path.join(args.passenger, "*zh*.wav")))
    print("driver clips %d, passenger clips %d" % (len(drv), len(pax)))

    t0 = time.perf_counter()
    de = [embed(p) for p in drv]
    pe = [embed(p) for p in pax]
    dt = (time.perf_counter() - t0) / max(1, len(de) + len(pe))
    print("embedding cost per clip: %.2f ms (CPU, %.1f x real time for 2.5 s audio)"
          % (dt * 1000.0, 2.5 / dt))

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-9)

    # leave-one-out enrolment: the driver centroid never contains the test clip
    gen, imp = [], []
    for i in range(len(de)):
        cen = unit(np.mean([de[j] for j in range(len(de)) if j != i], axis=0))
        gen.append(float(np.dot(unit(de[i]), cen)))
    cen_all = unit(np.mean(de, axis=0))
    for v in pe:
        imp.append(float(np.dot(unit(v), cen_all)))

    gen = np.array(gen)
    imp = np.array(imp)
    print("genuine  score: mean %.3f min %.3f" % (gen.mean(), gen.min()))
    print("impostor score: mean %.3f max %.3f" % (imp.mean(), imp.max()))
    for th in np.round(np.arange(0.30, 0.96, 0.10), 2):
        print("  threshold %.2f: FRR %5.1f%%  FAR %5.1f%%" % (th, 100 * (gen < th).mean(), 100 * (imp >= th).mean()))

    # best operating point by equal error
    best = None
    for th in np.linspace(0.0, 1.0, 1001):
        frr, far = (gen < th).mean(), (imp >= th).mean()
        d = abs(frr - far)
        if best is None or d < best[0]:
            best = (d, th, frr, far)
    print("EER-ish threshold %.3f: FRR %.1f%% FAR %.1f%%" % (best[1], 100 * best[2], 100 * best[3]))


if __name__ == "__main__":
    main()
