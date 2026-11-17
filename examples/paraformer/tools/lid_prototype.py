#!/usr/bin/env python3
"""Language and speaker separability on cabin fixtures.

Shows why an acoustic LID cannot be trained on the data we have: the same 160-dim
log-mel statistics identify speakers almost perfectly and languages only at
chance level once the voices are unseen. Decoder-side LID (script of the decoded
tokens) is the usable one; see docs/backend/snapdragon/lid-and-speaker-research.md.

    python3 lid_prototype.py --tingting DIR --voice2 DIR
"""

import argparse
import glob
import os
import wave

import numpy as np

SR, N_FFT, HOP, N_MEL = 16000, 400, 160, 80


def mel_filterbank():
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    hz = mel2hz(np.linspace(hz2mel(20), hz2mel(SR / 2), N_MEL + 2))
    bins = np.floor((N_FFT + 1) * hz / SR).astype(int)
    fb = np.zeros((N_MEL, N_FFT // 2 + 1))
    for m in range(1, N_MEL + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(1, c - l)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(1, r - c)
    return fb


FB = mel_filterbank()
WIN = np.hamming(N_FFT)


def features(path):
    with wave.open(path, "rb") as f:
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * WIN, n=N_FFT, axis=1)) ** 2
    mel = np.log(np.maximum(spec @ FB.T, 1e-10))
    e = mel.mean(axis=1)
    keep = e > (e.max() - 3.0)
    mel = mel[keep] if keep.sum() > 5 else mel
    return np.concatenate([mel.mean(0), mel.std(0)])


def read_manifest(pack):
    lang = {}
    m = os.path.join(pack, "manifest.tsv")
    if os.path.exists(m):
        for line in open(m, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3:
                lang[os.path.basename(f[0])] = f[1]
    return lang


def train(X, y, l2=1.0, iters=800, lr=0.5):
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def predict(X, w, b):
    return (1 / (1 + np.exp(-(X @ w + b))) > 0.5).astype(float)


def speaker_acc(X, labels):
    correct = 0
    for i in range(len(X)):
        mask = np.arange(len(X)) != i
        cents = {k: X[mask][labels[mask] == k].mean(0) for k in set(labels)}
        d2 = {k: ((X[i] - c) ** 2).sum() for k, c in cents.items()}
        correct += int(min(d2, key=d2.get) == labels[i])
    return 100.0 * correct / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tingting", required=True, help="pack with Tingting (zh) + Samantha (en)")
    ap.add_argument("--voice2", required=True, help="pack with Flo (zh) + Daniel (en)")
    args = ap.parse_args()

    rows = []
    for pack, voice in ((args.tingting, "tingting"), (args.voice2, "flo")):
        lang = read_manifest(pack)
        for p in sorted(glob.glob(os.path.join(pack, "*.wav"))):
            lg = lang.get(os.path.basename(p))
            if lg in ("zh", "en"):
                rows.append({"x": features(p), "y": 1.0 if lg == "zh" else 0.0,
                             "speaker": voice + "-" + lg})
    X = np.stack([r["x"] for r in rows])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = np.array([r["y"] for r in rows])
    spk = np.array([r["speaker"] for r in rows])
    spk = np.array([s.replace("tingting-", "tingting-") for s in spk])
    print("clips %d (zh %d, en %d)" % (len(rows), int(y.sum()), int((1 - y).sum())))

    # 5-fold within the same voices
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(rows)), 5)
    correct = 0
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        w, b = train(X[tr], y[tr])
        correct += int((predict(X[te], w, b) == y[te]).sum())
    print("LID 5-fold (same voices): %.1f%%" % (100.0 * correct / len(rows)))

    # cross-voice: train on tingting pack, test on voice2 pack
    tr = np.array([s.startswith("tingting") for s in spk])
    te = ~tr
    w, b = train(X[tr], y[tr])
    print("LID cross-voice (train tingting, test flo/daniel): %.1f%% of %d"
          % (100.0 * (predict(X[te], w, b) == y[te]).mean(), te.sum()))

    print("speaker ID, Mandarin pair: %.1f%%" % speaker_acc(X[(spk == "tingting-zh") | (spk == "flo-zh")],
                                                            spk[(spk == "tingting-zh") | (spk == "flo-zh")]))
    print("speaker ID, English pair:  %.1f%%" % speaker_acc(X[(spk == "tingting-en") | (spk == "flo-en")],
                                                            spk[(spk == "tingting-en") | (spk == "flo-en")]))
    print("speaker ID, four voices:   %.1f%%" % speaker_acc(X, spk))


if __name__ == "__main__":
    main()
