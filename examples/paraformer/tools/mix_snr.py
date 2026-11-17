#!/usr/bin/env python3
"""Mix a noise recording into speech at a target SNR.

Both signals are scaled so the noise power is speech power divided by
10^(snr/20). Unlike the earlier equal-weight mix this is level calibrated, so
the output can be used to measure how much a denoiser would have to recover.

    python3 mix_snr.py --speech pack/*.wav --noise cabin.wav --snr 10 --out noisy/
"""

import argparse
import os
import wave

import numpy as np


def read(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2, path
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return data.astype(np.float64) / 32768.0


def write(path, samples):
    pcm = np.clip(samples, -1.0, 1.0)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes((pcm * 32767.0).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", nargs="+", required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--snr", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    noise = read(args.noise)
    for path in args.speech:
        speech = read(path)
        # tile or trim the noise to the speech length
        reps = int(np.ceil(len(speech) / len(noise)))
        n = np.tile(noise, reps)[: len(speech)]
        sp = np.sqrt(np.mean(speech ** 2))
        npow = np.sqrt(np.mean(n ** 2))
        if npow < 1e-9 or sp < 1e-9:
            write(os.path.join(args.out, os.path.basename(path)), speech)
            continue
        n *= sp / (npow * (10.0 ** (args.snr / 20.0)))
        write(os.path.join(args.out, os.path.basename(path)), speech + n)
    print("mixed %d files at %.1f dB SNR into %s" % (len(args.speech), args.snr, args.out))


if __name__ == "__main__":
    main()
