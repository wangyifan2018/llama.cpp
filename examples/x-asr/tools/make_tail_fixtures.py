#!/usr/bin/env python3
"""Build endpoint-phase and truncation fixtures for the tail-padding sweep.

Prepended silence shifts the chunk grid relative to the end of the utterance
without changing the words. Truncated clips move the endpoint inside the
480 ms window. Both are written as mono 16-bit 16 kHz WAV.
"""

import argparse
import os
import struct
import wave


def read(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000, path
        return list(struct.unpack("<%dh" % f.getnframes(), f.readframes(f.getnframes())))


def write(path, samples):
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(struct.pack("<%dh" % len(samples), *samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--short", required=True, help="short clip to phase-shift")
    parser.add_argument("--long", required=True, help="long clip to truncate")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    short = read(args.short)
    long = read(args.long)
    # Prepended silence moves the last real frame across the 48-frame chunk grid.
    for frames in (0, 6, 12, 18, 24, 30, 36, 42):
        lead = [0] * (frames * 160)
        name = os.path.join(args.out, "phase-%02d.wav" % frames)
        write(name, lead + short)
        print(name, "samples", len(lead) + len(short))
    # Truncation moves the endpoint inside the 480 ms window.
    for ms in (8800, 9100, 9400, 9700, 10000):
        name = os.path.join(args.out, "cut-%05d.wav" % ms)
        write(name, long[: ms * 16])
        print(name, "samples", ms * 16)


if __name__ == "__main__":
    main()
