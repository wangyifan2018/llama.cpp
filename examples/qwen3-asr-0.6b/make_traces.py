#!/usr/bin/env python3
"""Prepare three batch-two replays using the same tagged PCM16 fixture files."""

import argparse
from pathlib import Path
import shutil
import wave


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fixtures", type=Path, help="directory containing zh.wav, en.wav, de.wav")
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("zh.wav", "en.wav", "de.wav"):
        source = args.fixtures / name
        with wave.open(str(source), "rb") as wav:
            if wav.getparams()[:3] != (1, 2, 16000) or not 0 < wav.getnframes() <= 16000 * 5:
                cli.error(f"expected a short 16 kHz mono PCM16 WAV: {source}")
        shutil.copyfile(source, args.output / name)
    with wave.open(str(args.output / "zh.wav"), "rb") as wav:
        audio = wav.readframes(wav.getnframes())
    with wave.open(str(args.output / "long.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(5760).join([audio] * 3))

    traces = {
        "simultaneous": [(0, "zh.wav"), (0, "en.wav")],
        "staggered": [(0, "zh.wav"), (300, "de.wav")],
        "refill": [(0, "long.wav"), (0, "en.wav"), (300, "de.wav")],
    }
    for name, requests in traces.items():
        (args.output / f"{name}.tsv").write_text(
            "".join(f"{arrival}\t{path}\n" for arrival, path in requests), encoding="utf-8")


if __name__ == "__main__":
    main()
