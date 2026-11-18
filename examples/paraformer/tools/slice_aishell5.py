#!/usr/bin/env python3
"""Select short non-overlapping utterances from local AISHELL-5 Dev DX01C01 recordings."""

import argparse
from pathlib import Path
import re
import wave


TIER = re.compile(r'(?m)^\s*name = "(P\d+)"')
INTERVAL = re.compile(r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*)"')
START_END_PADDING = 0.10


def read_grid(path):
    sections = TIER.split(path.read_text(encoding="utf-8-sig"))
    intervals = []
    for speaker, section in zip(sections[1::2], sections[2::2]):
        for begin, end, text in INTERVAL.findall(section):
            if text.strip():
                intervals.append((float(begin), float(end), speaker, text.replace('""', '"')))
    return sorted(intervals)


def write_session(folder, out, per_speaker, manifest, segments):
    grid = folder / "DX01C01.TextGrid"
    if not grid.is_file():
        return
    labels = read_grid(grid)
    if not labels:
        raise ValueError(f"no speaker annotations in {grid}")
    with wave.open(str(grid.with_suffix(".wav")), "rb") as src:
        if src.getparams()[:3] != (1, 2, 16000):
            raise ValueError(f"unexpected WAV format for {grid}")
        for speaker in sorted({label[2] for label in labels}):
            eligible = []
            for begin, end, spk, text in labels:
                start = max(0, begin - START_END_PADDING)
                stop = min(src.getnframes() / 16000, end + START_END_PADDING)
                if spk != speaker or not (1 <= end - begin <= 6) or len(re.findall(r'[\u4e00-\u9fff]', text)) < 4:
                    continue
                if any(other != (begin, end, spk, text) and other[1] > start and other[0] < stop for other in labels):
                    continue
                eligible.append((start, stop, text))
            count = min(per_speaker, len(eligible))
            for i in range(count):
                start, stop, text = eligible[i * len(eligible) // count]
                name = f"{folder.name}_{speaker}_{i + 1:02}.wav"
                first, last = round(start * 16000), round(stop * 16000)
                src.setpos(first)
                audio = src.readframes(last - first)
                with wave.open(str(out / name), "wb") as dst:
                    dst.setparams(src.getparams())
                    dst.writeframes(audio)
                print(name, "zh", text, sep="\t", file=manifest)
                print(name, folder.name, speaker, first, last, sep="\t", file=segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-speaker", type=int, default=6)
    args = parser.parse_args()
    if args.per_speaker < 1 or not args.dev.is_dir():
        parser.error("supply a Dev directory and positive per-speaker count")
    args.out.mkdir(exist_ok=True)
    with (args.out / "manifest.tsv").open("w", encoding="utf-8") as manifest, \
         (args.out / "segments.tsv").open("w", encoding="utf-8") as segments:
        for folder in sorted(args.dev.iterdir()):
            if folder.is_dir():
                write_session(folder, args.out, args.per_speaker, manifest, segments)


if __name__ == '__main__':
    main()
