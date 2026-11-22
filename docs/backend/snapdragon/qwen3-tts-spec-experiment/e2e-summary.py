#!/usr/bin/env python3
"""Validate streaming audio, summarize paired timing, and prepare blind WAVs."""
import argparse
import csv
from pathlib import Path
import random
import re
import shutil
import statistics
import wave

ROOT = Path(__file__).parent


def measured(pattern, log):
    match = re.search(pattern, log)
    if match is None:
        raise ValueError(f"missing trace {pattern}")
    return match.groups()


def read_pcm(path, frames, rate):
    raw = path.read_bytes()
    expected_bytes = frames * (rate * 8 // 100) * 2
    if raw.startswith(b"RIFF"):
        with wave.open(str(path), "rb") as f:
            if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, rate):
                raise ValueError(f"unexpected WAV header {path}")
            raw = f.readframes(f.getnframes())
    if len(raw) != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} PCM bytes, got {len(raw)}")
    return raw


def wav(path, rate, pcm):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm)


def prepare(run, dest, prompts):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "labeled").mkdir(exist_ok=True)
    (dest / "blind").mkdir(exist_ok=True)
    rng = random.Random(3601)
    rows = []
    with (dest / "blind-key.tsv").open("w", newline="") as key:
        key_writer = csv.writer(key, delimiter="\t")
        key_writer.writerow(("sample", "label", "variant"))
        for item in prompts:
            sample = item["id"]
            modes = ["original", "q8"]
            rng.shuffle(modes)
            for label, mode in zip(("A", "B"), modes):
                stem = f"{sample}-{mode}"
                path = run / (stem + ".log")
                log = path.read_text()
                frames, rate = map(int, measured(r"generated (\d+) frames, \d+ bytes of WAV audio \((\d+) Hz\)", log))
                seconds, = map(float, measured(r"timings: .* = total ([\d.]+)s", log))
                ttfa, rtf = map(float, measured(r"TTFA = ([\d.]+)s, RTF = ([\d.]+)", log))
                first, = map(int, measured(r"first audio at frame (\d+)", log))
                per_frame = [float(x) for x in re.findall(r"\[trace\] gen_code ([\d.]+) ms", log)]
                if len(per_frame) != frames or frames < 2:
                    raise ValueError(f"{stem}: invalid predictor frame count")
                pcm = read_pcm(run / (stem + ".wav"), frames, rate)
                labeled = dest / "labeled" / (stem + ".wav")
                wav(labeled, rate, pcm)
                shutil.copyfile(labeled, dest / "blind" / f"{sample}-{label}.wav")
                key_writer.writerow((sample, label, mode))
                rows.append(dict(sample=sample, language=item["lang"], mode=mode, frames=frames,
                                 frame_cap_hit=frames >= 96, audio_s=frames * 0.08, request_s=seconds,
                                 ttfa_s=ttfa, first_audio_frame=first, rtf=rtf, first_predictor_ms=per_frame[0],
                                 hot_predictor_p50_ms=statistics.median(per_frame[1:])))
    rows.sort(key=lambda row: (row["sample"], row["mode"]))
    with (dest / "paired.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (dest / "asr-trace.tsv").open("w") as file:
        for row in rows:
            file.write(f"0\t{dest.resolve()}/labeled/{row['sample']}-{row['mode']}.wav\n")
    with (dest / "blind" / "README.md").open("w") as file:
        file.write("# 盲听配对\n\n先听每对 A/B，再记录可懂度、读错或漏字、音色、韵律和破音。完成后再查看配对答案。\n\n")
        for item in prompts:
            file.write(f"- {item['id']} {item['text']}\n")
    for mode in ("original", "q8"):
        selected = [row for row in rows if row["mode"] == mode]
        print(mode, " ".join(f"{key}={statistics.median(row[key] for row in selected):.6f}" for key in
              ("hot_predictor_p50_ms", "ttfa_s", "rtf", "request_s", "audio_s")))
    for item in prompts:
        pair = {row["mode"]: row for row in rows if row["sample"] == item["id"]}
        original, q8 = pair["original"], pair["q8"]
        print(item["id"], " ".join(f"{key}={original[key]:.3f}->{q8[key]:.3f}" for key in
              ("hot_predictor_p50_ms", "ttfa_s", "rtf", "request_s", "audio_s")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    with (ROOT / "e2e-prompts.tsv").open() as stream:
        prepare(args.run, args.dest, list(csv.DictReader(stream, delimiter="\t")))
