#!/usr/bin/env python3
"""Run an unpacked FunASR Paraformer checkpoint on the same WAVs as the CLI."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("wavs", type=Path, nargs="+")
    args = parser.parse_args()
    if not (args.model / "model.pt").is_file() or args.threads < 1:
        parser.error("supply a local model directory and positive thread count")

    from funasr import AutoModel
    import soundfile

    model = AutoModel(model=str(args.model.resolve()), device="cpu", ncpu=args.threads, disable_update=True)
    with args.tsv.open("w", encoding="utf-8") as tsv, args.jsonl.open("w", encoding="utf-8") as jsonl:
        for wav in args.wavs:
            pcm, rate = soundfile.read(wav, dtype="float32")
            if rate != 16000 or pcm.ndim != 1 or len(pcm) == 0:
                raise ValueError(f"{wav} needs nonempty 16 kHz mono audio")
            start = time.monotonic()
            results = model.generate(input=pcm, fs=rate, cache={}, disable_pbar=True)
            wall_ms = (time.monotonic() - start) * 1000
            result, = results
            duration = len(pcm) * 1000 / rate
            text = " ".join(result["text"].split())
            print(f"{wav}\taudio_ms={duration:.3f}\twall_ms={wall_ms:.3f}\t{text}", file=tsv)
            print(json.dumps({"wav": str(wav), "audio_ms": duration, "wall_ms": wall_ms, **result},
                             ensure_ascii=False), file=jsonl)


if __name__ == "__main__":
    main()
