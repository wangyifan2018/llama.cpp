#!/usr/bin/env python3
"""Prepare matched phone frames and compare output-distribution and timing deltas."""
import argparse
import csv
from pathlib import Path
import runpy
from statistics import median

import numpy as np

ROOT = Path(__file__).parent
filtered = runpy.run_path(str(ROOT / "train-offset.py"))["filtered"]
read_scores = runpy.run_path(str(ROOT / "small-ar.py"))["read"]
MODES = ("original", "down-f16", "down-q8", "all-q8", "keep-down-q8")


def compare(base, candidate):
    if base.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError("bad candidate shape or non-finite logits")
    match = base.argmax(-1) == candidate.argmax(-1)
    prefix = np.logical_and.accumulate(match, axis=1)
    overlap = np.minimum(filtered(base), filtered(candidate)).sum(-1)
    delta = np.abs(base - candidate)
    return match, prefix, overlap, delta


def main(args):
    run = args.run
    if args.prepare:
        first = np.fromfile(args.prepare[0], "<i4", count=1040)
        frames = np.fromfile(args.prepare[1], "<i4").reshape(-1, 1040)
        first.tofile(run / "frame0.bin")
        np.concatenate([first[None]] + [frames[i:i + 10] for i in range(0, 160, 40)]).tofile(run / "phone-eval.bin")
        return

    sets = (
        ("mac_forced_eval", "host-{}.scores", slice(0, 160)),
        ("mac_forced_regression", "host-{}.scores", slice(160, 256)),
        ("android_free_eval", "phone-{}-1.scores", slice(1, 41)),
    )
    with (run / "depths.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["set", "mode", "depth", "top1", "prefix_agreement", "overlap", "mean_abs_logit_delta"])
        for label, path, index in sets:
            base = read_scores(run / path.format("original"))[index]
            for mode in MODES[1:]:
                candidate = read_scores(run / path.format(mode))[index]
                match, prefix, overlap, delta = compare(base, candidate)
                for g in range(15):
                    writer.writerow((label, mode, g+1, match[:, g].mean(), prefix[:, g].mean(),
                                     overlap[:, g].mean(), delta[:, g].mean()))
                print(f"{label} {mode} frames={len(base)} top1={match.mean():.6f} "
                      f"exact_prefix4={prefix[:, 3].mean():.6f} exact_frame={prefix[:, -1].mean():.6f} "
                      f"overlap={overlap.mean():.6f} mean_abs_logits={delta.mean():.6f}")
    timings = {}
    for mode in MODES:
        samples = []
        for block in (1, 2):
            with (run / f"phone-{mode}-{block}.csv").open() as file:
                rows = [float(row["compute_ms"]) for row in csv.DictReader(file)]
            assert len(rows) == 24
            samples.append(rows)
        a, b = samples
        timings[mode] = median(a + b)
        print(f"android {mode} block1_median_ms={median(a):.6f} block2_median_ms={median(b):.6f} "
              f"median_ms={timings[mode]:.6f} speedup={timings['original']/timings[mode]:.6f}x "
              f"min_ms={min(a+b):.6f} max_ms={max(a+b):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--prepare", nargs=2, type=Path, metavar=("E28_CAPTURE", "E34_EVAL"))
    main(parser.parse_args())
