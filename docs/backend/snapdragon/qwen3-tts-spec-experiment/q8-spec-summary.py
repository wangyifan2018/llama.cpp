#!/usr/bin/env python3
"""Summarize actual greedy speculation and estimate sampled-window performance."""
import argparse
import csv
from pathlib import Path
import runpy
from statistics import median
import struct

import numpy as np

ROOT = Path(__file__).parent
filtered = runpy.run_path(str(ROOT / "train-offset.py"))["filtered"]
window = runpy.run_path(str(ROOT / "window3.py"))
timings, trajectory_cost = (window[key] for key in ("timings", "trajectory_cost"))


def binary(path, kind, width):
    with path.open("rb") as f:
        if struct.unpack("<ii", f.read(8)) != (15, 2048):
            raise ValueError(f"unexpected replay header: {path}")
        data = np.fromfile(f, kind)
    return data.reshape(-1, width)


def actual(run):
    medians = {}
    for mode in ("whole", "16", "8", "2"):
        blocks = []
        all_rows = []
        for block in (1, 2):
            stem = f"q8-rounds-{mode}-{block}"
            ids = binary(run / (stem + ".codes"), "<i4", 15)
            target = binary(run / f"q8-rounds-whole-{block}.codes", "<i4", 15)
            assert len(ids) == len(target) == 41
            with (run / (stem + ".csv")).open() as file:
                rows = list(csv.DictReader(file))
            assert [int(row["frame"]) for row in rows] == list(range(41))
            matched = (ids[1:] == target[1:])
            rows = rows[1:]
            blocks.append(median(float(row["wall_ms"]) for row in rows))
            all_rows.extend((row, ok) for row, ok in zip(rows, matched))
        samples = [float(row["wall_ms"]) for row, _ in all_rows]
        rounds = np.mean([int(row["rounds"]) for row, _ in all_rows])
        drafts = sum(int(row["drafts"]) for row, _ in all_rows)
        accepted = sum(int(row["accepted"]) for row, _ in all_rows)
        exact = sum(ok.all() for _, ok in all_rows)
        tokens = sum(ok.sum() for _, ok in all_rows)
        medians[mode] = median(samples)
        print(f"greedy {mode} median_ms={medians[mode]:.6f} "
              f"blocks_ms={blocks} mean_rounds={rounds:.5f} drafts={drafts} accepted={accepted} "
              f"exact_frames={exact}/{len(all_rows)} exact_codes={tokens}/{len(all_rows)*15} "
              f"speedup_vs_whole={medians['whole']/medians[mode]:.6f}x")
        if mode in ("8", "2"):
            print(f"greedy {mode} speedup_vs_round_ar={medians['16']/medians[mode]:.6f}x")
    return medians


def controls(run):
    target = binary(run / "q8-rounds-whole-1.codes", "<i4", 15)[1:]
    reference = binary(run / "q8-ref-whole.codes", "<i4", 15)[1:]
    print(f"reference_whole_matches_optimized={np.all(target == reference, axis=-1).sum()}/{len(target)}")
    base_ms = None
    for mode in ("whole", "8", "2"):
        ids = binary(run / f"q8-ref-{mode}.codes", "<i4", 15)[1:]
        with (run / f"q8-ref-{mode}.csv").open() as file:
            rows = list(csv.DictReader(file))[1:]
        ms = median(float(row["wall_ms"]) for row in rows)
        if base_ms is None:
            base_ms = ms
        print(f"reference {mode} median_ms={ms:.6f} speedup_vs_reference_whole={base_ms/ms:.6f}x "
              f"exact_frames_vs_reference={np.all(ids == reference, axis=-1).sum()}/{len(ids)} "
              f"exact_frames_vs_optimized={np.all(ids == target, axis=-1).sum()}/{len(ids)}")
    for name in ("q8-window", "q8-ref-window"):
        value = binary(run / (name + ".scores"), "<f4", 2048)
        assert value.shape == (72, 2048)
        for width, start in ((2, 42), (3, 57)):
            delta = np.abs(value[:15] - value[start:start + 15])
            print(f"{name} width={width} fixed_prefix_mean_abs_delta={delta.mean():.9f} "
                  f"fixed_prefix_max_abs_delta={delta.max():.9f}")


def sampled(run):
    data = binary(run / "q8-sampled.scores", "<f4", 29 * 2048).reshape(-1, 29, 2048)
    frames = np.fromfile(run / "q8-capture" / "frames.bin", "<i4").reshape(-1, 1040)
    assert len(data) == len(frames)
    p = filtered(data[:, 1:15])
    q = filtered(data[:, 15:29])
    ids = frames[:, 1026:1040, None]
    observed = np.take_along_axis(p, ids, axis=-1)[..., 0]
    if np.any(observed <= 0):
        raise ValueError("sampled AR has tokens outside replayed target support")
    first = np.minimum(1, np.take_along_axis(q, ids, axis=-1)[..., 0] / observed)
    overlap = np.minimum(p, q).sum(-1).mean(0)
    measured = timings(run / "q8-window.csv")
    base = sum(measured[d, 1] for d in range(2, 16))
    print(f"sampled_ar_frames={len(data)} depth2to15_ar_compute_ms={base:.6f}")
    with (run / "q8-sampled-depths.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("depth", "acceptance", "T1_ms", "T2_ms", "T2_T1"))
        for depth, a in enumerate(overlap, 2):
            narrow = measured[depth, 1]
            wide = measured.get((depth, 2))
            writer.writerow((depth, a, narrow, wide, wide/narrow if wide else None))
            if depth in (2, 5, 8, 11, 14):
                print(f"sampled code{depth} acceptance={a:.6f} T1={narrow:.6f} T2={wide:.6f} "
                      f"T2_T1={wide/narrow:.6f} local_speedup={(1+a)*narrow/wide:.6f}")
    for gate in (2, 5, 8):
        widths = {d: 2 if gate <= d < 15 else 1 for d in range(2, 16)}
        for overhead in (0, 0.5, 1, 2):
            costs = trajectory_cost(measured, (first, None), widths, overhead)
            print(f"sampled gate={gate} overhead_ms={overhead} "
                  f"simulated_ms={costs.mean():.6f} simulated_speedup={base/costs.mean():.6f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--greedy-only", action="store_true")
    args = parser.parse_args()
    actual(args.run)
    if not args.greedy_only:
        controls(args.run)
        sampled(args.run)
