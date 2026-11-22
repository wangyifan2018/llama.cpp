#!/usr/bin/env python3
"""Calibrate a pruned AR predictor and score forced and free prefixes separately."""
import argparse
import csv
from pathlib import Path
import runpy
import struct
from statistics import median

import numpy as np

filtered = runpy.run_path(str(Path(__file__).with_name("train-offset.py")))["filtered"]


def read(path):
    with path.open("rb") as file:
        if struct.unpack("<ii", file.read(8)) != (15, 2048):
            raise ValueError(f"unexpected logits shape in {path}")
        return np.fromfile(file, "<f4").reshape(-1, 15, 2048)


def fit(source, target, rank):
    n = len(source)
    observed = source[np.arange(n), np.arange(n) % source.shape[1]]
    _, sigma, basis = np.linalg.svd(observed - observed.mean(0), full_matrices=False)
    projection = (basis[:rank] * (np.sqrt(n) / sigma[:rank, None])).astype(np.float32)
    features = source @ projection.T
    mean = features.mean(axis=0)
    features -= mean
    difference = target - source
    difference -= difference.mean(-1, keepdims=True)
    bias = difference.mean(0)
    maps = []
    for j in range(source.shape[1]):
        x = features[:, j]
        matrix = np.linalg.solve(x.T @ x + np.eye(rank, dtype=np.float32) * (0.5 * n),
                                 x.T @ (difference[:, j] - bias[j])).T
        maps.append(matrix)
        bias[j] -= matrix @ mean[j]
    return projection, np.asarray(maps), bias


def main(args):
    run = args.run
    p = read(args.teacher)
    if len(p) != 576:
        raise ValueError("expected the E29 train/eval/regression partition")
    if args.capture:
        f = np.fromfile(args.capture, "<i4").reshape(len(p), 1040)
        f[:, 1025:1040] = p.argmax(-1)
        f.tofile(run / "teacher-prefix.bin")
        f[320:].tofile(run / "eval.bin")
        return

    q = read(run / "forced.scores")
    if len(q) != len(p):
        raise ValueError("forced prefix frame count mismatch")
    proj, maps, bias = fit(q[:320], p[:320], args.rank)
    with (run / "correction.bin").open("wb") as file:
        file.write(struct.pack("<i", args.rank))
        for arr in (proj, maps, bias):
            arr.astype("<f4").tofile(file)

    estimated = q[320:] + np.einsum("bgr,gvr->bgv", q[320:] @ proj.T, maps, optimize=True) + bias
    if args.check_runtime:
        device = read(run / "forced-fit.scores")
        np.testing.assert_allclose(device, estimated, atol=1e-4, rtol=0)
        assert np.array_equal(device.argmax(-1), estimated.argmax(-1))
        print("Corrected host runtime max absolute logit delta", np.max(np.abs(device - estimated)))
    variants = [("forced_raw", q[320:]), ("forced_fit", estimated)]
    if args.check_runtime:
        variants += [("free_raw", read(run / "free.scores")), ("free_fit", read(run / "free-fit.scores"))]
    with (run / "depths.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "method", "depth", "overlap", "top1", "prefix_survival"])
        for name, idx in (("eval", slice(0, 160)), ("regression", slice(160, 256))):
            ref = p[320:][idx]
            dist = filtered(ref)
            for label, pred in variants:
                logits = pred[idx]
                overlap = np.minimum(dist, filtered(logits)).sum(-1)
                top1 = ref.argmax(-1) == logits.argmax(-1)
                prefix = np.logical_and.accumulate(top1, axis=1).mean(0) if label.startswith("free") else np.full(15, np.nan)
                for i in range(15):
                    writer.writerow((name, label, i + 1, overlap[:, i].mean(), top1[:, i].mean(), prefix[i]))
                print(f"{name} {label} overlap={overlap.mean():.6f} top1={top1.mean():.6f} "
                      f"code1_top1={top1[:,0].mean():.6f} exact_prefix4={prefix[3]:.6f} exact_frame={prefix[-1]:.6f}")

    if args.ablation:
        ref = p[320:480]
        dist = filtered(ref)
        for layers in ("014", "0134"):
            logits = read(run / f"forced-{layers}.scores")[:160]
            overlap = np.minimum(dist, filtered(logits)).sum(-1)
            top1 = ref.argmax(-1) == logits.argmax(-1)
            print(f"eval forced_layers_{layers} overlap={overlap.mean():.6f} top1={top1.mean():.6f} "
                  f"code1_top1={top1[:,0].mean():.6f}")

    if args.android:
        times = {}
        for mode in ("ar", "small", "fit"):
            with (run / f"e34-{mode}.csv").open() as f:
                samples = [float(row["compute_ms"]) for row in csv.DictReader(f)]
            assert len(samples) == 30
            times[mode] = median(samples)
            print(f"android {mode} median_ms={times[mode]:.6f} speedup={times['ar']/times[mode]:.6f}x "
                  f"min_ms={min(samples):.6f} max_ms={max(samples):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--android", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    main(parser.parse_args())
