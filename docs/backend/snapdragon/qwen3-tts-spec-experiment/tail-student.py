#!/usr/bin/env python3
"""Fit a low-rank residual on exact-prefix late predictor logits."""
import argparse
import csv
from pathlib import Path
import runpy
import struct

import numpy as np

filtered = runpy.run_path(str(Path(__file__).with_name("train-offset.py")))["filtered"]
VOCAB = 2048


def read(path, rows):
    return np.fromfile(path, "<f4", offset=8).reshape(-1, rows, VOCAB)


def fit(features, base, target, rank, ridge):
    n = len(base)
    mean = features.mean(axis=0)
    centered = features - mean
    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    # Fit on whitened train-only features and fold centering into the logit bias.
    projection = (directions[:rank] * (np.sqrt(n) / singular_values[:rank, None])).astype(np.float32)
    features = centered @ projection.T
    residual = target - base
    residual -= residual.mean(axis=-1, keepdims=True)
    residual = residual.reshape(n, -1)
    bias = residual.mean(axis=0)
    coef = np.linalg.solve(features.T @ features + np.eye(rank, dtype=np.float32) * (n * ridge),
                           features.T @ (residual - bias)).T
    bias -= coef @ (projection @ mean)
    heads = base.shape[1]
    return projection, coef.reshape(heads, VOCAB, rank), bias.reshape(heads, VOCAB)


def main(args):
    ref = read(args.e29 / "offset.scores", 29)[:, :15]
    sample = read(args.e31 / "teacher.scores", 15)
    greedy_ref = read(args.e31 / "eval-ar-greedy.scores", 15)
    greedy_base = read(args.e31 / "eval-greedy-prefix.scores", 15)
    if len(ref) != 576 or not np.array_equal(ref[:, 4], sample[:, 4]) or not np.array_equal(greedy_ref[:, 4], greedy_base[:, 4]):
        raise ValueError("E29/E31 shape or exact code5 anchor mismatch")
    with (args.e29 / "heads.bin").open("rb") as stream:
        dims = struct.unpack("<iii", stream.read(12))
        weights = np.fromfile(stream, "<f2").reshape(dims)
    offset = np.load(args.e31 / "results" / "teacher-late-offset.npy")
    shared_bias = weights[5:].astype(np.float32) @ offset
    proj, coef, bias = fit(sample[:320, 4], sample[:320, 5:] + shared_bias, ref[:320, 5:], args.rank, args.ridge)
    bias += shared_bias
    args.output.mkdir(exist_ok=True, parents=True)
    with (args.output / "tail.bin").open("wb") as stream:
        stream.write(struct.pack("<i", args.rank))
        proj.astype("<f4").tofile(stream)
        coef.astype("<f4").tofile(stream)
        bias.astype("<f4").tofile(stream)

    splits = {"train": (ref[:320], sample[:320]), "eval_sampled": (ref[320:480], sample[320:480]),
              "regression": (ref[480:], sample[480:]), "eval_greedy": (greedy_ref, greedy_base)}
    with (args.output / "depths.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "method", "depth", "overlap", "top1"])
        for name, (p, q) in splits.items():
            expected = filtered(p[:, 5:])
            z = q[:, 4] @ proj.T
            delta = np.einsum("br,gvr->bgv", z, coef, optimize=True) + bias
            corrected = q[:, 5:] + delta
            if name == "eval_greedy":
                np.save(args.output / "eval-greedy-tail.npy", corrected)
            for method, logits in (("frozen", q[:, 5:]), ("shared_KL", q[:, 5:] + shared_bias),
                                   ("low_rank", corrected)):
                overlap = np.minimum(expected, filtered(logits)).sum(-1)
                top1 = p[:, 5:].argmax(-1) == logits.argmax(-1)
                for depth, (o, t) in enumerate(zip(overlap.mean(0), top1.mean(0)), 6):
                    writer.writerow((name, method, depth, o, t))
                print(f"{name} {method} frames={len(p)} overlap={overlap.mean():.6f} top1={top1.mean():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("e29", type=Path)
    parser.add_argument("e31", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=0.5)
    main(parser.parse_args())
