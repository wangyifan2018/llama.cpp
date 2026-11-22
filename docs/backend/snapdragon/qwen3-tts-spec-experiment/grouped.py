#!/usr/bin/env python3
"""Fit draft head offsets for a frozen 4+11 predictor and evaluate prefix shift."""
import argparse
import csv
from pathlib import Path
import runpy

import numpy as np

shared = runpy.run_path(str(Path(__file__).with_name("train-offset.py")))
filtered, fit, load, softmax = (shared[key] for key in ("filtered", "fit", "load", "softmax"))


def read_groups(path, frames, vocab):
    with path.open("rb") as file:
        if tuple(np.fromfile(file, "<i4", 2)) != (15, vocab):
            raise ValueError("unexpected grouped score header")
        return np.fromfile(file, "<f4").reshape(frames, 15, vocab)


def main(args):
    groups, codes, scores, weights = load(args)
    train = np.array([split == "train" for split, _ in groups])
    p = scores[:, :15]
    original = read_groups(args.teacher, len(groups), p.shape[-1])
    for depth in (0, 4):
        if not np.array_equal(original[:, depth], p[:, depth]):
            raise ValueError(f"group anchor logits differ at code{depth + 1}")
    early_w = weights[1:4].astype(np.float32)
    early = fit(original[train, 1:4], softmax(p[train, 1:4]), early_w, args.steps, args.lr)
    early_bias = np.vstack((np.zeros((1, p.shape[-1]), np.float32), early_w @ early))
    args.output.mkdir(exist_ok=True, parents=True)
    early_bias.astype("<f4").tofile(args.output / "early-bias.bin")
    if args.student is None:
        return

    generated = read_groups(args.student, len(groups), p.shape[-1])
    if not np.array_equal(original[:, :4], generated[:, :4]):
        raise ValueError("stage 1 logits differ across prefix conditions")
    late_w = weights[5:].astype(np.float32)
    true_prefix = fit(original[train, 5:], softmax(p[train, 5:]), late_w, args.steps, args.lr)
    self_prefix = fit(generated[train, 5:], softmax(p[train, 5:]), late_w, args.steps, args.lr)
    np.save(args.output / "early-offset.npy", early)
    np.save(args.output / "teacher-late-offset.npy", true_prefix)
    np.save(args.output / "student-late-offset.npy", self_prefix)
    p_support = filtered(p)
    teacher_top1 = p.argmax(axis=-1)
    early_top1 = (original[:, :4] + early_bias).argmax(axis=-1)
    rows = [
        ("teacher_raw", original[:, 4:]),
        ("student_raw", generated[:, 4:]),
        ("teacher_KL", original[:, 4:] + np.vstack((np.zeros_like(early_bias[0]), late_w @ true_prefix))),
        ("student_teacher_KL", generated[:, 4:] + np.vstack((np.zeros_like(early_bias[0]), late_w @ true_prefix))),
        ("student_self_KL", generated[:, 4:] + np.vstack((np.zeros_like(early_bias[0]), late_w @ self_prefix))),
    ]
    with (args.output / "depths.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "method", "depth", "overlap", "teacher_top1", "sampled_match"])
        for split in ("train", "eval", "regression"):
            mask = np.array([name == split for name, _ in groups])
            print(f"{split} frames={mask.sum()} early_teacher_top1={np.mean(early_top1[mask] == teacher_top1[mask, :4]):.6f} "
                  f"early_sampled={np.mean(early_top1[mask] == codes[mask, 1:5]):.6f} "
                  f"exact_sampled_prefix={np.mean(np.all(early_top1[mask] == codes[mask, 1:5], axis=1)):.6f}")
            for name, logits in rows:
                q = logits[mask]
                overlap = np.minimum(filtered(q), p_support[mask, 4:]).sum(axis=-1)
                match = q.argmax(axis=-1) == teacher_top1[mask, 4:]
                sampled = q.argmax(axis=-1) == codes[mask, 5:16]
                for d, vals in enumerate(zip(overlap.mean(0), match.mean(0), sampled.mean(0)), 5):
                    writer.writerow((split, name, d, *vals))
                print(f"{split} {name} late_overlap={overlap.mean():.6f} late_teacher_top1={match.mean():.6f} "
                      f"late_sampled={sampled.mean():.6f} code5_overlap={overlap[:, 0].mean():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("offset-prompts.tsv"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    main(parser.parse_args())
