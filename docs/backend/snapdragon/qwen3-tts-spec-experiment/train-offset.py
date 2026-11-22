#!/usr/bin/env python3
"""Fit one shared hidden-space offset for future RVQ heads, with NumPy."""
import argparse
import csv
from pathlib import Path

import numpy as np


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def filtered(logits, top_k=50, top_p=0.8):
    ids = np.argsort(-logits, axis=-1, kind="stable")[..., :top_k]
    probs = softmax(np.take_along_axis(logits, ids, axis=-1).astype(np.float64))
    probs[np.cumsum(probs, axis=-1) - probs >= top_p] = 0
    probs /= probs.sum(axis=-1, keepdims=True)
    dense = np.zeros(logits.shape, np.float32)
    np.put_along_axis(dense, ids, probs.astype(np.float32), axis=-1)
    return dense


def fit(base, teacher, weights, steps, lr):
    # All examples at a depth share the same logit correction W_g @ offset.
    depth, vocab, hidden = weights.shape
    matrix = weights.reshape(depth * vocab, hidden)
    offset = np.zeros(hidden, np.float32)
    first = np.zeros_like(offset)
    second = np.zeros_like(offset)
    target_mean = teacher.mean(axis=0)
    for step in range(1, steps + 1):
        correction = (matrix @ offset).reshape(depth, vocab)
        pred_mean = softmax(base + correction).mean(axis=0)
        grad = (matrix.T @ (pred_mean - target_mean).ravel()) / depth
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad * grad
        offset -= (lr * first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
    return offset


def load(args, draft_distances=1):
    with args.weights.open("rb") as file:
        depths, vocab, hidden = np.fromfile(file, "<i4", 3)
        weights = np.fromfile(file, "<f2").reshape(depths, vocab, hidden)
    with args.scores.open("rb") as file:
        if tuple(np.fromfile(file, "<i4", 2)) != (depths, vocab):
            raise ValueError("head dimensions differ from score dimensions")
        rows = sum(depths - distance for distance in range(draft_distances + 1))
        scores = np.fromfile(file, "<f4").reshape(-1, rows, vocab)
    with args.manifest.open() as file:
        entries = list(csv.DictReader(file, delimiter="\t"))
    if args.regression:
        entries += [{"id": "e28", "split": "regression"}]
    groups = []
    codes = []
    frame_size = hidden + depths + 1
    for entry in entries:
        path = args.regression if entry["split"] == "regression" else args.capture_dir / (entry["id"] + ".bin")
        frames = np.fromfile(path, "<i4").reshape(-1, frame_size)
        if not len(frames):
            raise ValueError(f"empty capture {path}")
        groups.extend([(entry["split"], entry["id"])] * len(frames))
        codes.append(frames[:, hidden:])
    if len(groups) != len(scores):
        raise ValueError("capture frame count does not match scores")
    if depths != 15 or vocab != 2048:
        raise ValueError("this experiment expects fifteen 2048-entry predictor heads")
    return groups, np.concatenate(codes), scores, weights


def main(args):
    groups, codes, scores, weights = load(args)
    # Predict depths 2..14. Depth 15 cannot earn an extra code with window 2.
    p, q = scores[:, 1:14], scores[:, 15:28]
    weights = weights[1:14].astype(np.float32)
    train = np.array([split == "train" for split, _ in groups])
    hard = np.eye(p.shape[-1], dtype=np.float32)[codes[train, 2:15]]
    models = {
        "zero": np.zeros(weights.shape[-1], np.float32),
        "ce": fit(q[train], hard, weights, args.steps, args.lr),
        "kl": fit(q[train], softmax(p[train]), weights, args.steps, args.lr),
    }
    p_support = filtered(p)
    if not np.all(np.take_along_axis(p_support, codes[:, 2:15, None], axis=-1) > 0):
        raise ValueError("AR sample falls outside replayed target support")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-frame.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "split", "utterance", "frame", "depth", "overlap", "top1_match"])
        for name, offset in models.items():
            correction = weights @ offset
            np.save(args.output / (name + "-offset.npy"), offset)
            np.save(args.output / (name + "-logit-bias.npy"), correction)
            adjusted = q + correction
            overlap = np.minimum(p_support, filtered(adjusted)).sum(axis=-1)
            match = p.argmax(axis=-1) == adjusted.argmax(axis=-1)
            for i, (split, utterance) in enumerate(groups):
                writer.writerows((name, split, utterance, i, d, float(a), int(m))
                                 for d, (a, m) in enumerate(zip(overlap[i], match[i]), 2))
            for split in sorted({split for split, _ in groups}):
                mask = np.array([item == split for item, _ in groups])
                by_depth = overlap[mask].mean(axis=0)
                print(f"{name} {split} frames={sum(mask)} overlap={overlap[mask].mean():.6f} "
                      f"late={by_depth[6:].mean():.6f} top1={match[mask].mean():.6f} "
                      f"depth2to14={','.join(f'{x:.5f}' for x in by_depth)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("offset-prompts.tsv"))
    parser.add_argument("--regression", type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    main(parser.parse_args())
