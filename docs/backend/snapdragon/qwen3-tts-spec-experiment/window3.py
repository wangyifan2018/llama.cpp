#!/usr/bin/env python3
"""Train two draft offsets and replay sequential acceptance on AR trajectories."""
import argparse
import csv
from pathlib import Path
import runpy
from statistics import median

import numpy as np

shared = runpy.run_path(str(Path(__file__).with_name("train-offset.py")))
filtered, fit, load, softmax = (shared[key] for key in ("filtered", "fit", "load", "softmax"))


def timings(path):
    data = {}
    with path.open() as file:
        for row in csv.DictReader(file):
            key = (int(row["depth"]), int(row["width"]))
            data.setdefault(key, []).append(float(row["compute_ms"]))
    return {key: median(samples) for key, samples in data.items()}


def round_rates(p, q1, q2, codes):
    ids = codes[:, 2:16, None]
    actual = np.take_along_axis(p, ids, axis=-1)[..., 0]
    if (actual <= 0).any():
        raise ValueError("AR trajectory contains a token outside target support")
    take1 = np.minimum(1, np.take_along_axis(q1, ids, axis=-1)[..., 0] / actual)
    take2 = np.minimum(1, np.take_along_axis(q2, ids[:, 1:], axis=-1)[..., 0] / actual[:, 1:])
    overlap1 = np.minimum(p, q1).sum(axis=-1)
    overlap2 = np.minimum(p[:, 1:], q2).sum(axis=-1)
    # E_AR[A(x)*overlap2(x)] integrates over the first emitted token x.
    joint = take1[:, :-1] * overlap2
    return overlap1, joint, take1, take1[:, :-1] * take2


def transition(first, both, depth, width, costs):
    if width == 1:
        return costs[depth + 1]
    if width == 2:
        return (1 - first) * costs[depth + 1] + first * costs[depth + 2]
    return ((1 - first) * costs[depth + 1] + (first - both) * costs[depth + 2] +
            both * costs[depth + 3])


def policy(times, first, both, max_width, overhead):
    costs = np.zeros(17)
    widths = {}
    for depth in range(15, 1, -1):
        a = first[depth - 2]
        joint = both[depth - 2] if depth <= 13 else 0
        choices = []
        for width in range(1, min(max_width, 16 - depth) + 1):
            cost = times[depth, width] + (overhead if width > 1 else 0)
            choices.append((cost + transition(a, joint, depth, width, costs), width))
        costs[depth], widths[depth] = min(choices)
    return widths


def trajectory_cost(times, rates, widths, overhead):
    first, both = rates
    costs = np.zeros((17, len(first)))
    for depth in range(15, 1, -1):
        width = widths[depth]
        a = first[:, depth - 2]
        joint = both[:, depth - 2] if width == 3 else 0
        costs[depth] = (times[depth, width] + (overhead if width > 1 else 0) +
                        transition(a, joint, depth, width, costs))
    return costs[2]


def main(args):
    groups, codes, scores, weights = load(args, draft_distances=2)
    train = np.array([split == "train" for split, _ in groups])
    splits = {name: np.array([split == name for split, _ in groups])
              for name in ("train", "eval", "regression")}
    target_logits = scores[:, 1:15]
    p = filtered(target_logits)
    q1, q2 = scores[:, 15:29], scores[:, 29:42]
    w1, w2 = weights[1:].astype(np.float32), weights[2:].astype(np.float32)
    # Reuse E29's existing first-offset training range and held-out split.
    offset1 = fit(q1[train, :-1], softmax(target_logits[train, :-1]), w1[:-1], args.steps, args.lr)
    # The last depth can serve as a tail-independent calibration target for the same b2.
    offset2 = fit(q2[train], softmax(target_logits[train, 1:]), w2, args.steps, args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, offset in enumerate((offset1, offset2), 1):
        np.save(args.output / f"offset{index}.npy", offset)

    times = timings(args.timing)
    raw1, trained1 = filtered(q1), filtered(q1 + w1 @ offset1)
    raw2, trained2 = filtered(q2), filtered(q2 + w2 @ offset2)
    stages = {"raw": (raw1, raw2), "b1": (trained1, raw2), "b1+b2": (trained1, trained2)}
    base = sum(times[d, 1] for d in range(2, 16))
    print(f"sum_ar_depth2to15_ms={base:.5f}")
    with (args.output / "depths.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "split", "depth", "first_accept", "both_accept", "E_codes_window3"])
        for name, (first, second) in stages.items():
            a, b, sampled_a, sampled_b = round_rates(p, first, second, codes)
            for split, mask in splits.items():
                a_mean, b_mean = a[mask].mean(axis=0), b[mask].mean(axis=0)
                for d in range(2, 14):
                    writer.writerow((name, split, d, a_mean[d-2], b_mean[d-2], 1 + a_mean[d-2] + b_mean[d-2]))
                mid = slice(6, 12)
                print(f"{name} {split} depth8to13 first={a_mean[mid].mean():.5f} "
                      f"joint={b_mean[mid].mean():.5f} codes={1+a_mean[mid].mean()+b_mean[mid].mean():.5f}")
            for overhead in (0, 0.5, 1, 2):
                for width in (2, 3):
                    chosen = policy(times, a[train].mean(axis=0), b[train].mean(axis=0), width, overhead)
                    costs = trajectory_cost(times, (sampled_a, sampled_b), chosen, overhead)
                    encoded = "".join(str(chosen[d]) for d in range(2, 16))
                    print(f"{name} max_width={width} overhead_ms={overhead} policy={encoded} "
                          f"eval_speedup={base / costs[splits['eval']].mean():.5f} "
                          f"regression_speedup={base / costs[splits['regression']].mean():.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("timing", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("offset-prompts.tsv"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    main(parser.parse_args())
