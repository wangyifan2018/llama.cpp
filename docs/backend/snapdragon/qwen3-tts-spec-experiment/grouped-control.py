#!/usr/bin/env python3
"""Replay grouped late predictions from an AR greedy prefix on the E29 eval split."""
import argparse
from pathlib import Path
import runpy

import numpy as np

filtered = runpy.run_path(str(Path(__file__).with_name("train-offset.py")))["filtered"]


def read(path, rows):
    return np.fromfile(path, "<f4", offset=8).reshape(-1, rows, 2048)


parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path, help="E31 run directory")
parser.add_argument("distillation", type=Path, help="E29 run directory")
parser.add_argument("--prepare", action="store_true")
args = parser.parse_args()
run = args.run
greedy = read(run / "eval-ar-greedy.scores", 15)
if args.prepare:
    frames = np.fromfile(run / "eval.bin", "<i4").reshape(len(greedy), 1040)
    frames[:, 1025:1029] = greedy[:, :4].argmax(axis=-1)
    frames.tofile(run / "eval-greedy-prefix.bin")
else:
    mask = slice(320, 480)
    original = read(args.distillation / "offset.scores", 29)[mask, :15]
    teacher = read(run / "teacher.scores", 15)[mask]
    student = read(run / "student.scores", 15)[mask]
    grouped_greedy = read(run / "eval-greedy-prefix.scores", 15)
    assert np.array_equal(grouped_greedy[:, 4], greedy[:, 4])
    bias = np.fromfile(run / "results" / "early-bias.bin", "<f4").reshape(4, 2048)
    early = (student[:, :4] + bias).argmax(axis=-1)
    greedy_codes = greedy[:, :4].argmax(axis=-1)
    print("stage1_match_vs_full_ar_greedy", *(f"{x:.6f}" for x in (early == greedy_codes).mean(0)))
    print("stage1_exact_prefix_match_vs_full_ar_greedy", np.all(early == greedy_codes, axis=-1).mean())
    orig_p, greedy_p = filtered(original[:, 4:]), filtered(greedy[:, 4:])
    for name, logits in (("sampled_ar_prefix", teacher), ("greedy_ar_prefix", grouped_greedy),
                         ("grouped_student_prefix", student)):
        q = filtered(logits[:, 4:])
        orig = np.minimum(orig_p, q).sum(-1)
        ref = np.minimum(greedy_p, q).sum(-1)
        print(f"{name} vs_sampled_ar={orig.mean():.6f} vs_greedy_ar={ref.mean():.6f} "
              f"code5_vs_sampled_ar={orig[:, 0].mean():.6f} code5_vs_greedy_ar={ref[:, 0].mean():.6f}")
