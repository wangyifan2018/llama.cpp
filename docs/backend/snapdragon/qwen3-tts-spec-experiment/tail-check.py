#!/usr/bin/env python3
"""Check fitted tail logits against ggml and summarize on-device timing."""
import argparse
import csv
from pathlib import Path
from statistics import median

import numpy as np


def read(path):
    return np.fromfile(path, "<f4", offset=8).reshape(-1, 15, 2048)


parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("e31", type=Path)
args = parser.parse_args()
run = args.run
host = read(run / "eval-tail.scores")
ar = read(args.e31 / "eval-ar-greedy.scores")
expected = np.load(run / "results" / "eval-greedy-tail.npy")
assert np.array_equal(host[:, :5], ar[:, :5])
assert np.array_equal(read(run / "e32-tail.scores")[:, :5], read(run / "e32-ar.scores")[:, :5])
np.testing.assert_allclose(host[:, 5:], expected, atol=1e-4, rtol=0)
assert np.array_equal(host[:, 5:].argmax(-1), expected.argmax(-1))
print("Host and Android match same-backend AR at code1–5. Host late argmax matches NumPy.")
print("Host late max absolute logit delta", np.max(np.abs(host[:, 5:] - expected)))
numbers = {}
for mode in ("ar", "tail"):
    times = [float(x["compute_ms"]) for x in csv.DictReader((run / f"e32-{mode}.csv").open())]
    assert len(times) == 30
    numbers[mode] = median(times)
    print(f"{mode} n={len(times)} median_ms={median(times):.6f} min_ms={min(times):.6f} max_ms={max(times):.6f}")
print(f"compute_speedup={numbers['ar'] / numbers['tail']:.6f}")
