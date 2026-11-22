#!/usr/bin/env python3
"""Summarize E33 Android timings and verify shared greedy prefixes."""
import argparse
import csv
from pathlib import Path
from statistics import median

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
run = parser.parse_args().run


def logits(name):
    return np.fromfile(run / f"e33-{name}.scores", "<f4", offset=8).reshape(15, 2048)


ar, base, raw, fit, ar9 = [logits(name) for name in ("ar", "base", "split-raw", "split-fit", "ar9")]
assert np.array_equal(ar[:5], base[:5])
assert np.array_equal(base[:9], raw[:9])
assert np.array_equal(raw[:10], fit[:10])
assert np.array_equal(ar[:10], ar9[:10])
print("Android verified unchanged grouped prefix and exact AR9 logits through code10.")
measurements = {}
for name in ("ar", "base", "split-raw", "split-fit", "ar9"):
    with (run / f"e33-{name}.csv").open() as file:
        times = [float(row["compute_ms"]) for row in csv.DictReader(file)]
    assert len(times) == 30
    measurements[name] = median(times)
    print(f"{name} n={len(times)} median_ms={median(times):.6f} min_ms={min(times):.6f} max_ms={max(times):.6f}")
for name, elapsed in measurements.items():
    print(f"{name} vs_ar={measurements['ar']/elapsed:.6f}x")
