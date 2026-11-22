#!/usr/bin/env python3
"""Fit and verify the second group of the E33 predictor."""
import argparse
import csv
from pathlib import Path
import runpy
import struct

import numpy as np

shared = runpy.run_path(str(Path(__file__).with_name("tail-student.py")))
fit, filtered, read = (shared[key] for key in ("fit", "filtered", "read"))


def export(path, params):
    projection, weights, bias = params
    rank = projection.shape[0]
    with path.open("wb") as stream:
        stream.write(struct.pack("<i", rank))
        projection.astype("<f4").tofile(stream)
        np.pad(weights, ((5, 0), (0, 0), (0, 0))).astype("<f4").tofile(stream)
        np.pad(bias, ((5, 0), (0, 0))).astype("<f4").tofile(stream)


def corrected(logits, params):
    proj, weights, bias = params
    z = logits[:, 9] @ proj.T
    tail = logits[:, 10:] + np.einsum("br,gvr->bgv", z, weights, optimize=True) + bias
    return np.concatenate((logits[:, :10], tail), axis=1)


def main(args):
    run = args.run
    target = read(run / "greedy.scores", 15)
    if args.oracle_input:
        frames = np.fromfile(args.oracle_input, "<i4").reshape(-1, 1040)
        if len(frames) != len(target):
            raise ValueError("greedy logits and capture counts differ")
        frames[:, 1029:1034] = target[:, 4:9].argmax(-1)
        frames.tofile(run / "oracle.bin")
        return

    self_prefix, oracle = (read(run / f"{name}.scores", 15) for name in ("raw", "oracle"))
    baseline = read(args.e32 / "eval-tail.scores", 15)
    assert len(target) == len(oracle) == len(self_prefix) == 576
    assert np.array_equal(oracle[:, 9], target[:, 9])
    assert np.array_equal(self_prefix[:, :9], oracle[:, :9])
    assert np.array_equal(self_prefix[:, :5], target[:, :5])
    assert np.array_equal(self_prefix[320:480, :9], baseline[:, :9])

    self_fit = fit(self_prefix[:320, 9], self_prefix[:320, 10:], target[:320, 10:], 32, 0.5)
    oracle_fit = fit(oracle[:320, 9], oracle[:320, 10:], target[:320, 10:], 32, 0.5)
    export(run / "second.bin", self_fit)
    export(run / "second-oracle.bin", oracle_fit)
    ctrl = target[320:480]
    raw = self_prefix[320:480]
    ideal = oracle[320:480]
    self_trained = corrected(raw, self_fit)
    np.save(run / "expected.npy", self_trained)
    if args.check_runtime:
        runtime = read(run / "eval-adapted.scores", 15)
        assert np.array_equal(runtime[:, :10], self_trained[:, :10])
        np.testing.assert_allclose(runtime[:, 10:], self_trained[:, 10:], atol=1e-4, rtol=0)
        assert np.array_equal(runtime.argmax(-1), self_trained.argmax(-1))
        print("Mac adjusted max absolute logit delta", np.max(np.abs(runtime[:, 10:] - self_trained[:, 10:])))

    if args.check_ar9:
        exact = read(run / "eval-ar9.scores", 15)
        expected_exact = corrected(ideal, oracle_fit)
        assert np.array_equal(exact[:, :10], ctrl[:, :10])
        np.testing.assert_allclose(exact[:, 10:], expected_exact[:, 10:], atol=1e-4, rtol=0)
        print("Mac AR9 max absolute late logit delta", np.max(np.abs(exact[:, 10:] - expected_exact[:, 10:])))
    expected = filtered(ctrl)
    with (run / "depths.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "depth", "overlap", "top1"])
        for name, logits in (("one_tail", baseline), ("two_groups_raw", raw),
                             ("two_groups_self_fit", self_trained),
                             ("two_groups_oracle_fit_at_self", corrected(raw, oracle_fit)),
                             ("two_groups_oracle_raw", ideal), ("two_groups_oracle_fit", corrected(ideal, oracle_fit))):
            overlap = np.minimum(expected, filtered(logits)).sum(-1)
            match = ctrl.argmax(-1) == logits.argmax(-1)
            for depth, (o, m) in enumerate(zip(overlap.mean(0), match.mean(0)), 1):
                writer.writerow((name, depth, o, m))
            print(f"{name} code5to15_overlap={overlap[:, 4:].mean():.6f} code10_overlap={overlap[:, 9].mean():.6f} "
                  f"code10to15_overlap={overlap[:, 9:].mean():.6f} code11to15_overlap={overlap[:, 10:].mean():.6f} "
                  f"code10to15_top1={match[:, 9:].mean():.6f}")
    expected_codes = ctrl[:, 4:9].argmax(-1)
    actual_codes = raw[:, 4:9].argmax(-1)
    print("first_group_top1", (expected_codes == actual_codes).mean(0))
    print("first_group_exact_prefix", np.mean(np.all(expected_codes == actual_codes, axis=1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("e32", type=Path)
    parser.add_argument("--oracle-input", type=Path)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--check-ar9", action="store_true")
    main(parser.parse_args())
