#!/usr/bin/env python3
"""Read replay outputs and report paired logits agreement and draft overlap."""
import argparse
import array
import csv
import math
import statistics
import struct
import sys
from pathlib import Path


def distribution(logits, top_k, top_p):
    indices = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:top_k]
    peak = logits[indices[0]]
    weights = [math.exp(logits[i] - peak) for i in indices]
    cutoff = top_p * math.fsum(weights)
    kept = {}
    mass = 0.0
    for index, weight in zip(indices, weights):
        kept[index] = weight
        mass += weight
        if mass >= cutoff:
            break
    return {index: weight / mass for index, weight in kept.items()}


def summarize(scores, timing, output, top_k, top_p, capture):
    with scores.open("rb") as stream:
        depths, vocab = struct.unpack("<ii", stream.read(8))
        data = array.array("f")
        data.frombytes(stream.read())
    if sys.byteorder == "big":
        data.byteswap()
    stride = (3 * depths - 1) * vocab
    if depths < 2 or vocab < top_k or not data or len(data) % stride:
        raise ValueError("invalid or incomplete replay output")
    n_frames = len(data) // stride
    captured_codes = []
    if capture:
        raw = capture.read_bytes()
        row_bytes, remainder = divmod(len(raw), n_frames)
        code_bytes = 4 * (depths + 1)
        if remainder or row_bytes <= code_bytes or row_bytes % 4:
            raise ValueError("capture does not match replay frame count")
        for end in range(row_bytes, len(raw) + 1, row_bytes):
            captured_codes.append(struct.unpack_from(f"<{depths + 1}i", raw, end - code_bytes))

    medians = {}
    with timing.open() as stream:
        groups = {}
        for row in csv.DictReader(stream):
            key = (int(row["depth"]), int(row["width"]))
            groups.setdefault(key, []).append(float(row["compute_ms"]))
        for key, samples in sorted(groups.items()):
            medians[key] = statistics.median(samples)
            print(f"depth {key[0]:2} width {key[1]} n={len(samples):3} "
                  f"p50={medians[key]:7.3f} ms range={min(samples):.3f}..{max(samples):.3f}")

    by_depth = {depth: [] for depth in range(1, depths + 1)}
    outside_support = 0
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "depth", "overlap", "draft_argmax_match", "wide_argmax_match", "wide_abs_max"])
        for frame, base in enumerate(range(0, len(data), stride)):
            for depth in by_depth:
                start = base + (depth - 1) * vocab
                narrow = data[start:start + vocab]
                start += (2 * depths - 1) * vocab
                wide = data[start:start + vocab]
                if not all(map(math.isfinite, narrow)) or not all(map(math.isfinite, wide)):
                    raise ValueError(f"non-finite target at frame {frame}, depth {depth}")
                narrow_id = max(range(vocab), key=narrow.__getitem__)
                same = int(narrow_id == max(range(vocab), key=wide.__getitem__))
                delta = max(abs(a - b) for a, b in zip(narrow, wide))
                overlap = draft_match = None
                p = distribution(narrow, top_k, top_p)
                if captured_codes:
                    outside_support += captured_codes[frame][depth] not in p
                if depth >= 2:
                    start = base + (depths + depth - 2) * vocab
                    draft = data[start:start + vocab]
                    if not all(map(math.isfinite, draft)):
                        raise ValueError(f"non-finite draft at frame {frame}, depth {depth}")
                    q = distribution(draft, top_k, top_p)
                    overlap = math.fsum(min(p.get(index, 0.0), prob) for index, prob in q.items())
                    draft_match = int(narrow_id == max(range(vocab), key=draft.__getitem__))
                record = (overlap, draft_match, same, delta)
                writer.writerow((frame, depth, *record))
                by_depth[depth].append(record)

    print(f"frames={n_frames} top_k={top_k} top_p={top_p} temperature=1")
    if captured_codes:
        print(f"captured_codes_outside_replayed_support={outside_support}/{n_frames * depths}")
    print("depth  overlap  draft_top1  wide_top1  wide_abs_max  T2/T1  approx_round_speedup")
    for depth, rows in by_depth.items():
        a, guess, same, delta = zip(*rows)
        overlap = statistics.mean(a) if depth > 1 else None
        format_score = lambda score: f"{score:7.2%}" if score is not None else "      -"
        print(f"{depth:5}  {format_score(overlap)}  "
              f"{format_score(statistics.mean(guess) if depth > 1 else None):>10}  "
              f"{statistics.mean(same):9.2%}  {max(delta):12.6f}", end="")
        if (depth, 2) in medians:
            ratio = medians[depth, 2] / medians[depth, 1]
            print(f"   {ratio:5.3f}               {(1 + overlap) / ratio:5.3f}", end="")
        print()
    rates = [row[0] for depth, rows in by_depth.items() if 2 <= depth < depths for row in rows]
    print(f"mean_overlap_bonus_eligible={statistics.mean(rates):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("timing", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.8)
    args = parser.parse_args()
    if args.top_k < 1 or not 0 < args.top_p <= 1:
        parser.error("top-k must be positive and top-p must be in (0, 1]")
    summarize(args.scores, args.timing, args.output, args.top_k, args.top_p, args.capture)
