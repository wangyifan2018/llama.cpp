#!/usr/bin/env python3
"""Check batch lifecycle and token parity, then report measured latency and throughput."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(values)
    index = q * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def read_run(path, slots, requests):
    active = {}
    output = {}
    pairs = Counter()
    summary = None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        event = row["event"]
        if event == "admit":
            assert 0 <= row["slot"] < slots, (path, row)
            assert len(active) < slots and row["slot"] not in active.values(), (path, row)
            assert row["id"] not in active and row["id"] not in output, (path, row)
            active[row["id"]] = row["slot"]
        elif event == "finish":
            assert active.pop(row["id"]) == row["slot"], (path, row)
            assert row["finish_ms"] >= row["first_token_ms"] >= row["admitted_ms"] >= row["arrival_ms"], (path, row)
            assert row["text"].strip() and row["stop"] == "eog", (path, row)
            output[row["id"]] = row
        elif event == "paired_decode":
            pair = tuple(sorted(row["ids"]))
            assert slots == 2 and len(pair) == 2 and pair[0] != pair[1] and all(i in active for i in pair), (path, row)
            assert row["decode_ms"] > 0, (path, row)
            pairs[pair] += 1
        elif event == "summary":
            assert summary is None, path
            summary = row
        else:
            raise ValueError((path, row))
    assert summary and not active and sorted(output) == list(range(requests)), path
    assert summary["capacity"] == slots and summary["requests"] == requests, path
    assert summary["paired_calls"] == sum(pairs.values()) <= summary["decode_calls"], path
    return output, pairs, summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("rounds", type=int)
    args = cli.parse_args()
    if args.rounds < 1:
        cli.error("rounds must be positive")
    report = {}
    for label, requests in (("simultaneous", 2), ("staggered", 2), ("refill", 3)):
        timings = {1: [], 2: []}
        latency = {1: [], 2: []}
        decode = {1: [], 2: []}
        prefill = {1: [], 2: []}
        calls = {1: [], 2: []}
        paired = []
        request_latency = {width: {i: [] for i in range(requests)} for width in (1, 2)}
        for round_id in range(1, args.rounds + 1):
            old, _, serial = read_run(args.output / f"{label}.1.{round_id}.jsonl", 1, requests)
            new, pairs, batched = read_run(args.output / f"{label}.2.{round_id}.jsonl", 2, requests)
            assert (0, 1) in pairs, (label, round_id, pairs)
            if label == "refill":
                assert any(2 in pair for pair in pairs), (label, round_id, pairs)
            paired.append(batched["paired_calls"])
            for i in old:
                assert old[i]["arrival_ms"] == new[i]["arrival_ms"] and old[i]["audio_ms"] == new[i]["audio_ms"], (label, round_id, i)
                assert old[i]["tokens"] == new[i]["tokens"], (label, round_id, i, old[i]["text"], new[i]["text"])
            for width, rows, summary in [(1, old, serial), (2, new, batched)]:
                timings[width].append(summary["makespan_ms"])
                decode[width].append(summary["decode_ms"])
                prefill[width].append(sum(r["prefill_ms"] for r in rows.values()))
                calls[width].append(summary["decode_calls"])
                for i, row in rows.items():
                    duration = row["finish_ms"] - row["arrival_ms"]
                    latency[width].append(duration)
                    request_latency[width][i].append(duration)
        result = {}
        for width in (1, 2):
            span = statistics.median(timings[width])
            result[str(width)] = {"makespan_p50_ms": span, "requests_per_second": requests * 1000 / span,
                                  "decode_p50_ms": statistics.median(decode[width]),
                                  "prefill_p50_ms": statistics.median(prefill[width]),
                                  "decode_calls": calls[width],
                                  "per_request_latency_p50_ms": {i: statistics.median(values) for i, values in request_latency[width].items()},
                                  **{f"latency_p{q}_ms": percentile(latency[width], q / 100) for q in (50, 95, 99)}}
        result["paired_calls"] = paired
        result["speedup"] = result["1"]["makespan_p50_ms"] / result["2"]["makespan_p50_ms"]
        report[label] = result
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
