#!/usr/bin/env python3
"""Compare scheduling policies on the same traces (batch 2).

Reads the tree run_device.sh produces when QWEN_SCHEDULE is set:

    OUT/<schedule>/<trace>.<capacity>.<round>.jsonl

and reports, per schedule: makespan, the first request's time to final, the
arrival-to-final percentiles, decode calls, paired calls, and whether every
capacity-2 run produced the same tokens as the capacity-1 control of that round.

    python3 compare_schedules.py --dir /tmp/qwen-b2
"""

import argparse
import json
import statistics
from pathlib import Path

SCHEDULES = ["blocking", "decode-first", "interleaved", "idle"]
TRACES = {"simultaneous": 2, "staggered": 2, "refill": 3}


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    index = q * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def read_run(path, capacity, requests):
    rows, summary = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event"] == "finish":
            rows[event["id"]] = event
        elif event["event"] == "summary":
            summary = event
    if summary is None or len(rows) != requests:
        return None, None
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rounds", type=int, default=0)
    args = ap.parse_args()

    base = Path(args.dir)
    report = {}
    for schedule in SCHEDULES:
        folder = base / schedule
        if not folder.is_dir():
            continue
        per_trace = {}
        for trace, requests in TRACES.items():
            makespan, first, latencies, calls, paired = [], [], [], [], []
            phases = {"mel": [], "encode": [], "prefill": [], "decode_prefill": []}
            parity = True
            rounds = sorted(int(p.name.split(".")[2]) for p in folder.glob(f"{trace}.2.*.jsonl"))
            if args.rounds:
                rounds = rounds[:args.rounds]
            for round_id in rounds:
                serial, _ = read_run(folder / f"{trace}.1.{round_id}.jsonl", 1, requests)
                pair, summary = read_run(folder / f"{trace}.2.{round_id}.jsonl", 2, requests)
                if pair is None or summary is None or serial is None:
                    continue
                for i, baseline in serial.items():
                    if pair[i]["tokens"] != baseline["tokens"]:
                        parity = False
                makespan.append(summary["makespan_ms"])
                calls.append(summary["decode_calls"])
                paired.append(summary["paired_calls"])
                first.append(min(r["finish_ms"] for r in pair.values()))
                for r in pair.values():
                    latencies.append(r["finish_ms"] - r["arrival_ms"])
                    phases["mel"].append(r["prepare_ms"])
                    phases["encode"].append(r["encode_ms"])
                    phases["prefill"].append(r["prefill_ms"])
                    phases["decode_prefill"].append(r["decoder_prefill_ms"])
            if not makespan:
                continue
            per_trace[trace] = {
                "rounds": len(makespan),
                "makespan_ms": statistics.median(makespan),
                "first_final_ms": statistics.median(first),
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "p99_ms": percentile(latencies, 0.99),
                "decode_calls": statistics.median(calls),
                "paired_calls": statistics.median(paired),
                "token_parity": parity,
                "phase_ms": {k: statistics.median(v) for k, v in phases.items() if v},
            }
        report[schedule] = per_trace

    if not report:
        raise SystemExit("no completed runs found under %s" % base)

    baseline = report.get("blocking", {})
    print("| trace | schedule | makespan | first final | P50 | P95 | P99 | decode | paired | parity |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for trace in TRACES:
        for schedule, per_trace in report.items():
            row = per_trace.get(trace)
            if not row:
                continue
            delta = ""
            if schedule != "blocking" and trace in baseline:
                base_row = baseline[trace]
                dm = row["makespan_ms"] - base_row["makespan_ms"]
                df = row["first_final_ms"] - base_row["first_final_ms"]
                delta = " (%+.0f / %+.0f)" % (dm, df)
            print("| %s | %s | %.0f%s | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %s |" % (
                trace, schedule, row["makespan_ms"], delta, row["first_final_ms"],
                row["p50_ms"], row["p95_ms"], row["p99_ms"], row["decode_calls"], row["paired_calls"],
                "ok" if row["token_parity"] else "MISMATCH"))
    print()
    print("| trace | schedule | mel | audio encoder | decoder prefill | prefill total |")
    print("|---|---|---:|---:|---:|---:|")
    for trace in TRACES:
        for schedule, per_trace in report.items():
            row = per_trace.get(trace)
            if not row or "phase_ms" not in row:
                continue
            ph = row["phase_ms"]
            print("| %s | %s | %.0f | %.0f | %.0f | %.0f |" % (
                trace, schedule, ph.get("mel", 0), ph.get("encode", 0),
                ph.get("decode_prefill", 0), ph.get("prefill", 0)))
    print()
    print("delta column: makespan / first-final against the blocking baseline")
    out = base / "schedule-report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
