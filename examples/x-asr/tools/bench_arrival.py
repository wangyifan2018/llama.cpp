#!/usr/bin/env python3
"""Drive one arrival scenario matrix against a running ASR server.

Scenarios cover the cabin cases: one stream, four starting together, four with
staggered starts, mixed lengths, and a VAD-like burst where each segment arrives
at once. Every request is recorded, so percentiles are computed over requests,
never inferred from a single-stream RTF.

    python3 bench_arrival.py --port 8765 --label htp-b1 \
        --short zh.wav --short-en en.wav --long zh-10.wav --long-en en-20.wav
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


SCENARIOS = [
    # name, sessions, extra client arguments
    ("single", 1, ["--wav", "{short}", "--paced"]),
    ("aligned4", 4, ["--wav", "{short}", "--paced"]),
    ("stagger4", 4, ["--wav", "{short}", "--paced", "--start-offsets-ms", "0,250,500,750"]),
    ("mixed4", 4, ["--wavs", "{short},{short_en},{long},{long_en}", "--paced"]),
    ("burst4", 4, ["--wav", "{short}", "--burst"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--short", required=True)
    parser.add_argument("--short-en", required=True)
    parser.add_argument("--long", required=True)
    parser.add_argument("--long-en", required=True)
    parser.add_argument("--scenarios", default=",".join(s[0] for s in SCENARIOS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    wanted = set(args.scenarios.split(","))
    rows = []
    for scenario, sessions, extra in SCENARIOS:
        if scenario not in wanted:
            continue
        cmd = [sys.executable, os.path.join(HERE, "asr_client.py"), "--host", args.host,
               "--port", str(args.port), "--sessions", str(sessions), "--reps", "1", "--json"]
        cmd += [a.format(short=args.short, short_en=args.short_en, long=args.long, long_en=args.long_en)
                for a in extra]
        for rep in range(args.reps):
            out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
            got = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
            errors = [r for r in got if "error" in r]
            if errors:
                print("%s %s rep %d: %d errors" % (args.label, scenario, rep, len(errors)))
                continue
            for r in got:
                r["arm"] = args.label
                r["scenario"] = scenario
                r["round"] = rep
                rows.append(r)

    print("%-10s %-10s %4s %8s %8s %8s %8s" % ("arm", "scenario", "N", "p50", "p95", "p99", "max"))
    for scenario, _, _ in SCENARIOS:
        if scenario not in wanted:
            continue
        vals = [r["endpoint_to_final_ms"] for r in rows if r["scenario"] == scenario]
        if not vals:
            continue
        print("%-10s %-10s %4d %8.1f %8.1f %8.1f %8.1f"
              % (args.label, scenario, len(vals), pct(vals, 50), pct(vals, 95), pct(vals, 99), max(vals)))
        per = {}
        for r in rows:
            if r["scenario"] == scenario:
                per.setdefault(r["session"], []).append(r["endpoint_to_final_ms"])
        print("           per stream p50: " + ", ".join("s%d=%.0f" % (k, statistics.median(v))
                                                        for k, v in sorted(per.items())))
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("raw:", args.out)


if __name__ == "__main__":
    main()
