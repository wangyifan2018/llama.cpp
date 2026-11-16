#!/usr/bin/env python3
"""Repeat a service arrival scenario and report latency percentiles.

Runs the socket client many times against a running server and summarises the
client-observed endpoint to final time, which includes the queue and the socket.
Server-side queue and commit time are printed from its own log.
"""

import argparse
import json
import statistics
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = {
    # name: (sessions, extra client arguments)
    "single":   (1, ["--wav", "{short}"]),
    "aligned4": (4, ["--wav", "{short}"]),
    "stagger4": (4, ["--wav", "{long}", "--stagger-ms", "1500"]),
    "mixed4":   (4, ["--wavs", "{short},{short},{long},{long2}"]),
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--short", required=True, help="2 to 3 s clip")
    parser.add_argument("--long", required=True, help="10 s clip")
    parser.add_argument("--long2", required=True, help="20 s clip")
    parser.add_argument("--scenarios", default="single,aligned4,stagger4,mixed4")
    parser.add_argument("--out", default=None, help="write raw JSONL here")
    args = parser.parse_args()

    rows = []
    for name in args.scenarios.split(","):
        sessions, extra = SCENARIOS[name]
        cmd = [sys.executable, os.path.join(HERE, "asr_client.py"), "--port", str(args.port),
               "--sessions", str(sessions), "--paced", "--reps", str(args.reps), "--json"]
        cmd += [a.format(short=args.short, long=args.long, long2=args.long2) for a in extra]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        got = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        for row in got:
            if "error" in row:
                raise SystemExit("%s: %s" % (name, row["error"]))
            row["scenario"] = name
            rows.append(row)
        tail = [r["endpoint_to_final_ms"] for r in got]
        rtt = [r["commit_rtt_ms"] for r in got]
        print("%-9s N=%3d  endpoint p50=%7.1f p95=%7.1f p99=%7.1f max=%7.1f   commit rtt p50=%7.1f p95=%7.1f"
              % (name, len(tail), pct(tail, 50), pct(tail, 95), pct(tail, 99), max(tail),
                 pct(rtt, 50), pct(rtt, 95)))
        # per-stream breakdown so head-of-line is visible
        by_session = {}
        for r in got:
            by_session.setdefault(r["session"], []).append(r["endpoint_to_final_ms"])
        print("          per stream p50: " + ", ".join("s%d=%.0f" % (k, statistics.median(v))
                                                      for k, v in sorted(by_session.items())))
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("raw:", args.out)


if __name__ == "__main__":
    main()
