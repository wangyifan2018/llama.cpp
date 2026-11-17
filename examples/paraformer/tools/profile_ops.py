#!/usr/bin/env python3
"""Aggregate Hexagon HTP per-op profile logs into a per-stage cost table.

Produce a log with (on device, one warmup + one measured utterance):

    GGML_HEXAGON_PROFILE=1 GGML_LOG_LEVEL=debug \
      ./bin/llama-paraformer MODEL.gguf WAV --device HTP0:0 --warmup WAV

Then feed the captured stderr to this script:

    python3 tools/profile_ops.py prof.log

The HTP backend emits one `OPBATCH` line per ggml graph submission (encoder
and decoder for Paraformer) followed by `profile-op <OP>|...|usec N` lines.
Each OPBATCH is reported separately; ops are attributed to the most recent
batch. The table shows count, total usec, share of the batch, and mean usec,
which distinguishes kernel-bound cost (large per-op usec) from dispatch-bound
cost (many tiny ops).
"""

import argparse
import collections
import re
import sys

RE_BATCH = re.compile(r"OPBATCH.*?n-ops\s+(\d+).*?usec\s+(\d+)")
RE_OP = re.compile(r"profile-op\s+([A-Z0-9_+]+)\|.*?usec\s+(\d+)")


def parse(path):
    batches = []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_BATCH.search(line)
            if m:
                cur = {"n_ops": int(m.group(1)), "wall_usec": int(m.group(2)), "ops": []}
                batches.append(cur)
                continue
            m = RE_OP.search(line)
            if m and cur is not None:
                cur["ops"].append((m.group(1), int(m.group(2))))
    return batches


def report(batches, top):
    for idx, b in enumerate(batches):
        summed = sum(u for _, u in b["ops"])
        label = "encoder" if idx == 0 else ("decoder" if idx == 1 else f"batch{idx}")
        print(f"=== {label}: submitted n-ops={b['n_ops']}, "
              f"wall={b['wall_usec']/1000:.1f} ms, "
              f"sum(op usec)={summed/1000:.1f} ms, "
              f"timed ops={len(b['ops'])} ===")
        agg = collections.defaultdict(lambda: [0, 0])
        for op, usec in b["ops"]:
            agg[op][0] += 1
            agg[op][1] += usec
        for op, (cnt, usec) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
            share = 100.0 * usec / summed if summed else 0.0
            print(f"  {op:<16s} cnt={cnt:4d} usec={usec:8d} ({share:5.1f}%) "
                  f"avg={usec/cnt:7.1f} us")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="captured HTP profile log")
    ap.add_argument("--top", type=int, default=12, help="rows per batch (default 12)")
    args = ap.parse_args()
    batches = parse(args.log)
    if not batches:
        print("no OPBATCH profile lines found; was GGML_HEXAGON_PROFILE=1 set?", file=sys.stderr)
        return 1
    report(batches, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
