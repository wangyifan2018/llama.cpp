#!/usr/bin/env python3
"""Check that a phrase bias cannot cross between concurrent streams.

`--hotwords` loads one list into the model and every stream keeps its own
token history, so a bias is supposed to be per stream. That is an argument
about the code; this is the measurement. Run each clip alone, then run all of
them at the same endpoint, and require the same transcript both times.

The negative clips matter as much as the commands: if a bias leaked, a
statement that merely names a device would come back with a direction.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(port, wavs, sessions, host="127.0.0.1"):
    """One client invocation; returns the per-session records it printed."""
    cmd = [sys.executable, os.path.join(HERE, "asr_client.py"),
           "--host", host, "--port", str(port), "--sessions", str(sessions),
           "--segment", "--json"]
    if sessions == 1:
        cmd += ["--wav", wavs[0]]
    else:
        cmd += ["--wavs", ",".join(wavs)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--wavs", required=True, help="comma separated, one per session")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json", help="write every record here")
    args = parser.parse_args()

    wavs = [w.strip() for w in args.wavs.split(",") if w.strip()]
    if len(wavs) < 2:
        raise SystemExit("need at least two clips to compare against each other")

    baseline = {}
    for wav in wavs:
        record = run(args.port, [wav], 1, args.host)[0]
        baseline[os.path.basename(wav)] = record["text"]
        print("alone  %-14s %s" % (os.path.basename(wav), record["text"]))

    diffs = 0
    records = []
    for rnd in range(args.rounds):
        for record in run(args.port, wavs, len(wavs), args.host):
            name = os.path.basename(record["wav"])
            want = baseline[name]
            same = record["text"] == want
            records.append(dict(record, round=rnd, alone=want, same=same))
            if not same:
                diffs += 1
                print("DIFF   round %d %-14s alone=%s together=%s" % (
                    rnd, name, want, record["text"]))
        print("round %d: %d/%d identical to the single-stream transcript, "
              "endpoint->final %.0f..%.0f ms" % (
                  rnd, sum(1 for r in records if r["round"] == rnd and r["same"]),
                  len(wavs),
                  min(r["endpoint_to_final_ms"] for r in records if r["round"] == rnd),
                  max(r["endpoint_to_final_ms"] for r in records if r["round"] == rnd)))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=1)
    if diffs:
        raise SystemExit("%d of %d concurrent transcripts differ from alone" % (
            diffs, len(records)))
    print("%d concurrent transcripts over %d rounds match their single-stream text" % (
        len(records), args.rounds))


if __name__ == "__main__":
    main()
