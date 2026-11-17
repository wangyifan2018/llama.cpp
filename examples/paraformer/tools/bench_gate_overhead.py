#!/usr/bin/env python3
"""Measure what the speaker gate costs the tail at a given concurrency.

Runs N requests at a fixed number of parallel clients against the REST endpoint
and reports wall latency percentiles plus the server-side X-Infer-Ms, so the gate
can be compared on and off without changing anything else.

    python3 bench_gate_overhead.py --port 8899 --wav zh.wav --lanes 2 --runs 10
"""

import argparse
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid


def pct(vals, q):
    v = sorted(vals)
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def post_once(url, wav_bytes, name, out, idx):
    boundary = "----" + uuid.uuid4().hex
    body = ("--" + boundary + "\r\n").encode()
    body += ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % name).encode()
    body += b"Content-Type: audio/wav\r\n\r\n" + wav_bytes + b"\r\n"
    body += ("--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    t0 = time.monotonic()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
                infer = float(r.headers.get("X-Infer-Ms", 0))
            out[idx] = {"wall": (time.monotonic() - t0) * 1000.0, "infer": infer}
            return
        except Exception:
            if attempt == 2:
                out[idx] = None
                return
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--label", default="")
    ap.add_argument("--ensure-forward", action="store_true")
    args = ap.parse_args()

    if args.ensure_forward:
        # adb drops forwards when another adb client reconnects, so re-issue it
        # before each run and retry once on a connection error.
        subprocess.run(["adb", "forward", "tcp:%d" % args.port, "tcp:%d" % args.port],
                       capture_output=True, check=False)

    url = "http://%s:%d/v1/audio/transcriptions" % (args.host, args.port)
    data = open(args.wav, "rb").read()
    name = args.wav.split("/")[-1]
    walls, infers = [], []
    for _ in range(args.runs):
        out = [None] * args.lanes
        threads = [threading.Thread(target=post_once, args=(url, data, name, out, i)) for i in range(args.lanes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for r in out:
            if r:
                walls.append(r["wall"])
                infers.append(r["infer"])
    print("%-16s lanes=%d n=%d  wall p50=%5.0f p95=%5.0f max=%5.0f | infer p50=%5.0f p95=%5.0f"
          % (args.label or "run", args.lanes, len(walls), pct(walls, 50), pct(walls, 95), max(walls),
             pct(infers, 50), pct(infers, 95)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
