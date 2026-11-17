#!/usr/bin/env python3
"""Benchmark llama-paraformer-server: endpoint-to-final latency under concurrency.

Paraformer is whole-utterance: one VAD segment goes out as a single RECOGNIZE
frame, so request-sent -> FINAL-received is the endpoint-to-final latency the
product cares about (it includes socket, server queue and inference).

The server must already be running. This tool:

  * can emit a float32 warmup file from a WAV (--emit-warmup PATH --wav WAV),
    matching the server's --warmup F32 input;
  * drives N simultaneous connections with one utterance each, repeated, and
    reports P50/P95/P99/P99, throughput and transcript consistency;
  * writes per-request JSONL and a summary.

Wire framing matches the server: request <III op,reserved,nbytes>+payload;
reply <II status,nbytes>+payload.
"""

import argparse
import json
import os
import socket
import statistics
import struct
import sys
import threading
import time
import wave

OP_RECOGNIZE = 0
ST_FINAL, ST_ERROR = 1, 2


def read_wav(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000, path
        data = f.readframes(f.getnframes())
    vals = struct.unpack("<%dh" % (len(data) // 2), data)
    return [v / 32768.0 for v in vals]


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("server closed connection")
        buf += chunk
    return buf


def one_request(host, port, samples):
    payload = struct.pack("<%df" % len(samples), *samples)
    t0 = time.monotonic()
    sock = socket.create_connection((host, port), timeout=60.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.sendall(struct.pack("<III", OP_RECOGNIZE, 0, len(payload)) + payload)
        head = recv_exact(sock, 8)
        status, size = struct.unpack("<II", head)
        body = recv_exact(sock, size) if size else b""
    finally:
        sock.close()
    dt = (time.monotonic() - t0) * 1000.0
    return status, body.decode("utf-8", "replace"), dt


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8876)
    ap.add_argument("--wavs", required=True, help="comma-separated WAV paths (one per lane; repeats if fewer than lanes)")
    ap.add_argument("--lanes", default="1,2,4", help="comma-separated concurrency levels")
    ap.add_argument("--reps", type=int, default=8, help="repetitions per lane (rounds)")
    ap.add_argument("--stagger-ms", type=float, default=0.0, help="delay between lane starts within a round")
    ap.add_argument("--warm-first", action="store_true", help="send one throwaway request before measuring")
    ap.add_argument("--out", default=None, help="JSONL output path")
    ap.add_argument("--emit-warmup", default=None, help="write WAV[0] as float32 for server --warmup, then exit")
    args = ap.parse_args()

    wavs = [w for w in args.wavs.split(",") if w]
    samples_by_wav = [read_wav(w) for w in wavs]
    if args.emit_warmup:
        with open(args.emit_warmup, "wb") as f:
            f.write(struct.pack("<%df" % len(samples_by_wav[0]), *samples_by_wav[0]))
        print("wrote", args.emit_warmup, len(samples_by_wav[0]), "float32 samples")
        return

    if args.warm_first:
        one_request(args.host, args.port, samples_by_wav[0])

    rows = []
    for lanes in [int(x) for x in args.lanes.split(",")]:
        round_lat = []
        for rep in range(args.reps):
            results = [None] * lanes
            threads = []

            def run(i):
                smp = samples_by_wav[i % len(samples_by_wav)]
                if args.stagger_ms:
                    time.sleep(args.stagger_ms * i / 1000.0)
                try:
                    status, text, dt = one_request(args.host, args.port, smp)
                    results[i] = {"lane": i, "wav": wavs[i % len(wavs)], "status": status,
                                  "text": text, "latency_ms": round(dt, 3)}
                except Exception as e:
                    results[i] = {"lane": i, "wav": wavs[i % len(wavs)], "error": str(e)}

            t_start = time.monotonic()
            for i in range(lanes):
                t = threading.Thread(target=run, args=(i,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            round_wall = (time.monotonic() - t_start) * 1000.0
            good = [r for r in results if r and r.get("status") == ST_FINAL]
            for r in results:
                row = dict(r or {}, lanes=lanes, rep=rep, round_wall_ms=round(round_wall, 3))
                rows.append(row)
            lat = [r["latency_ms"] for r in good]
            round_lat.extend(lat)
            # per-round transcript consistency (same wav -> same text)
            texts = [(r["wav"], r["text"]) for r in good]
            consistent = len(set(texts)) == len(set(w for w, _ in texts))
            if args.reps <= 3 or rep == args.reps - 1:
                print("lanes=%d rep=%d ok=%d/%d wall=%.1fms consistent=%s"
                      % (lanes, rep, len(good), lanes, round_wall, consistent))
        ok_all = [r["latency_ms"] for r in rows if r.get("lanes") == lanes and r.get("status") == ST_FINAL]
        errs = [r for r in rows if r.get("lanes") == lanes and r.get("error")]
        print("lanes=%d N=%d errs=%d  p50=%.1f p95=%.1f p99=%.1f max=%.1f ms"
              % (lanes, len(ok_all), len(errs), pct(ok_all, 50), pct(ok_all, 95),
                 pct(ok_all, 99), max(ok_all) if ok_all else float("nan")))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote", args.out, len(rows), "rows")


if __name__ == "__main__":
    main()
