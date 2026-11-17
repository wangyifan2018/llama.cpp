#!/usr/bin/env python3
"""Replay one finalized long segment and three late short segments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import struct
import time

from asr_client import Client, OP_APPEND, OP_CLOSE, OP_COMMIT, OP_OPEN, ST_FINAL, ST_OK, read_wav


def transcribe(port, start, offset_ms, payload):
    time.sleep(max(0.0, start + offset_ms / 1000.0 - time.monotonic()))
    endpoint = time.monotonic()
    client = Client("127.0.0.1", port)
    try:
        status, body = client.call(OP_OPEN)
        if status != ST_OK:
            raise RuntimeError(f"OPEN failed: {body!r}")
        sid, = struct.unpack("<I", body)
        client.send(OP_APPEND, sid, payload)
        status, body = client.call(OP_COMMIT, sid)
        elapsed = 1000.0 * (time.monotonic() - endpoint)
        if status != ST_FINAL or client.call(OP_CLOSE, sid)[0] != ST_OK:
            raise RuntimeError(f"COMMIT or CLOSE failed: {body!r}")
        return {"offset_ms": offset_ms, "server_session": sid,
                "endpoint_to_final_ms": elapsed, "text": body.decode()}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument("long_wav")
    parser.add_argument("short_wav")
    parser.add_argument("--reps", type=int, default=8)
    args = parser.parse_args()
    if args.reps <= 0:
        parser.error("--reps must be positive")
    long_pcm, short_pcm = [struct.pack(f"<{len(samples)}f", *samples)
                           for samples in map(read_wav, (args.long_wav, args.short_wav))]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for rep in range(args.reps):
            start = time.monotonic() + 0.05
            jobs = [pool.submit(transcribe, args.port, start, offset, payload)
                    for offset, payload in ((0, long_pcm), (550, short_pcm), (600, short_pcm), (650, short_pcm))]
            for job in jobs:
                print(json.dumps({"rep": rep, **job.result()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
