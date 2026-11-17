#!/usr/bin/env python3
"""Realtime concurrency bench for /v1/realtime.

Opens N WebSocket sessions that append 20 ms chunks in real time and commit at
the end, then reports per-session endpoint-to-final plus the server counters from
/metrics. Used to check that partial decodes never sit in front of a commit.

    python3 bench_ws_realtime.py --url ws://127.0.0.1:8899/v1/realtime --wav a.wav --sessions 4 --rounds 3
"""

import argparse
import asyncio
import base64
import json
import statistics
import wave

import websockets


def read_wav(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000, path
        return f.readframes(f.getnframes())


async def session(url, pcm, results, idx, start_at, chunk_ms=20):
    step = chunk_ms * 32
    await asyncio.sleep(max(0.0, start_at - asyncio.get_event_loop().time()))
    async with websockets.connect(url, max_size=None) as ws:
        await ws.recv()  # session.created
        t0 = asyncio.get_event_loop().time()
        for off in range(0, len(pcm), step):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[off:off + step]).decode()}))
            target = t0 + (off + len(pcm[off:off + step])) / 32000.0
            delay = target - asyncio.get_event_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)
        endpoint = asyncio.get_event_loop().time()
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        deadline = endpoint + 30
        while asyncio.get_event_loop().time() < deadline:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if msg.get("type") == "conversation.item.input_audio_transcription.completed":
                results[idx] = {"e2e_ms": (asyncio.get_event_loop().time() - endpoint) * 1000.0,
                                "text": msg["transcript"]}
                return
            if msg.get("type") == "error":
                results[idx] = {"error": msg}
                return
        results[idx] = {"error": "timeout"}


def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8899/v1/realtime")
    ap.add_argument("--metrics", default="http://127.0.0.1:8899/metrics")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--stagger-ms", type=int, default=0)
    args = ap.parse_args()

    pcm = read_wav(args.wav)
    print("audio %.2f s, %d sessions, %d rounds, stagger %d ms"
          % (len(pcm) / 32000.0, args.sessions, args.rounds, args.stagger_ms))
    all_ms = []
    for r in range(args.rounds):
        results = [None] * args.sessions
        start = asyncio.get_event_loop().time() + 0.1
        tasks = [session(args.url, pcm, results, i, start + i * args.stagger_ms / 1000.0)
                 for i in range(args.sessions)]
        await asyncio.gather(*tasks)
        errs = [x for x in results if x is None or "error" in x]
        vals = [x["e2e_ms"] for x in results if x and "e2e_ms" in x]
        all_ms += vals
        texts = sorted(set(x["text"] for x in results if x and "text" in x))
        print("round %d: n=%d errors=%d p50=%.0f max=%.0f consistent=%s"
              % (r, len(vals), len(errs), statistics.median(vals) if vals else -1,
                 max(vals) if vals else -1, len(texts) == 1))
    if all_ms:
        print("overall n=%d p50=%.0f p95=%.0f p99=%.0f max=%.0f"
              % (len(all_ms), pct(all_ms, 50), pct(all_ms, 95), pct(all_ms, 99), max(all_ms)))
    try:
        import urllib.request
        print("server metrics:", urllib.request.urlopen(args.metrics, timeout=5).read().decode())
    except Exception as e:
        print("metrics unavailable:", e)


if __name__ == "__main__":
    asyncio.run(main())
