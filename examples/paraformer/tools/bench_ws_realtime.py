#!/usr/bin/env python3
"""Realtime concurrency bench for /v1/realtime.

Opens N WebSocket sessions that append 20 ms chunks in real time and commit at
the end, then reports per-session endpoint-to-final plus the server counters from
/metrics.json. Used to check that partial decodes never sit in front of a commit.

    python3 bench_ws_realtime.py --url ws://127.0.0.1:8899/v1/realtime --wav a.wav --sessions 4 --rounds 3
"""

import argparse
import asyncio
import base64
import json
from pathlib import Path
import statistics
import wave

import websockets


def read_wav(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000, path
        return f.readframes(f.getnframes())


async def session(url, pcm, results, idx, start_at, chunk_ms=20, keywords=None):
    step = chunk_ms * 32
    async with websockets.connect(url, max_size=None) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)  # session.created
        if keywords:
            await ws.send(json.dumps({"type": "session.update", "session": {"audio": {"input": {
                "transcription": {"keywords": keywords}}}}}))
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            if event.get("type") != "session.updated" or event["session"]["audio"]["input"]["transcription"]["keywords"] != keywords:
                raise RuntimeError(f"server did not apply requested keywords: {event}")
        await asyncio.sleep(max(0.0, start_at - asyncio.get_event_loop().time()))
        for off in range(0, len(pcm), step):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[off:off + step]).decode()}))
            target = start_at + min(len(pcm), off + step) / 32000.0
            delay = target - asyncio.get_event_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)
        endpoint = asyncio.get_event_loop().time()
        scheduled_endpoint = start_at + len(pcm) / 32000.0
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        deadline = endpoint + 30
        completed = None
        while (remaining := deadline - asyncio.get_event_loop().time()) > 0:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            if msg.get("type") == "conversation.item.input_audio_transcription.completed":
                now = asyncio.get_event_loop().time()
                completed = {"item_id": msg["item_id"], "e2e_ms": (now - endpoint) * 1000.0,
                             "scheduled_e2e_ms": (now - scheduled_endpoint) * 1000.0,
                             "commit_lag_ms": (endpoint - scheduled_endpoint) * 1000.0,
                             "text": msg["transcript"], "speaker_status": msg.get("speaker_status"),
                             "speaker_reason": msg.get("speaker_reason"), "speaker_score": msg.get("speaker_score")}
            if completed and msg.get("type") == "x.pf.metrics" and msg.get("item_id") == completed["item_id"]:
                results[idx] = {**completed, **{key: msg.get(key) for key in ("queue_ms", "infer_ms", "vad_ms", "backend")}}
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
    ap.add_argument("--metrics", default="http://127.0.0.1:8899/metrics.json")
    audio = ap.add_mutually_exclusive_group(required=True)
    audio.add_argument("--wav", help="replay the same WAV on all lanes")
    audio.add_argument("--wavs", nargs="+", help="provide one WAV per lane")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--stagger-ms", type=int, default=0)
    ap.add_argument("--start-offsets-ms", nargs="+", type=int)
    ap.add_argument("--keyword", action="append", help="Mandarin keyword scoped to each benchmark session")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    paths = args.wavs if args.wavs else [args.wav] * args.sessions
    offsets = args.start_offsets_ms if args.start_offsets_ms else [i * args.stagger_ms for i in range(args.sessions)]
    if args.sessions <= 0 or args.rounds <= 0 or len(paths) != args.sessions or len(offsets) != args.sessions or min(offsets) < 0:
        ap.error("supply positive counts and one WAV and nonnegative start offset per lane")
    clips = [read_wav(path) for path in paths]
    durations = [len(pcm) / 32000.0 for pcm in clips]
    print(f"audio seconds={durations}, start offsets ms={offsets}, rounds={args.rounds}")
    all_ms = []
    rows = []
    for r in range(args.rounds):
        results = [None] * args.sessions
        start = asyncio.get_event_loop().time() + 0.3
        tasks = [session(args.url, pcm, results, i, start + offsets[i] / 1000.0, keywords=args.keyword)
                 for i, pcm in enumerate(clips)]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for i, outcome in enumerate(outcomes):
            if isinstance(outcome, Exception):
                results[i] = {"error": str(outcome) or type(outcome).__name__}
        errs = [x for x in results if x is None or "error" in x]
        rows.extend({"round": r, "lane": i, "wav": paths[i], "start_offset_ms": offsets[i],
                     "audio_ms": durations[i] * 1000, **(x or {"error": "missing reply"})} for i, x in enumerate(results))
        vals = [x["e2e_ms"] for x in results if x and "e2e_ms" in x]
        all_ms += vals
        texts = {(paths[i], x["text"]) for i, x in enumerate(results) if x and "text" in x}
        consistent = len(texts) == len({wav for wav, _ in texts})
        print("round %d: n=%d errors=%d p50=%.0f max=%.0f consistent=%s"
              % (r, len(vals), len(errs), statistics.median(vals) if vals else -1,
                 max(vals) if vals else -1, consistent))
    if all_ms:
        print("overall n=%d p50=%.0f p95=%.0f p99=%.0f max=%.0f"
              % (len(all_ms), pct(all_ms, 50), pct(all_ms, 95), pct(all_ms, 99), max(all_ms)))
    if args.out:
        with args.out.open("w", encoding="utf-8") as out:
            for row in rows:
                print(json.dumps(row, ensure_ascii=False), file=out)
    try:
        import urllib.request
        print("server metrics:", urllib.request.urlopen(args.metrics, timeout=5).read().decode())
    except Exception as e:
        print("metrics unavailable:", e)
    if any("error" in row for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
