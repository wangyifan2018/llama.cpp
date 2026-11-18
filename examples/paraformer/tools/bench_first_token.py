#!/usr/bin/env python3
"""Measure what a driver actually waits for on /v1/realtime.

The other WS bench reports endpoint-to-final, which is the wrong number for a
cabin: nobody watches for the final, they watch for text to appear. This drives
the socket in real time and reports two separate waits, both in milliseconds
measured from the instant the last audio chunk was committed:

  already_visible   text was on screen before the speaker stopped talking. This
                    is the normal partials-on case and the good one.
  post_commit_wait  from stopping talking to the first *new* text. Zero when
                    nothing new was needed; otherwise it is what the driver
                    sits through looking at an unchanged screen.
  to_final          from stopping talking to the completed transcript.

Keeping the two apart is the point. A server that never emits partials reports
post_commit_wait == to_final for every session, which is exactly the difference
this exists to expose.

    python3 bench_first_token.py --url ws://127.0.0.1:18884/v1/realtime \
        --wav cmd.wav --sessions 4 --rounds 5
"""

import argparse
import asyncio
import base64
import json
import wave

import websockets


def read_wav(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000, path
        return f.readframes(f.getnframes())


def pct(vals, q):
    v = sorted(vals)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


async def session(url, pcm, out, idx, start_at, chunk_ms=20):
    step = chunk_ms * 32
    loop = asyncio.get_running_loop()
    state = {"commit_at": None, "seen": [], "pre": None, "final": None, "text": ""}
    try:
        async with websockets.connect(url, max_size=None) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)        # session.created

            # A reader task owns the socket for the whole session. Draining
            # inline between sends loses deltas: the pacing sleep runs before
            # the read, so a delta arriving mid-chunk is only noticed a chunk
            # later and one landing on the final chunk is lost entirely -- which
            # turns an on-screen partial into an apparent post-commit wait.
            async def reader():
                while True:
                    try:
                        raw = await ws.recv()
                    except Exception:                             # noqa: BLE001
                        return
                    msg = json.loads(raw)
                    kind = msg.get("type", "")
                    text = (msg.get("delta") or msg.get("text") or msg.get("transcript") or "").strip()
                    now = loop.time()
                    if state["commit_at"] is None:
                        if text:
                            state["seen"].append(text)
                        continue
                    if text and state["pre"] is None:
                        state["pre"] = (now - state["commit_at"]) * 1000.0
                    if kind.endswith("transcription.completed") or kind == "final":
                        state["final"] = (now - state["commit_at"]) * 1000.0
                        state["text"] = text
                        return

            rt = asyncio.create_task(reader())
            await asyncio.sleep(max(0.0, start_at - loop.time()))
            for off in range(0, len(pcm), step):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(pcm[off:off + step]).decode()}))
                target = start_at + min(len(pcm), off + step) / 32000.0
                delay = target - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            # Freeze what is on screen at the instant the speaker stops. Read
            # from "seen" before the commit timestamp is published, so a delta
            # that arrives later cannot be counted as having been visible.
            visible = state["seen"][-1] if state["seen"] else ""
            state["commit_at"] = loop.time()
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            try:
                await asyncio.wait_for(rt, timeout=30)
            except asyncio.TimeoutError:
                rt.cancel()
            out[idx] = {"pre": state["pre"], "final": state["final"], "text": state["text"], "seen": visible}
            return
    except Exception as exc:                                       # noqa: BLE001
        out[idx] = {"error": repr(exc)[:160]}
        return


async def run(args, pcm):
    pres, fins, finals, seen = [], [], [], []
    loop = asyncio.get_running_loop()
    for _ in range(args.rounds):
        out = [None] * args.sessions
        t0 = loop.time() + 0.35
        await asyncio.gather(*[session(args.url, pcm, out, i, t0 + i * args.stagger_ms / 1000.0)
                               for i in range(args.sessions)])
        for r in out:
            if not r:
                continue
            if r.get("error"):
                print("   error:", r["error"], flush=True)
                continue
            if r["final"] is not None:
                fins.append(r["final"])
            if r["pre"] is not None:
                pres.append(r["pre"])
            if r["text"]:
                finals.append(r["text"])
            if r["seen"]:
                seen.append(r["seen"])
    if not fins:
        print("no completed transcript came back; is the server up?", flush=True)
        return 1
    already = len(seen)
    print("%-16s sessions=%d rounds=%d n=%d | text visible at commit: %d/%d | "
          "post-commit wait p50=%5.0f p95=%5.0f | final p50=%5.0f p95=%5.0f"
          % (args.label or "run", args.sessions, args.rounds, len(fins), already, len(fins),
             pct(pres, 50), pct(pres, 95), pct(fins, 50), pct(fins, 95)), flush=True)
    if args.show_text:
        for t in finals[:3]:
            print("   final:", t, flush=True)
        for t in seen[:2]:
            print("   seen :", t, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:18884/v1/realtime")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--stagger-ms", type=int, default=0, help="offset each session's start")
    ap.add_argument("--label", default="")
    ap.add_argument("--show-text", action="store_true")
    args = ap.parse_args()
    return asyncio.run(run(args, read_wav(args.wav)))


if __name__ == "__main__":
    raise SystemExit(main())