#!/usr/bin/env python3
"""Realtime client for the Paraformer WebSocket endpoint.

Sends one VAD segment the way a cabin front end does: 20 ms PCM16 appends while
the user speaks, then a commit at the endpoint. Prints the partial deltas and the
final transcript, plus the server-side queue and inference time.

    python3 ws_client.py --url ws://127.0.0.1:8080/v1/realtime --wav a.wav --paced
"""

import argparse
import asyncio
import base64
import json
import time
import wave

import websockets


def read_wav(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2, path
        assert f.getframerate() == 16000, "expected 16 kHz, got %d" % f.getframerate()
        return f.readframes(f.getnframes())


async def run(args):
    pcm = read_wav(args.wav)
    step = args.chunk_ms * 32  # bytes per 20 ms at 16 kHz mono 16-bit
    deltas = []
    async with websockets.connect(args.url, max_size=None) as ws:
        created = json.loads(await ws.recv())
        print("event:", created["type"])
        await ws.send(json.dumps({"type": "session.update",
                                  "session": {"input_audio_transcription": {"language": args.language}}}))
        print("event:", json.loads(await ws.recv())["type"])

        t0 = time.monotonic()
        for off in range(0, len(pcm), step):
            chunk = pcm[off:off + step]
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(chunk).decode()}))
            if args.paced:
                target = t0 + (off + len(chunk)) / 32000.0
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            # drain whatever the server pushed while we were appending
            while True:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.001))
                except asyncio.TimeoutError:
                    break
                kind = msg.get("type")
                if kind == "conversation.item.input_audio_transcription.delta":
                    deltas.append(msg["delta"])
                    print("delta:", repr(msg["delta"]))
                elif kind not in ("session.created", "session.updated"):
                    print("event:", kind)
        endpoint = time.monotonic()
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        final = None
        metrics = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and final is None:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            kind = msg.get("type")
            if kind == "conversation.item.input_audio_transcription.completed":
                final = msg["transcript"]
                print("completed:", final)
                print("endpoint_to_final_ms: %.0f" % ((time.monotonic() - endpoint) * 1000))
            elif kind == "x.pf.metrics":
                metrics = msg
                print("server: queue_ms=%.0f infer_ms=%.0f" % (msg["queue_ms"], msg["infer_ms"]))
            else:
                print("event:", kind, msg.get("error", ""))
        if final is None:
            raise SystemExit("no completed event")
        # The server sends its timing after the transcript; do not lose it.
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
            if msg.get("type") == "x.pf.metrics":
                print("server: queue_ms=%.0f infer_ms=%.0f" % (msg["queue_ms"], msg["infer_ms"]))
        except asyncio.TimeoutError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8080/v1/realtime")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--paced", action="store_true")
    ap.add_argument("--chunk-ms", type=int, default=20)
    ap.add_argument("--language", default="zh")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
