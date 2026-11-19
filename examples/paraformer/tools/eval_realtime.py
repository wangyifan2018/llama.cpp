#!/usr/bin/env python3
"""Replay manifest audio through the Realtime API and keep item-level outcomes."""

import argparse
import asyncio
import base64
import json
from pathlib import Path

import websockets

from bench_ws_realtime import pct, read_wav


async def transcribe(url, pcm, server_vad, paced, start_at):
    loop = asyncio.get_running_loop()
    async with websockets.connect(url) as ws:
        assert json.loads(await asyncio.wait_for(ws.recv(), timeout=5))["type"] == "session.created"

        async def send(kind, **fields):
            await ws.send(json.dumps({"type": kind, **fields}))

        if server_vad:
            await send("session.update", session={"turn_detection": {"type": "server_vad", "silence_duration_ms": 300}})
            updated = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            if updated["type"] != "session.updated":
                raise RuntimeError(updated)

        seen = {}
        ready = asyncio.Event()

        async def collect():
            async for message in ws:
                event = json.loads(message)
                kind = event["type"]
                if kind == "error":
                    raise RuntimeError(event)
                if kind == "input_audio_buffer.committed":
                    item = event["item_id"]
                    if item in seen:
                        raise RuntimeError(f"duplicate acknowledgement for {item}")
                    seen[item] = {}
                elif kind == "conversation.item.input_audio_transcription.completed":
                    row = seen[event["item_id"]]
                    if "transcript" in row:
                        raise RuntimeError(f"duplicate final for {event['item_id']}")
                    row.update(transcript=event["transcript"], received_ms=loop.time() * 1000)
                elif kind == "x.pf.metrics":
                    seen[event["item_id"]]["metrics"] = event
                    ready.set()

        collector = asyncio.create_task(collect())
        try:
            audio = pcm + bytes(32000) if server_vad else pcm
            await asyncio.sleep(max(0, start_at - loop.time()))
            start = loop.time()
            endpoint = start + len(pcm) / 32000 if paced else None
            for off in range(0, len(audio), 640):
                chunk = audio[off:off + 640]
                await send("input_audio_buffer.append", audio=base64.b64encode(chunk).decode("ascii"))
                if paced:
                    await asyncio.sleep(max(0, start + (off + len(chunk)) / 32000 - loop.time()))
            if not server_vad:
                endpoint = loop.time()
                await send("input_audio_buffer.commit")
            waiter = asyncio.create_task(ready.wait())
            try:
                done, _ = await asyncio.wait((waiter, collector), timeout=10, return_when=asyncio.FIRST_COMPLETED)
                if collector in done:
                    collector.result()
                    raise RuntimeError("socket closed before a final")
                if waiter not in done:
                    raise TimeoutError("no final metrics")
            finally:
                waiter.cancel()
            if server_vad:
                # Capture additional auto-segments from the same clip.
                done, _ = await asyncio.wait((collector,), timeout=0.75)
                if done:
                    collector.result()
            if not seen or any("metrics" not in row or "transcript" not in row for row in seen.values()):
                raise RuntimeError(f"missing terminal from items {seen}")
            elapsed = (max(row["received_ms"] for row in seen.values()) - endpoint * 1000) if endpoint is not None else None
            return seen, elapsed
        finally:
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)


async def replay(args):
    samples, failures = [], 0
    root = args.manifest.parent
    entries = [line.split("\t", 2) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    if not entries:
        raise SystemExit("manifest is empty")

    async def replay_clip(entry, start_at):
        name, language, reference = entry
        wav = root / name
        row = {"wav": str(wav), "reference": reference}
        try:
            pcm = read_wav(str(wav))
            items, elapsed = await transcribe(args.url, pcm, args.server_vad, not args.unpaced, start_at)
            separator = "，" if language == "zh" else ", "
            text = separator.join(item["transcript"] for item in items.values() if item["transcript"])
            row.update(text=text, items=items, endpoint_to_final_ms=elapsed, audio_ms=len(pcm) / 32)
        except (RuntimeError, TimeoutError, OSError, KeyError, websockets.exceptions.WebSocketException) as error:
            row["error"] = str(error) or type(error).__name__
        return row

    with args.out.open("w", encoding="utf-8") as tsv, args.details.open("w", encoding="utf-8") as details:
        for offset in range(0, len(entries), args.lanes):
            start = asyncio.get_running_loop().time() + (0.3 if args.lanes > 1 else 0)
            group = entries[offset:offset + args.lanes]
            rows = await asyncio.gather(*(replay_clip(entry, start + i * args.stagger_ms / 1000)
                                          for i, entry in enumerate(group)))
            for row in rows:
                if "error" in row:
                    failures += 1
                else:
                    elapsed = row["endpoint_to_final_ms"]
                    timing = ""
                    if elapsed is not None:
                        samples.append(elapsed)
                        timing = f"\twall_ms={elapsed:.3f}"
                    tsv.write(f'{row["wav"]}\taudio_ms={row["audio_ms"]:.3f}{timing}\t{row["text"]}\n')
                print(json.dumps(row, ensure_ascii=False), file=details, flush=True)
    if samples:
        print(f"n={len(samples)} p50={pct(samples, 50):.0f} p95={pct(samples, 95):.0f} p99={pct(samples, 99):.0f} ms")
    if failures:
        raise SystemExit(f"{failures} replay(s) failed, see {args.details}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--server-vad", action="store_true")
    parser.add_argument("--unpaced", action="store_true", help="reduce runtime for accuracy-only checks")
    parser.add_argument("--lanes", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--stagger-ms", type=int, default=0, help="start offset between adjacent lanes")
    args = parser.parse_args()
    if args.stagger_ms < 0:
        parser.error("--stagger-ms must be nonnegative")
    asyncio.run(replay(args))
