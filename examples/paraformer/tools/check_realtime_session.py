#!/usr/bin/env python3
"""Check nested settings and per-commit keyword snapshots on the Realtime API."""

import argparse
import asyncio
import base64
import json
from pathlib import Path

import websockets

from bench_ws_realtime import read_wav


async def send(ws, kind, **fields):
    await ws.send(json.dumps({"type": kind, **fields}))


async def receive(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), 10))


async def update(ws, **input_config):
    await send(ws, "session.update", session={"type": "transcription", "audio": {"input": input_config}})
    return await receive(ws)


def config(event):
    assert event["type"] in ("session.created", "session.updated"), event
    return event["session"]["audio"]["input"]


async def commit(ws, pcm):
    if pcm:
        await send(ws, "input_audio_buffer.append", audio=base64.b64encode(pcm).decode("ascii"))
    await send(ws, "input_audio_buffer.commit")
    event = await receive(ws)
    assert event["type"] == "input_audio_buffer.committed", event
    return event["item_id"]


async def final(ws, item):
    completed = None
    while True:
        event = await receive(ws)
        assert event["type"] != "error", event
        if event["type"] == "conversation.item.input_audio_transcription.completed":
            assert event["item_id"] == item and completed is None, event
            completed = event
        if event["type"] == "x.pf.metrics":
            assert event["item_id"] == item and completed is not None, event
            return {"text": completed["transcript"], "backend": event["backend"], "infer_ms": event["infer_ms"]}


async def check(args):
    pcm = read_wav(str(args.wav))
    async with websockets.connect(args.url) as first, websockets.connect(args.url) as second:
        for ws in first, second:
            initial = config(await receive(ws))
            assert initial["format"] == {"type": "audio/pcm", "rate": 16000}
            assert initial["turn_detection"] is None
            assert initial["transcription"]["keywords"] == []

        changed = config(await update(first, transcription={"language": "cmn", "keywords": [args.keyword, args.keyword]},
                                     turn_detection=None))
        assert changed["transcription"]["language"] == "cmn"
        assert changed["transcription"]["keywords"] == [args.keyword]
        for invalid in (
            {"format": {"type": "audio/pcm", "rate": 24000}},
            {"turn_detection": {"type": "server_vad", "silence_duration_ms": "bad"},
             "transcription": {"keywords": []}},
            {"transcription": {"keywords": [args.keyword] * 25}},
            {"transcription": {"keywords": ["car\u0000name"]}},
        ):
            error = await update(first, **invalid)
            assert error["type"] == "error", error
            assert config(await update(first)) == changed

        await send(second, "input_audio_buffer.append", audio=base64.b64encode(pcm).decode("ascii"))
        await send(second, "session.update", session={"input_audio_transcription": {"language": "zh", "keywords": []},
                                                      "turn_detection": None})
        assert config(await receive(second))["transcription"]["language"] == "zh"
        first_item, second_item = await asyncio.gather(commit(first, pcm), commit(second, None))

        await send(first, "session.update", session={"audio": {"input": {"transcription": {"keywords": []}}}})
        # The worker may deliver the final before processing the config acknowledgement.
        replies = {}
        while set(replies) != {"session.updated", "conversation.item.input_audio_transcription.completed", "x.pf.metrics"}:
            event = await receive(first)
            assert event["type"] != "error" and event["type"] not in replies, event
            replies[event["type"]] = event
        assert config(replies["session.updated"])["transcription"]["keywords"] == []
        themed = replies["conversation.item.input_audio_transcription.completed"]
        assert themed["item_id"] == first_item and args.keyword in themed["transcript"], themed
        assert replies["x.pf.metrics"]["item_id"] == first_item
        plain = await final(second, second_item)
        cleared = await final(first, await commit(first, pcm))
        for result in plain, cleared:
            assert args.unbiased in result["text"] and args.keyword not in result["text"], result
        config(await update(second, transcription={"keywords": [args.keyword, args.unbiased]}))
        ambiguous = await final(second, await commit(second, pcm))
        assert ambiguous["text"] == plain["text"], ambiguous
        print(json.dumps({"keyword": themed["transcript"], "plain": plain, "cleared": cleared,
                          "both_candidates": ambiguous, "keyword_backend": replies["x.pf.metrics"]["backend"]},
                         ensure_ascii=False))

        if args.expect_vad:
            vad = {"type": "server_vad", "silence_duration_ms": 300}
            active = config(await update(first, turn_detection=vad, transcription={"keywords": [args.keyword]}))
            assert active["turn_detection"]["silence_duration_ms"] == 300
            assert active["transcription"]["keywords"] == [args.keyword]
            cut = len(pcm) // 4 * 2
            await send(first, "input_audio_buffer.append", audio=base64.b64encode(pcm[:cut]).decode("ascii"))
            assert config(await update(first, turn_detection=vad)) == active
            await send(first, "input_audio_buffer.append", audio=base64.b64encode(pcm[cut:] + bytes(32000)).decode("ascii"))
            ack = await receive(first)
            assert ack["type"] == "input_audio_buffer.committed", ack
            automatic = await final(first, ack["item_id"])
            assert args.keyword in automatic["text"], automatic
            print(json.dumps({"automatic": automatic}, ensure_ascii=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--url", required=True)
    cli.add_argument("--wav", type=Path, required=True)
    cli.add_argument("--keyword", default="王梓涵")
    cli.add_argument("--unbiased", default="王子涵")
    cli.add_argument("--expect-vad", action="store_true")
    asyncio.run(check(cli.parse_args()))
