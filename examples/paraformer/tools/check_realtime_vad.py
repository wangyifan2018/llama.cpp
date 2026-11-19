#!/usr/bin/env python3
"""Exercise the shared item lifecycle for automatic and explicit endpoints."""

import argparse
import asyncio
import math

import websockets

from check_admission import append, receive, send
from smoke_production import read_wav


async def update(ws, turn_detection):
    await send(ws, "session.update", session={"turn_detection": turn_detection})
    return await receive(ws)


async def check_item(ws, number, silence_ms, deadline):
    item = f"item_{number}"
    ack = await receive(ws)
    assert ack["type"] == "input_audio_buffer.committed" and ack["item_id"] == item, ack
    assert ack["previous_item_id"] == (f"item_{number - 1}" if number > 1 else None), ack
    terminal = await receive(ws)
    if terminal["type"] == "x.pf.partial":
        assert terminal["item_id"] == item, terminal
        terminal = await receive(ws)
    assert terminal["item_id"] == item, terminal
    if deadline:
        assert terminal["type"] == "error" and terminal["error"]["code"] == "deadline_exceeded", terminal
    else:
        assert terminal["type"].endswith(".completed") and terminal["transcript"], terminal
        metrics = await receive(ws)
        assert metrics["type"] == "x.pf.metrics" and metrics["item_id"] == item, metrics
        assert math.isclose(metrics["tail_ms"] - metrics["queue_ms"] - metrics["infer_ms"], silence_ms, abs_tol=0.1), metrics
    print(f"{item}: {terminal['type']}, silence_ms={silence_ms}")


async def replay(port, pcm, deadline):
    vad = {"type": "server_vad", "silence_duration_ms": 550}
    async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
        assert (await receive(ws))["type"] == "session.created"
        assert (await update(ws, {**vad, "silence_duration_ms": "550"}))["type"] == "error"
        assert (await update(ws, vad))["type"] == "session.updated"
        await append(ws, pcm + bytes(32000))
        await check_item(ws, 1, 550, deadline)

        await append(ws, pcm)
        await send(ws, "input_audio_buffer.commit")
        await check_item(ws, 2, 0, deadline)

        await append(ws, pcm)
        await send(ws, "input_audio_buffer.clear")
        assert (await receive(ws))["type"] == "input_audio_buffer.cleared"
        await send(ws, "input_audio_buffer.commit")
        empty = await receive(ws)
        assert empty["type"] == "error" and "item_id" not in empty, empty

        assert (await update(ws, None))["type"] == "session.updated"
        await append(ws, pcm)
        await send(ws, "input_audio_buffer.commit")
        await check_item(ws, 3, 0, deadline)
        try:
            raise AssertionError(await asyncio.wait_for(ws.recv(), timeout=0.25))
        except asyncio.TimeoutError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--expect-deadline", action="store_true", help="run against a short deadline, such as 20 ms")
    args = parser.parse_args()
    pcm, rate = read_wav(args.wav)
    assert rate == 16000
    asyncio.run(replay(args.port, pcm, args.expect_deadline))


if __name__ == "__main__":
    main()
