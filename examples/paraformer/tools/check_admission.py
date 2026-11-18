#!/usr/bin/env python3
"""Check API admission with --queue-mib 1 --deadline-ms 20 --engines 1 --no-partials."""

import argparse
import asyncio
import base64
from contextlib import AsyncExitStack
import json
import struct
import time

import websockets
from websockets.exceptions import ConnectionClosed

from smoke_production import http_json, post_multipart, post_raw, read_wav


async def send(ws, event, **fields):
    await ws.send(json.dumps({"type": event, **fields}))


async def receive(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=5))


async def append(ws, pcm):
    for i in range(0, len(pcm), 16000):
        await send(ws, "input_audio_buffer.append", audio=base64.b64encode(pcm[i:i + 16000]).decode())


def metrics(http):
    # /metrics is the Prometheus exposition text; /metrics.json is the same
    # snapshot in JSON. These assertions want fields, so they read the JSON twin.
    return json.loads(http_json(http + '/metrics.json')[2])


def check_deadline(error):
    assert error["code"] == "deadline_exceeded", error


async def check_ws(url, http, pcm):
    async with AsyncExitStack() as stack:
        clients = []
        for _ in range(4):
            ws = await stack.enter_async_context(websockets.connect(url))
            assert (await receive(ws))["type"] == "session.created"
            clients.append(ws)
        assert metrics(http)["ws_sessions"] == 4
        async with websockets.connect(url) as excess:
            try:
                await receive(excess)
                raise AssertionError("unexpected fifth session")
            except ConnectionClosed as e:
                assert e.rcvd and e.rcvd.code == 1008, e

        ws = clients[0]
        await append(ws, bytes(20 * 16000 * 2))
        for _ in range(2):
            await send(ws, "input_audio_buffer.commit")
            reply = await receive(ws)
            assert reply["error"]["type"] == "rate_limit_error", reply
        await send(ws, "input_audio_buffer.clear")
        assert (await receive(ws))["type"] == "input_audio_buffer.cleared"

        await append(ws, pcm)
        now = time.monotonic()
        await send(ws, "input_audio_buffer.commit")
        ack = await receive(ws)
        assert ack["type"] == "input_audio_buffer.committed" and ack["item_id"] == "item_1", ack
        error = await receive(ws)
        assert error["type"] == "error" and error["item_id"] == ack["item_id"], error
        check_deadline(error["error"])
        print(f"WS deadline in {(time.monotonic() - now) * 1000:.1f} ms")

        until = time.monotonic() + 8
        while metrics(http)["inflight"]:
            assert time.monotonic() < until, "worker failed to drain"
            await asyncio.sleep(0.02)
        try:
            extra = await asyncio.wait_for(ws.recv(), timeout=0.15)
            raise AssertionError(f"unexpected late WS result: {extra}")
        except asyncio.TimeoutError:
            pass
        assert metrics(http)["queue_bytes"] == 0

    async with websockets.connect(url) as reopened:
        assert (await receive(reopened))["type"] == "session.created"


async def check_disconnect(url, http, pcm):
    before = metrics(http)
    async with websockets.connect(url) as active:
        assert (await receive(active))["type"] == "session.created"
        await append(active, pcm * 8)
        await send(active, "input_audio_buffer.commit")
        active_ack = await receive(active)
        assert active_ack["type"] == "input_audio_buffer.committed", active_ack
        until = time.monotonic() + 2
        while metrics(http)["inflight"] != 1:
            assert time.monotonic() < until, "long request did not reach the sole worker"
            await asyncio.sleep(0.01)
        async with websockets.connect(url) as queued:
            assert (await receive(queued))["type"] == "session.created"
            for _ in range(4):
                await append(queued, pcm)
                await send(queued, "input_audio_buffer.commit")
                ack = await receive(queued)
                assert ack["type"] == "input_audio_buffer.committed", ack
            await append(queued, pcm)
            await send(queued, "input_audio_buffer.commit")
            rejected = await receive(queued)
            assert rejected["error"]["type"] == "rate_limit_error", rejected
            snapshot = metrics(http)
            assert snapshot["inflight"] == 1 and snapshot["queue"] == 4 and snapshot["queue_bytes"] > 0, snapshot
        until = time.monotonic() + 2
        while True:
            snapshot = metrics(http)
            if snapshot["cancelled_queued"] - before["cancelled_queued"] == 4:
                assert snapshot["inflight"] == 1 and snapshot["queue_bytes"] == 0, snapshot
                break
            assert time.monotonic() < until, snapshot
            await asyncio.sleep(0.01)
        final = await receive(active)
        assert final["type"] == "conversation.item.input_audio_transcription.completed", final
        assert final["item_id"] == active_ack["item_id"], final
        print(json.dumps(snapshot, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--disconnect-only", action="store_true", help="use default deadlines with one engine")
    args = parser.parse_args()
    http = f"http://127.0.0.1:{args.port}"
    pcm, rate = read_wav(args.wav)
    assert rate == 16000
    ws = f"ws://127.0.0.1:{args.port}/v1/realtime"
    if args.disconnect_only:
        assert len(pcm) <= 30 * rate * 2 // 8, "choose a command of at most 3.75 seconds"
        asyncio.run(check_disconnect(ws, http, pcm))
        return
    samples = [x / 32768.0 for x, in struct.iter_unpack("<h", pcm)]
    encoded = struct.pack(f"<{len(samples)}f", *samples)
    before = metrics(http)
    assert before["queue_bytes_limit"] == 1024 * 1024, before
    status, _, body = post_raw(http + "/asr/segment", bytes(20 * rate * 4))
    assert status == 429 and json.loads(body)["error"]["type"] == "rate_limit_error", (status, body)
    for post in (
        lambda: post_raw(http + "/asr/segment", encoded),
        lambda: post_multipart(http + "/v1/audio/transcriptions", args.wav, {}),
    ):
        now = time.monotonic()
        status, _, body = post()
        assert status == 504, (status, body)
        check_deadline(json.loads(body)["error"])
        print(f"HTTP deadline in {(time.monotonic() - now) * 1000:.1f} ms")
        until = time.monotonic() + 8
        while metrics(http)["inflight"]:
            assert time.monotonic() < until, "worker failed to drain"
            time.sleep(0.02)
    asyncio.run(check_ws(ws, http, pcm))
    after = metrics(http)
    assert after["deadline_expired"] - before["deadline_expired"] == 3, after
    assert after["rejected"] - before["rejected"] == 4, after
    print(json.dumps(after, sort_keys=True))


if __name__ == "__main__":
    main()
