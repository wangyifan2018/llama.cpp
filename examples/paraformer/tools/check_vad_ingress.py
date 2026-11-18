#!/usr/bin/env python3
"""Compare identical VAD inputs through WAV, raw PCM, and WebSocket."""

import argparse
import asyncio
import base64
import json
from pathlib import Path
import struct
import wave

from smoke_production import post_multipart, post_raw, read_wav


def write_wav(path, audio):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(audio)


def fixture(path, seconds, clips):
    pcm = bytearray(32000 * seconds)
    for start_ms, clip in clips:
        start = start_ms * 32
        assert start + len(clip) <= len(pcm)
        pcm[start:start + len(clip)] = clip
    write_wav(path, pcm)
    return pcm


async def ws_transcribe(port, pcm):
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        reply, metrics = None, None
        while reply is None or metrics is None:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert msg["type"] != "error", msg
            if msg["type"].endswith(".completed"):
                reply = msg
            elif msg["type"] == "x.pf.metrics":
                metrics = msg
        return reply, metrics


def compare(port, wav, pcm, expected):
    base = f"http://127.0.0.1:{port}"
    api = base + "/v1/audio/transcriptions"
    f32 = b"".join(struct.pack("<f", s / 32768.0) for (s,) in struct.iter_unpack("<h", pcm))
    status, wav_headers, body = post_multipart(api, str(wav), {"response_format": "verbose_json"})
    assert status == 200, body
    result = json.loads(body)
    status, raw_headers, body = post_raw(base + "/asr/segment", f32)
    assert status == 200, body
    raw = json.loads(body)
    ws, metrics = asyncio.run(ws_transcribe(port, pcm))
    assert result["text"] == raw["text"] == ws["transcript"], (result["text"], raw, ws)
    count = int(wav_headers["X-Vad-Pieces"])
    assert count == int(raw_headers["X-Vad-Pieces"]) == ws["vad_pieces"], (wav_headers, raw_headers, ws)
    if expected == 0:
        assert count == 0 and result["text"] == "" and result["words"] == result["segments"] == []
        assert wav_headers["X-Empty"] == raw_headers["X-Empty"] == ws["empty"] == "silence"
        for fmt in ("text", "srt"):
            status, headers, body = post_multipart(api, str(wav), {"response_format": fmt})
            assert status == 200 and headers["X-Empty"] == "silence" and body == b"", (fmt, status, headers, body)
    else:
        assert count >= expected and result["text"] and result["words"]
        end = 0.0
        for w in result["words"]:
            assert end <= w["start"] <= w["end"] <= result["duration"], w
            end = w["end"]
        if wav.stem.startswith("one-late"):
            assert count == 1 and result["words"][0]["start"] >= 14 and end >= 16, result
        if wav.stem == "several":
            assert end >= 18, result
    (wav.parent / (wav.stem + ".json")).write_text(json.dumps(
        {"wav": result, "raw": raw, "ws": ws, "metrics": metrics,
         "wav_headers": wav_headers, "raw_headers": raw_headers}, ensure_ascii=False, indent=2))
    print(f"{wav.name} agrees across transports, pieces={count}, words={len(result['words'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--wav", required=True, nargs=2)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a, b = [read_wav(p) for p in args.wav]
    assert a[1] == b[1] == 16000
    cases = [("silence", 2, [], 0), ("short", 3, [(0, a[0])], 1),
             ("one-late", 22, [(15200, a[0])], 1),
             ("several", 22, [(200, a[0]), (3400, b[0]), (7700, a[0]), (10900, b[0]),
                              (15000, a[0]), (18200, b[0])], 2)]
    for name, seconds, clips, count in cases:
        path = args.out / (name + ".wav")
        compare(args.port, path, fixture(path, seconds, clips), count)
    # Replay after long requests to check detector state and the global offset reset.
    path = args.out / "one-late-after-long.wav"
    compare(args.port, path, fixture(path, 22, [(15200, a[0])]), 1)
    previous = json.loads((args.out / "one-late.json").read_text())
    current = json.loads((args.out / "one-late-after-long.json").read_text())
    assert previous["wav"] == current["wav"]


if __name__ == "__main__":
    main()
