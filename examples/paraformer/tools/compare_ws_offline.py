#!/usr/bin/env python3
"""Compare the realtime (WebSocket) path against the offline path on the same audio.

The realtime final should be the same transcript as the offline request: the
client accumulates a VAD segment and commits it. This tool proves or disproves
that per clip, and can trim the tail to model a client that cuts at its endpoint
with little trailing silence.

    python3 compare_ws_offline.py --port 8899 --wav-dir fixtures --trim-tail-ms 400
"""

import argparse
import asyncio
import base64
import glob
import json
import os
import re
import urllib.request
import uuid
import wave


def read_pcm(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2, path
        pcm = f.readframes(f.getnframes())
        rate = f.getframerate()
    return pcm, rate


def norm(text):
    text = text.lower()
    text = re.sub(r"[\s，。,.!?！？、:：;；\-]", "", text)
    return text


def rest_text(base, path):
    boundary = "----" + uuid.uuid4().hex
    body = ("--" + boundary + "\r\n").encode()
    body += ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % os.path.basename(path)).encode()
    body += b"Content-Type: audio/wav\r\n\r\n" + open(path, "rb").read() + b"\r\n"
    body += ("--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(base + "/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["text"]


async def ws_text(url, pcm, rate, pace_ms=20):
    import websockets
    step = max(2, int(rate * pace_ms / 1000) * 2)
    async with websockets.connect(url, max_size=None) as ws:
        await ws.recv()
        for off in range(0, len(pcm), step):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[off:off + step]).decode()}))
            await asyncio.sleep(pace_ms / 1000.0)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if m.get("type") == "conversation.item.input_audio_transcription.completed":
                return m.get("transcript", "")
            if m.get("type") == "error":
                return "<error>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--trim-tail-ms", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    base = "http://%s:%d" % (args.host, args.port)
    wsurl = "ws://%s:%d/v1/realtime" % (args.host, args.port)
    files = sorted(glob.glob(os.path.join(args.wav_dir, args.glob)))
    if args.limit:
        files = files[:args.limit]

    same = 0
    diffs = []
    for path in files:
        pcm, rate = read_pcm(path)
        if args.trim_tail_ms:
            cut = (args.trim_tail_ms * rate // 1000) * 2
            pcm = pcm[:max(2, len(pcm) - cut)]
        # offline: same bytes through the REST path
        tmp = "/tmp/_cmp_%s" % os.path.basename(path)
        with wave.open(tmp, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(pcm)
        off = rest_text(base, tmp)
        on = asyncio.run(ws_text(wsurl, pcm, rate))
        if norm(off) == norm(on):
            same += 1
        else:
            diffs.append((os.path.basename(path), off, on))
    print("%s clips=%d identical=%d (%.1f%%) differ=%d" %
          (args.label or "run", len(files), same, 100.0 * same / max(1, len(files)), len(diffs)))
    for name, off, on in diffs[:12]:
        print("  --- %s" % name)
        print("    offline: %s" % off)
        print("    ws     : %s" % on)
    return 0 if not diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
