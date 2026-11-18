#!/usr/bin/env python3
"""Production smoke test for a deployed Paraformer ASR service.

One command that checks the whole contract a caller depends on: health and
capabilities, the four response formats, punctuation, error handling, the
Realtime WebSocket (single and four concurrent sessions), silence and very short
segments, and the speaker-gate headers when the gate is on.

    python3 smoke_production.py --port 8899 --wav zh.wav
    python3 smoke_production.py --port 8899 --wav zh.wav --expect-gate --expect-vad

Exit code 0 only when every check passes.
"""

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("%-46s %s%s" % (name, "PASS" if ok else "FAIL", ("  " + detail) if detail else ""))
    return ok


def http_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, dict(r.headers), r.read()


def post_multipart(url, wav_path, fields, timeout=60):
    boundary = "----" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += ("--" + boundary + "\r\n").encode()
        body += ('Content-Disposition: form-data; name="%s"\r\n\r\n' % k).encode()
        body += str(v).encode() + b"\r\n"
    body += ("--" + boundary + "\r\n").encode()
    body += ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % os.path.basename(wav_path)).encode()
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += open(wav_path, "rb").read() + b"\r\n"
    body += ("--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


def post_raw(url, payload, timeout=60):
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


def read_wav(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2, "need mono 16-bit"
        return f.readframes(f.getnframes()), f.getframerate()


async def ws_session(url, pcm, results, idx, sessions, start_at):
    import websockets
    step = 640  # 20 ms
    await asyncio.sleep(max(0.0, start_at - asyncio.get_event_loop().time()))
    try:
        async with websockets.connect(url, max_size=None) as ws:
            await ws.recv()
            for off in range(0, len(pcm), step):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(pcm[off:off + step]).decode()}))
                await asyncio.sleep(0.002 if sessions > 1 else 0.02)
            endpoint = time.monotonic()
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if m.get("type") == "conversation.item.input_audio_transcription.completed":
                    results[idx] = {"ms": (time.monotonic() - endpoint) * 1000.0,
                                    "text": m.get("transcript", ""),
                                    "score": m.get("speaker_score"),
                                    "speaker_status": m.get("speaker_status"),
                                    "speaker_reason": m.get("speaker_reason")}
                    return
                if m.get("type") == "error":
                    results[idx] = {"error": m}
                    return
    except Exception as e:  # noqa: BLE001
        results[idx] = {"error": str(e)}


async def ws_bench(url, pcm, sessions):
    results = [None] * sessions
    start = asyncio.get_event_loop().time() + 0.1
    await asyncio.gather(*[ws_session(url, pcm, results, i, sessions, start) for i in range(sessions)])
    return results


async def ws_commit_lifecycle(url, pcm):
    import websockets

    async with websockets.connect(url) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        bad = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        check("empty WS commit returns a request error", bad.get("error", {}).get("type") == "invalid_request_error")

        for _ in range(2):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        ack = []
        final = []
        drained = False
        while len(final) < 2 or not drained:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if m["type"] == "input_audio_buffer.committed":
                ack.append(m["item_id"])
            elif m["type"] == "conversation.item.input_audio_transcription.completed":
                final.append((m["item_id"], bool(m["transcript"]) and m["usage"]["seconds"] > 0))
            elif m["type"] == "input_audio_buffer.cleared":
                drained = True
            elif m["type"] == "error":
                raise RuntimeError(m)
        check("back-to-back WS commits survive clear", len(set(ack)) == 2 and sorted(final) == sorted((item, True) for item in ack))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--expect-gate", action="store_true", help="fail unless /health reports speaker_gate")
    ap.add_argument("--expect-vad", action="store_true", help="fail unless /health reports server_vad")
    ap.add_argument("--expect-backend", help="require this actual backend on the first engine")
    ap.add_argument("--expect-degraded", action="store_true", help="expect explicit CPU fallback")
    args = ap.parse_args()

    base = "http://%s:%d" % (args.host, args.port)
    ws_url = "ws://%s:%d/v1/realtime" % (args.host, args.port)
    pcm, rate = read_wav(args.wav)

    # 1. health and capabilities
    status, _, body = http_json(base + "/health")
    health = json.loads(body)
    expected_status = "degraded" if args.expect_degraded else "ok"
    check("health", status == 200 and health.get("status") == expected_status, json.dumps(health))
    if args.expect_backend:
        backend = next(iter(health.get("backends", [])), {})
        check("actual backend and warmup placement",
              backend.get("actual") == args.expect_backend and backend.get("encoder_device_ops", 0) > 0 and
              backend.get("decoder_device_ops", 0) > 0, json.dumps(backend))
    if args.expect_gate:
        check("health reports speaker_gate", health.get("speaker_gate") is True)
    if args.expect_vad:
        check("health reports server_vad", health.get("server_vad") is True)

    # 2. models
    status, _, body = http_json(base + "/v1/models")
    models = json.loads(body)
    caps = models["data"][0].get("capabilities", {})
    check("models + capabilities", status == 200 and "streaming" in caps, json.dumps(caps))

    # 3. the four response formats
    st, hdr, body = post_multipart(base + "/v1/audio/transcriptions", args.wav, {})
    text = json.loads(body).get("text", "") if st == 200 else ""
    check("json", st == 200 and text != "", "%.0f ms infer, %.2f s queued" %
          (float(hdr.get("X-Infer-Ms", 0)), float(hdr.get("X-Queue-Ms", 0)) / 1000.0))

    st, hdr, body = post_multipart(base + "/v1/audio/transcriptions", args.wav, {"response_format": "text"})
    check("text", st == 200 and body.decode().strip() != "")

    st, hdr, body = post_multipart(base + "/v1/audio/transcriptions", args.wav,
                                   {"response_format": "verbose_json"})
    vj = json.loads(body) if st == 200 else {}
    check("verbose_json words/segments", st == 200 and len(vj.get("words", [])) > 0 and len(vj.get("segments", [])) > 0)
    check("headers report language", hdr.get("X-Language") in ("zh", "en", "mixed"), hdr.get("X-Language", ""))

    st, hdr, body = post_multipart(base + "/v1/audio/transcriptions", args.wav, {"response_format": "srt"})
    check("srt", st == 200 and b"-->" in body)

    st, hdr, body = post_multipart(base + "/v1/audio/transcriptions", args.wav, {"punctuate": "true"})
    punct = json.loads(body).get("text", "") if st == 200 else ""
    check("punctuate", st == 200 and (punct.endswith("。") or punct.endswith(".")), punct[:40])

    if args.expect_gate:
        score = hdr.get("X-Speaker-Score")
        state = hdr.get("X-Speaker-Status")
        check("speaker response state", (state == "assessed" and score is not None) or
              (state == "not_assessed" and score is None and hdr.get("X-Speaker-Reason")),
              "state=%s score=%s" % (state, score))

    # 4. error handling
    st, _, body = post_multipart(base + "/v1/audio/transcriptions", args.wav, {"response_format": "weird"})
    check("400 on unsupported format", st == 400 and b"error" in body)
    st, _, body = post_multipart(base + "/v1/audio/transcriptions", args.wav, {})
    # a missing file must be rejected: send a request without the file part
    boundary = "----" + uuid.uuid4().hex
    body = ("--" + boundary + "\r\n").encode() + \
           b'Content-Disposition: form-data; name="model"\r\n\r\nx\r\n' + \
           ("--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(base + "/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        urllib.request.urlopen(req, timeout=10)
        check("400 on missing file", False)
    except urllib.error.HTTPError as e:
        check("400 on missing file", e.code == 400)

    # 5. device endpoint (raw float32)
    import array
    a = array.array("h")
    a.frombytes(pcm)
    f32 = array.array("f", [v / 32768.0 for v in a])
    st, hdr, body = post_raw(base + "/asr/segment", f32.tobytes())
    check("/asr/segment", st == 200 and json.loads(body).get("text", "") != "")
    if args.expect_gate:
        state = json.loads(body).get("speaker_status")
        check("raw exposes speaker state", state in ("assessed", "not_assessed") and state == hdr.get("X-Speaker-Status"))
    st, _, _ = post_raw(base + "/asr/segment", struct.pack("<f", float("nan")))
    check("raw rejects non-finite PCM", st == 400)
    st, _, _ = post_raw(base + "/asr/segment", b"\x00" * (4 * (16000 * 30 + 1)))
    check("raw rejects segments over 30 s", st == 413)
    st, _, body = post_raw(base + "/asr/segment", struct.pack("<f", 0.0))
    check("valid sub-frame PCM returns empty text", st == 200 and json.loads(body).get("text") == "")

    # 6. digital silence is short-circuited with or without server VAD.
    st, hdr, body = post_raw(base + "/asr/segment", array.array("f", [0.0] * 32000).tobytes())
    check("silence returns empty text", st == 200 and json.loads(body).get("text") == "" and hdr.get("X-Empty") == "silence",
          "X-Empty=%s" % hdr.get("X-Empty", ""))
    if args.expect_gate:
        check("silence never carries a speaker score", hdr.get("X-Speaker-Status") == "not_assessed" and
              hdr.get("X-Speaker-Reason") == "silence" and "X-Speaker-Score" not in hdr)

    # 7. realtime, single and four concurrent
    res = asyncio.run(ws_bench(ws_url, pcm, 1))
    ok = res[0] and "text" in res[0] and res[0]["text"] != ""
    check("ws single commit", bool(ok), "%.0f ms" % res[0]["ms"] if ok else str(res[0]))
    res = asyncio.run(ws_bench(ws_url, pcm, 4))
    ok = all(r and "text" in r and r["text"] != "" for r in res)
    texts = set(r["text"] for r in res if r and "text" in r)
    check("ws four concurrent", ok and len(texts) == 1,
          "max %.0f ms, consistent=%s" % (max(r["ms"] for r in res if r and "ms" in r), len(texts) == 1))
    if args.expect_gate:
        check("ws overlapping commits expose speaker state",
              all(r.get("speaker_status") == "assessed" and r.get("score") is not None or
                  r.get("speaker_status") == "not_assessed" and r.get("score") is None and r.get("speaker_reason")
                  for r in res if r and "error" not in r) and
              any(r and r.get("speaker_reason") == "busy" for r in res))
    asyncio.run(ws_commit_lifecycle(ws_url, pcm))

    # 8. queue accounting
    status, _, body = http_json(base + "/metrics.json")
    m = json.loads(body)
    check("metrics", status == 200 and m.get("requests", 0) > 0, json.dumps(m))

    failed = [r for r in RESULTS if not r[1]]
    print()
    print("%d checks, %d failed" % (len(RESULTS), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
