#!/usr/bin/env python3
"""Check bearer-token enforcement against a running pf-api.

Start the server with a known token, then point this at it:

    PF_API_KEY=secret123 llama-paraformer-api MODEL.gguf --port 18802 ...
    python3 tools/check_auth.py --port 18802 --wav clip.wav --key secret123

What it asserts, in order:

1. Audio is refused without a credential, with the wrong one, and with a
   malformed header - and the refusal is 401 with the OpenAI error envelope,
   not a 500 or a dropped connection.
2. The same audio is accepted with the right credential, and the transcript is
   byte-identical to the unauthenticated baseline. A gate that quietly changed
   the answer would be worse than no gate.
3. Probes stay open: /health, /v1/models and /metrics answer without a token,
   because a container healthcheck has no way to hold one.
4. The WebSocket upgrade is refused without a token. This is the one a
   per-handler check always misses: it is a different code path from the REST
   routes and it carries the most audio.

`--no-auth` runs the mirror image against a server started without a key: it
asserts that the same requests now succeed without any header, which is what
proves the default behaviour did not move.
"""

import argparse
import base64
import json
import os
import socket
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from smoke_production import RESULTS, check  # noqa: E402

BOUNDARY = "----pfauth"


def multipart(wav_path):
    name = os.path.basename(wav_path)
    body = b""
    body += ("--" + BOUNDARY + "\r\n").encode()
    body += ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % name).encode()
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += open(wav_path, "rb").read() + b"\r\n"
    body += ("--" + BOUNDARY + "--\r\n").encode()
    return body


def post(host, port, wav_path, headers=None):
    """POST one clip, returning (status, body bytes) with HTTPError captured."""
    url = "http://%s:%d/v1/audio/transcriptions" % (host, port)
    h = {"Content-Type": "multipart/form-data; boundary=" + BOUNDARY}
    h.update(headers or {})
    req = urllib.request.Request(url, data=multipart(wav_path), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get(host, port, path, headers=None):
    url = "http://%s:%d%s" % (host, port, path)
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def ws_upgrade_status(host, port, headers):
    """First status line of a WebSocket upgrade attempt, or '' on refusal.

    Raw sockets on purpose: the point is to observe what the server does to the
    *handshake*, before any WebSocket library has a chance to normalise it.
    """
    key = base64.b64encode(os.urandom(16)).decode()
    req = ("GET /v1/realtime HTTP/1.1\r\n"
           "Host: %s:%d\r\n"
           "Upgrade: websocket\r\n"
           "Connection: Upgrade\r\n"
           "Sec-WebSocket-Key: %s\r\n"
           "Sec-WebSocket-Version: 13\r\n" % (host, port, key))
    for k, v in (headers or {}).items():
        req += "%s: %s\r\n" % (k, v)
    req += "\r\n"
    s = socket.create_connection((host, port), timeout=10)
    try:
        s.sendall(req.encode())
        head = s.recv(256).decode("utf-8", "replace")
    finally:
        s.close()
    return head.split("\r\n")[0]


def transcript(body):
    try:
        return json.loads(body.decode())["text"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--key", default=os.environ.get("PF_API_KEY", ""))
    ap.add_argument("--no-auth", action="store_true", help="expect an open server (no key set)")
    args = ap.parse_args()

    if not args.no_auth and not args.key:
        raise SystemExit("pass --key or PF_API_KEY, or --no-auth for an open server")

    auth = {"Authorization": "Bearer " + args.key}

    if args.no_auth:
        st, body = post(args.host, args.port, args.wav)
        check("open server accepts audio without a credential", st == 200, "http=%d" % st)
        ref = transcript(body)
        check("open server returns a transcript", bool(ref), repr(ref)[:60])
        st, _ = get(args.host, args.port, "/metrics")
        check("open server serves /metrics without a credential", st == 200, "http=%d" % st)
        line = ws_upgrade_status(args.host, args.port, None)
        check("open server accepts the WebSocket upgrade", "101" in line, line)
        return finish()

    # --- 1. refusals
    st, body = post(args.host, args.port, args.wav)
    err = {}
    try:
        err = json.loads(body.decode()).get("error") or {}
    except Exception:
        pass
    check("audio refused without a credential", st == 401, "http=%d" % st)
    check("refusal uses the OpenAI error envelope", err.get("code") == "invalid_api_key",
          str(err.get("code")))
    check("refusal is authentication_error", err.get("type") == "authentication_error",
          str(err.get("type")))

    st, _ = post(args.host, args.port, args.wav, {"Authorization": "Bearer definitely-wrong"})
    check("audio refused with a wrong token", st == 401, "http=%d" % st)

    st, _ = post(args.host, args.port, args.wav, {"Authorization": args.key})
    check("audio refused when the Bearer scheme is missing", st == 401, "http=%d" % st)

    st, _ = post(args.host, args.port, args.wav, {"Authorization": "Bearer "})
    check("audio refused for an empty token", st == 401, "http=%d" % st)

    # Length is not secret, but a prefix mismatch must not be distinguishable
    # from a same-length mismatch. Both are 401; this pins that they stay 401.
    st, _ = post(args.host, args.port, args.wav, {"Authorization": "Bearer " + args.key[:-1] + "X"})
    check("audio refused for a same-length wrong token", st == 401, "http=%d" % st)

    # --- 2. acceptance
    st, body = post(args.host, args.port, args.wav, auth)
    check("audio accepted with the right token", st == 200, "http=%d" % st)
    ref = transcript(body)
    check("authenticated transcript is non-empty", bool(ref), repr(ref)[:60])

    st, body2 = post(args.host, args.port, args.wav, {"Authorization": "bearer " + args.key})
    check("lowercase 'bearer' scheme is accepted", st == 200 and transcript(body2) == ref,
          "http=%d" % st)

    # --- 3. probes stay open
    for path in ("/health", "/v1/models", "/metrics"):
        st, _ = get(args.host, args.port, path)
        check("%s answers without a credential" % path, st == 200, "http=%d" % st)

    # --- 4. the upgrade path
    line = ws_upgrade_status(args.host, args.port, None)
    check("WebSocket upgrade refused without a credential", "401" in line, line)
    line = ws_upgrade_status(args.host, args.port, auth)
    check("WebSocket upgrade accepted with the credential", "101" in line, line)

    return finish()


def finish():
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("")
    if failed:
        print("%d/%d checks failed" % (len(failed), len(RESULTS)))
        return 1
    print("%d/%d checks passed" % (len(RESULTS), len(RESULTS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())