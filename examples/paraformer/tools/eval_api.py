#!/usr/bin/env python3
"""Grade the shipped accuracy through the OpenAI endpoint.

The reproduce gate drives the CLI. Production serves /v1/audio/transcriptions, so
this posts each WAV there and writes the same run table eval_vehicle.py reads:
  <path> <TAB> audio_ms=... <TAB> wall_ms=... <TAB> rtf=... <TAB> text

    python3 eval_api.py --port 8899 --wav-dir fixtures --out run.tsv
"""

import argparse
import glob
import json
import os
import statistics
import time
import urllib.request
import uuid
import wave


def post(base, path, fields, timeout=120):
    boundary = "----" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += ("--" + boundary + "\r\n").encode()
        body += ('Content-Disposition: form-data; name="%s"\r\n\r\n' % k).encode()
        body += str(v).encode() + b"\r\n"
    body += ("--" + boundary + "\r\n").encode()
    body += ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % os.path.basename(path)).encode()
    body += b"Content-Type: audio/wav\r\n\r\n" + open(path, "rb").read() + b"\r\n"
    body += ("--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(base + "/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read()
        head = dict(r.headers)
    return (time.monotonic() - t0) * 1000.0, head, payload


def audio_ms(path):
    with wave.open(path, "rb") as f:
        return 1000.0 * f.getnframes() / f.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--punctuate", action="store_true")
    args = ap.parse_args()

    base = "http://%s:%d" % (args.host, args.port)
    files = sorted(glob.glob(os.path.join(args.wav_dir, args.glob)))
    rows, walls, infers, queues = [], [], [], []
    for path in files:
        fields = {"punctuate": "true"} if args.punctuate else {}
        wall, head, payload = post(base, path, fields)
        text = json.loads(payload)["text"]
        dur = audio_ms(path)
        rows.append("%s\taudio_ms=%.3f\twall_ms=%.3f\trtf=%.4f\t%s" %
                    (path, dur, wall, wall / max(1.0, dur), text))
        walls.append(wall)
        infers.append(float(head.get("X-Infer-Ms", 0)))
        queues.append(float(head.get("X-Queue-Ms", 0)))
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")

    def pct(v, q):
        s = sorted(v)
        k = (len(s) - 1) * q / 100.0
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    print("clips=%d  wall p50=%.0f p95=%.0f max=%.0f ms | infer p50=%.0f p95=%.0f | queue p95=%.0f"
          % (len(files), pct(walls, 50), pct(walls, 95), max(walls),
             pct(infers, 50), pct(infers, 95), pct(queues, 95)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
