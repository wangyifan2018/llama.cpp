#!/usr/bin/env python3
"""Transcribe one fixture directory and report per-language error rates.

Reads a directory produced by gen_voice_fixtures.py (or any directory with a
manifest.tsv of name/language/text) and posts each clip to a running API, then
reports character error rate for Mandarin and word error rate for English
against the manifest text.

    python3 score_voice.py --dir /tmp/voice-eddy --url http://127.0.0.1:18099
    python3 score_voice.py --dir A --dir B --url ... --out compare.json

This compares voices, not channels. It does not apply the vehicle rewrite
profile unless the server has it configured, so run it against the same server
options you intend to compare.
"""

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request
import uuid


def read_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) >= 3:
            rows.append((fields[0], fields[1], fields[2]))
    return rows


def post_wav(url, path):
    boundary = f"----pf{uuid.uuid4().hex}"
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += path.read_bytes()
    body += f"\r\n--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="response_format"\r\n\r\njson'
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{url}/v1/audio/transcriptions", data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def norm_zh(text):
    return re.sub(r"[^\u4e00-\u9fff0-9]", "", text)


def norm_en(text):
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def edit_distance(ref, hyp):
    """Levenshtein over sequences (str or list)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1]


def score_dir(url, path):
    rows = read_manifest(path / "manifest.tsv")
    per_clip, err_count = [], {"zh": 0, "en": 0}
    lengths = {"zh": 0, "en": 0}
    lat = []
    for name, lang, ref in rows:
        clip = path / name
        if not clip.exists():
            print(f"missing {clip}", file=sys.stderr)
            continue
        t0 = time.time()
        try:
            hyp = post_wav(url, clip).get("text", "")
        except Exception as e:
            print(f"{name}: request failed: {e}", file=sys.stderr)
            continue
        lat.append((time.time() - t0) * 1000.0)

        if lang == "zh":
            r, h = norm_zh(ref), norm_zh(hyp)
        else:
            r, h = norm_en(ref), norm_en(hyp)
        d = edit_distance(r, h)
        per_clip.append({"name": name, "lang": lang, "ref": ref, "hyp": hyp,
                         "errors": d, "length": len(r)})
        err_count[lang] += d
        lengths[lang] += len(r)

    out = {"clips": len(per_clip), "per_clip": per_clip}
    for lang in ("zh", "en"):
        if lengths[lang]:
            out[f"{lang}_rate"] = 100.0 * err_count[lang] / lengths[lang]
            out[f"{lang}_err"] = err_count[lang]
            out[f"{lang}_len"] = lengths[lang]
    if lat:
        lat.sort()
        out["latency_ms"] = {"p50": round(statistics.median(lat), 1),
                             "p95": round(lat[int(len(lat) * 0.95)] if len(lat) > 1 else lat[0], 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, action="append", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18099")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    results = {}
    for d in args.dir:
        r = score_dir(args.url, d)
        results[d.name] = r
        print(f"{d.name:12s} clips={r['clips']:3d} "
              f"zh={r.get('zh_rate', float('nan')):6.2f}% ({r.get('zh_err', 0)}/{r.get('zh_len', 0)}) "
              f"en={r.get('en_rate', float('nan')):6.2f}% ({r.get('en_err', 0)}/{r.get('en_len', 0)}) "
              f"p50={r.get('latency_ms', {}).get('p50', 0):.0f}ms")

    if len(results) > 1:
        print("\nworst clips per voice:")
        for name, r in results.items():
            bad = sorted(r["per_clip"], key=lambda c: -c["errors"])[:3]
            bad = [c for c in bad if c["errors"]]
            if bad:
                for c in bad:
                    print(f"  {name:10s} {c['name']:14s} err={c['errors']} "
                          f"ref={c['ref']!r} hyp={c['hyp']!r}")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()