#!/usr/bin/env python3
"""Run one engine configuration over a labelled set and report CER and WER.

The label is the synthesiser input used to build the clip, so this measures
recognition on clean synthetic speech, not field accuracy. Use it to rank
variants (quantisation, backend, join location) against each other.
"""

import argparse
import os
import re
import subprocess
import sys


SPECIAL = ("<sos/eos>", "<unk>", "<blank>")


def strip_special(s):
    hit = False
    for tok in SPECIAL:
        if tok in s:
            hit = True
            s = s.replace(tok, "")
    return s, hit


def norm_zh(s):
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[，。！？、；：,.!?;:\"'()\[\]<>“”‘’—…-]", "", s)


def norm_en(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bin", default="build-x-asr/bin/llama-x-asr")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--suffix", default="", help="appended to the manifest file names")
    parser.add_argument("--label", default="")
    parser.add_argument("--from-file", help="parse a saved TSV of the binary instead of running it")
    parser.add_argument("extra", nargs="*", help="extra arguments for the binary")
    args = parser.parse_args()

    rows = []
    for line in open(args.manifest, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        name, lang, text = parts[0], parts[1], "\t".join(parts[2:])
        rows.append((name, lang, text))

    root = os.path.dirname(os.path.abspath(args.manifest))
    files = []
    for name, lang, text in rows:
        if args.suffix and name.endswith(".wav"):
            name = name[: -len(".wav")] + args.suffix + ".wav"
        files.append(os.path.join(root, name))

    if args.from_file:
        out = open(args.from_file, encoding="utf-8", errors="replace").read()
    else:
        cmd = [args.bin, args.model, args.device] + files + list(args.extra)
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    got = {}
    for line in out.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            got[fields[0].split("/")[-1]] = fields[-1]

    zh_err = zh_len = en_err = en_len = 0
    misses = []
    special = 0
    for name, lang, text in rows:
        fname = name if not args.suffix else name[: -len(".wav")] + args.suffix + ".wav"
        hyp = got.get(fname, "")
        hyp, had_special = strip_special(hyp)
        special += 1 if had_special else 0
        if lang == "zh":
            ref_n, hyp_n = norm_zh(text), norm_zh(hyp)
            d = edit_distance(ref_n, hyp_n)
            zh_err += d
            zh_len += len(ref_n)
            if d:
                misses.append((fname, lang, d, len(ref_n), text, hyp))
        else:
            ref_n, hyp_n = norm_en(text).split(), norm_en(hyp).split()
            d = edit_distance(ref_n, hyp_n)
            en_err += d
            en_len += len(ref_n)
            if d:
                misses.append((fname, lang, d, len(ref_n), text, hyp))

    label = args.label or args.device
    print("%s: zh CER %.2f%% (%d/%d), en WER %.2f%% (%d/%d), %d/%d clips with errors, %d with a special token"
          % (label,
             100.0 * zh_err / max(zh_len, 1), zh_err, zh_len,
             100.0 * en_err / max(en_len, 1), en_err, en_len,
             len(misses), len(rows), special))
    for fname, lang, d, n, ref, hyp in misses[:12]:
        print("  %-18s %s  %d/%d  ref=%s  hyp=%s" % (fname, lang, d, n, ref, hyp))


if __name__ == "__main__":
    main()
