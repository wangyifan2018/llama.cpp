#!/usr/bin/env python3
"""Decide whether a fixture set is measuring the engine or the synthesizer.

A voice sweep that only runs this engine cannot tell "the model does not
generalise" from "these clips are not intelligible". Both produce high error.
`score_voice.py` reports the engine's number; this script runs a second,
independent recogniser over the same clips and compares.

The distinction matters because the two conclusions need opposite responses.
If the clips are damaged, the sweep measured the TTS and the engine may be
fine. If the clips are clean and only this engine fails, that is a product
defect worth fixing.

    # whisper's TSV files, one directory per voice, name <clip id>.tsv
    python3 check_audio_quality.py --refs tools/cabin_commands.tsv \
        --arm ours=ours.tsv --arm whisper=whisper-out/

`--arm` accepts either a transcript TSV (`path<TAB>...<TAB>text`, the CLI and
score_voice.py format) or a directory tree of whisper `--output_format tsv`
files, `<dir>/<voice>/<clip>.tsv`. Whisper emits Traditional Chinese, so its
output is converted with OpenCC before scoring; without that, every Traditional
character counts as an error and the comparison is meaningless.
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Match examples/x-asr/tools/eval_cer.py so these rates are comparable with the
# cabin table in PRODUCTION.md. norm_zh there strips punctuation and whitespace
# but keeps Latin letters: a Mandarin clip that decodes to "ghonqq" is wrong and
# has to cost characters, not vanish from the denominator.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "x-asr" / "tools"))
from eval_cer import edit_distance, norm_en, norm_zh  # noqa: E402


def read_refs(path):
    """`name<TAB>lang<TAB>text` -> {clip id: (lang, text)}."""
    refs = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        refs[Path(fields[0]).stem] = (fields[1], fields[2])
    return refs


def read_tsv(path):
    """`path<TAB>meta=..<TAB>text` -> {clip id: text}."""
    out = {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        out[Path(fields[0]).stem] = fields[-1]
    return out


def read_whisper_tree(root):
    """`<root>/<voice>/<clip>.tsv` -> {voice: {clip id: text}}."""
    out = {}
    for voice_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        clips = {}
        for tsv in sorted(voice_dir.glob("*.tsv")):
            lines = tsv.read_text(encoding="utf-8").splitlines()[1:]
            clips[tsv.stem] = "".join(l.split("\t")[-1] for l in lines if l.strip())
        out[voice_dir.name] = clips
    return out


def load_t2s():
    try:
        from opencc import OpenCC
    except ImportError:
        sys.exit("opencc is required to compare against whisper: "
                 "pip install opencc-python-reimplemented")
    return OpenCC("t2s").convert


def edit_distance(ref, hyp):
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


def norm_arm_zh(text, t2s):
    return norm_zh(t2s(text))


def score(clips, refs, t2s, lang_filter=None):
    err, total, per_clip = {"zh": 0, "en": 0}, {"zh": 0, "en": 0}, []
    for cid, hyp in clips.items():
        if cid not in refs:
            continue
        lang, ref = refs[cid]
        if lang_filter and lang != lang_filter:
            continue
        if lang == "zh":
            r, h = norm_zh(ref), norm_arm_zh(hyp, t2s)
        else:
            # norm_en returns a string; edit_distance would then compare
            # characters and report a character error rate, not WER.
            r, h = norm_en(ref).split(), norm_en(hyp).split()
        d = edit_distance(r, h)
        err[lang] += d
        total[lang] += len(r)
        per_clip.append({"id": cid, "lang": lang, "errors": d, "length": len(r),
                         "ref": ref, "hyp": hyp})
    return err, total, per_clip


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refs", type=Path, required=True,
                    help="cabin_commands.tsv with name/lang/text")
    ap.add_argument("--arm", action="append", required=True,
                    metavar="LABEL=PATH", help="transcript TSV or whisper tree")
    ap.add_argument("--lang", choices=("zh", "en"), help="score one language only")
    ap.add_argument("--worst", type=int, default=0,
                    help="print this many worst clips per arm and voice")
    args = ap.parse_args()

    refs = read_refs(args.refs)
    t2s = load_t2s()
    arms = []
    for spec in args.arm:
        label, _, path = spec.partition("=")
        p = Path(path)
        if p.is_dir():
            arms.append((label, read_whisper_tree(p)))
        elif p.is_file():
            arms.append((label, {"all": read_tsv(p)}))
        else:
            sys.exit(f"no such path: {path}")

    voices = sorted({v for _, tree in arms for v in tree})
    header = f"{'voice':12s}" + "".join(f"{label:>14s}" for label, _ in arms)
    print(header)
    summary = {}
    for voice in voices:
        row = f"{voice:12s}"
        for label, tree in arms:
            if voice not in tree:
                row += f"{'--':>14s}"
                continue
            err, total, _ = score(tree[voice], refs, t2s, args.lang)
            lng = args.lang or ("zh" if total["zh"] else "en")
            rate = 100.0 * err[lng] / total[lng] if total[lng] else float("nan")
            summary[(voice, label)] = rate
            row += f"{rate:13.2f}%"
        print(row)

    if len(arms) == 2:
        (la, ta), (lb, tb) = arms
        print(f"\n{la} minus {lb}:")
        for voice in voices:
            a, b = summary.get((voice, la)), summary.get((voice, lb))
            if a is None or b is None:
                continue
            verdict = "clips suspect" if a > b else "engine suspect"
            print(f"  {voice:12s} {a:6.2f} - {b:6.2f} = {a - b:+6.2f}  {verdict}")

    if args.worst:
        print()
        for label, tree in arms:
            for voice in voices:
                if voice not in tree:
                    continue
                _, _, per_clip = score(tree[voice], refs, t2s, args.lang)
                bad = sorted(per_clip, key=lambda c: -c["errors"])[:args.worst]
                for c in bad:
                    if c["errors"]:
                        print(f"  {label:10s} {voice:10s} {c['id']:12s} "
                              f"err={c['errors']}/{c['length']} "
                              f"ref={c['ref']!r} hyp={c['hyp']!r}")


if __name__ == "__main__":
    main()