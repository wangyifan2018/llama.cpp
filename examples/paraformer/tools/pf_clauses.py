#!/usr/bin/env python3
"""Restore clause separators in Paraformer output so the semantic grader can
split multi-command utterances.

Paraformer emits no punctuation. The cabin evaluator splits clauses on
punctuation (Chinese) or on punctuation plus "and" (English), so a raw Chinese
two-command utterance grades as one clause and the whole command is lost. This
is a post-processing stage, not a model change: it inserts one separator before
known clause starters and nothing else.

    python3 pf_clauses.py --in run.tsv --out run-clauses.tsv
    python3 examples/x-asr/tools/eval_vehicle.py \
        --pack pack=manifest.tsv --run raw=run.tsv --run clauses=run-clauses.tsv
"""

import argparse
import os
import re
import sys

ZH_STARTERS = (
    "然后", "接着", "再", "并且", "并", "同时", "把", "打开", "关闭", "调高", "调低",
    "调到", "设置", "导航", "播放", "提醒", "途经", "到了", "之后", "给",
)
EN_STARTERS = (" then ", " also ", " plus ", " when ", " once ")


def restore(text, lang):
    """Insert a clause separator before each starter, never at position 0."""
    if lang == "zh":
        out = text
        for word in ZH_STARTERS:
            out = re.sub(r"(?<=[^，。；、\s])" + re.escape(word), "，" + word, out)
        return out
    out = " " + text.strip() + " "
    for word in EN_STARTERS:
        out = out.replace(word, "," + word)
    # "and" already splits in the English grader; normalise the space before it
    out = re.sub(r"\s+", " ", out).strip()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--no-en", action="store_true", help="leave English untouched")
    args = parser.parse_args()

    rows = 0
    changed = 0
    with open(args.src, encoding="utf-8") as f, open(args.dst, "w", encoding="utf-8") as g:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                g.write(line)
                continue
            lang = "zh" if re.search(r"[\u4e00-\u9fff]", fields[4]) else "en"
            new = fields[4] if (args.no_en and lang == "en") else restore(fields[4], lang)
            if new != fields[4]:
                changed += 1
            fields[4] = new
            g.write("\t".join(fields) + "\n")
            rows += 1
    print("pf-clauses: %d rows, %d changed" % (rows, changed))


if __name__ == "__main__":
    main()
