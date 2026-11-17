#!/usr/bin/env python3
"""Insert clause separators at acoustic pauses instead of at keyword guesses.

The earlier keyword restorer regressed the command field because it split inside
single commands ("把左前车窗打开一半"). Paraformer already emits a CIF fire time
per token, so a long silence between two words is evidence of a clause boundary:
insert a separator only where the gap crosses a threshold, and never inside a
BPE word.

    llama-paraformer MODEL fixtures/*.wav --timestamps > run.tsv
    python3 pause_punct.py --in run.tsv --out run-punct.tsv --gap-ms 350
"""

import argparse

SPECIAL = ("<s>", "</s>", "<blank>", "<unk>", "<sos/eos>")

# A clause-initial particle is often coarticulated with the next word, so its
# CIF fire time lands after the pause and the separator shifts one word late.
# When the word before the gap is one of these, put the separator before it.
# A clause-initial particle is coarticulated with the next word, so its fire time
# can land after the pause and the separator shows up one word late. Shifting it
# left was measured and did not change the command field, so the simple rule
# stays; see the design doc for the example.


def split_words(tokens):
    """Group BPE pieces (a piece ending in @@ continues the same word)."""
    words, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if not tok.endswith("@@"):
            words.append(cur)
            cur = []
    if cur:
        words.append(cur)
    return words


def restore(tokens, times, gap_ms, lang):
    """Join tokens, inserting a separator where the pause exceeds the threshold."""
    if not tokens:
        return ""
    words = split_words(tokens)
    # index of the first token of each word
    starts, i = [], 0
    for w in words:
        starts.append(i)
        i += len(w)
    sep = "，" if lang == "zh" else ","
    out = ""
    for wi, w in enumerate(words):
        text = "".join(t[:-2] if t.endswith("@@") else t for t in w)
        if wi > 0:
            prev_last = starts[wi] - 1
            gap = times[starts[wi]] - times[prev_last]
            need_space = lang != "zh"
            if gap >= gap_ms:
                out += sep
                need_space = False
            if need_space:
                out += " "
        out += text
    out += "。" if lang == "zh" else "."
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--gap-ms", type=float, default=350)
    args = ap.parse_args()
    rows = 0
    with open(args.src, encoding="utf-8") as f, open(args.dst, "w", encoding="utf-8") as g:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7:
                g.write(line)
                continue
            meta = {}
            for fld in fields[1:-1]:
                k, _, v = fld.partition("=")
                meta[k] = v
            tokens = [t for t in meta.get("tok", "").split("|") if t]
            times = [float(x) for x in meta.get("ts_cs", "").split(",") if x]
            if len(tokens) != len(times):
                g.write(line)
                continue
            lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in tokens[0]) else "en"
            meta_text = restore(tokens, times, args.gap_ms, lang)
            fields[0] = fields[0]
            g.write("\t".join([fields[0], fields[1], fields[2], fields[3], meta_text]) + "\n")
            rows += 1
    print("pause_punct: %d rows, gap %.0f ms" % (rows, args.gap_ms))


if __name__ == "__main__":
    main()
