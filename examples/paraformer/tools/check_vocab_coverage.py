#!/usr/bin/env python3
"""Check whether a new command domain's words survive the checkpoint vocabulary.

A logit bias can only choose among tokens the model already emits: it cannot
write a word the vocabulary lacks, and it cannot change how many tokens a word
costs. So before writing hotwords for a new domain, find out which of its words
are spellable and which are not. The answer decides the fix -- a gain, a
conditional tail rule, or a span rewrite -- and it says plainly which words no
bias can reach.

Two conventions, and they are opposite, so the script detects them instead of
assuming:

  zh checkpoint   continuation is a trailing "@@"  (导@@ + 航)
  en checkpoint   word start is a leading "▁"      (▁nav@@ ig@@ ate)
  this zh model   emits Chinese as ONE TOKEN PER CHARACTER, so a Chinese word
                  is never assembled -- it is a sequence of whole tokens, and
                  the question becomes whether each character exists at all.

    PF_DUMP_VOCAB= build-x-asr/bin/llama-paraformer MODEL WAV --quiet 2>&1 \
        | grep '^pf-vocab:' | sed 's/^pf-vocab: [0-9]* //' > /tmp/vocab.txt
    python3 tools/check_vocab_coverage.py --vocab /tmp/vocab.txt --terms terms.txt
"""

import argparse


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f}


def is_cjk(ch):
    return "\u4e00" <= ch <= "\u9fff"


def detect_marker(vocab):
    """Which continuation convention this checkpoint uses.

    The zh checkpoint marks continuation with a trailing "@@" and has no
    leading "▁" tokens at all -- English words in it are assembled from
    ``dr@@|iv@@|er`` style pieces, not from whole ``▁driver`` tokens. Checking
    for "▁" first and then giving up would report every English word as
    unspellable on a checkpoint that spells them fine, so the tail marker wins
    whenever it is present.
    """
    has_tail = any(t.endswith("@@") for t in vocab)
    has_lead = any(t.startswith("\u2581") for t in vocab)
    if has_tail:
        return "tail"
    if has_lead:
        return "lead"
    return "unknown"


def check_zh(word, vocab, marker):
    """Every character must exist. Missing character = unspellable word."""
    chars = [c for c in word if is_cjk(c)]
    missing = [c for c in chars if c not in vocab]
    if missing:
        return "missing", missing
    if len(chars) > 1 and word in vocab:
        return "wholes", []
    if len(chars) == 1:
        return "wholes", []
    return "per-char", []


def check_en(word, vocab, marker):
    """Can this word be rendered as ONE word, i.e. with no spurious space?

    The renderer decides word boundaries from the marker, not from the
    spelling: a token that does not carry the continuation marker ends the
    word. So a cover is only usable when every non-final piece carries the
    marker and the final piece does not. Getting that backwards is silent --
    the transcript still reads almost right, just with a space in the middle
    of the word (`bright est`, `suns had e`, `highestlevel`), and a span
    rewrite built on such a cover manufactures the defect it was meant to
    remove. Verified on the zh checkpoint:

        R 99.0 re@@+r => re@@+ar@@   ->  `rearfot lights`   (final piece @@@)
        R 99.0 re@@+r => re@@+ar    ->  `rear fot lights`   (correct)

    Returns (kind, detail) where kind is one of:
        wholes   the word is a single vocabulary token
        pieces   a contiguous cover exists; detail is the piece list
        split    every character is coverable but no contiguous cover exists,
                 so the model can say the word and a rewrite still cannot
                 produce it in one piece
        missing  a character has no token at all
    """
    lower = word.lower()
    if marker == "lead":
        return _check_lead(lower, vocab)

    if lower in vocab:
        return "wholes", []
    pieces, i = [], 0
    n = len(lower)
    while i < n:
        hit = None
        for length in range(min(10, n - i), 0, -1):
            chunk = lower[i:i + length]
            # A non-final piece must continue a word, so only the "@@" form is
            # legal here; the last piece must end one, so only the bare form.
            # Test the form directly rather than requiring the bare chunk to
            # exist first: `co` may be absent while `co@@` is present, and
            # gating on the bare spelling collapses the cover to single letters.
            form = chunk + "@@" if i + length < n else chunk
            if form in vocab:
                hit = (form, length)
                break
        if hit is None:
            # Fall back to any token that starts here, carried with the marker.
            # That tells `split` apart from `missing`: the character is spoken
            # for, it just cannot be joined to the previous piece.
            for length in range(min(10, n - i), 0, -1):
                chunk = lower[i:i + length]
                if chunk + "@@" in vocab:
                    return "split", pieces + [chunk + "@@"]
            return "missing", [lower[i:]]
        pieces.append(hit[0])
        i += hit[1]
    return "pieces", pieces


def _check_lead(lower, vocab):
    """The leading-marker convention, used by the en checkpoint."""
    if "\u2581" + lower in vocab or lower in vocab:
        return "wholes", []
    pieces, i, first = [], 0, True
    n = len(lower)
    while i < n:
        hit = None
        for length in range(min(10, n - i), 0, -1):
            chunk = lower[i:i + length]
            form = ("\u2581" + chunk) if first else chunk
            if form in vocab:
                hit = (form, length)
                break
        if hit is None:
            return "missing", [lower[i:]]
        pieces.append(hit[0])
        i += hit[1]
        first = False
    return "pieces", pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--terms", required=True, help="one term per line")
    ap.add_argument("--show-ok", action="store_true", help="also print spellable terms")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    marker = detect_marker(vocab)
    print("vocab %d tokens, continuation convention: %s" % (len(vocab), marker))

    groups = {"wholes": [], "per-char": [], "pieces": [], "split": [], "missing": []}
    terms = []
    with open(args.terms, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                terms.append(line)

    for word in terms:
        if any(is_cjk(c) for c in word):
            kind, detail = check_zh(word, vocab, marker)
        else:
            kind, detail = check_en(word, vocab, marker)
        groups[kind].append((word, detail))

    for word, detail in groups["missing"]:
        print("MISSING   %-14s lacking: %s" % (word, "".join(detail)))
    for word, pieces in groups["split"]:
        print("split     %-14s stops after %s: no single-token cover"
              % (word, "|".join(pieces)))
    for word, pieces in groups["pieces"]:
        print("pieces    %-14s %d tokens: %s" % (word, len(pieces), "|".join(pieces)))
    if args.show_ok:
        for word, _ in groups["wholes"]:
            print("whole     %-14s" % word)
        for word, _ in groups["per-char"]:
            print("per-char  %-14s" % word)

    print("\n%d terms: %d whole-token, %d per-character, %d assembled, %d split, %d unspellable"
          % (len(terms), len(groups["wholes"]), len(groups["per-char"]),
             len(groups["pieces"]), len(groups["split"]), len(groups["missing"])))
    if groups["split"]:
        print("\n`split` words are spoken for character by character but have no cover whose")
        print("non-final pieces continue and whose final piece ends. A span rewrite needs")
        print("exactly that shape, so these are unreachable too -- the renderer would put a")
        print("space inside the word rather than join it.")
    if groups["missing"]:
        print("\nUnspellable words cannot be reached by a gain, a conditional rule or a")
        print("span rewrite against these spellings. Either retrain, or rewrite onto a")
        print("word the model can already say and let the downstream map it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())