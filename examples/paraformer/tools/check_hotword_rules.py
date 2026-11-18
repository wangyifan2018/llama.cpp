#!/usr/bin/env python3
"""Check every span-rewrite rule against the vocabulary it will run with.

A rule whose wrong side or right side uses a token that is not in the
checkpoint's table is silently dropped: the parser prints
"replacement token not in vocab" once at load and then the rule never fires.
That failure looks identical to "the model is at its limit", so it needs to
be caught mechanically rather than by reading the log.

    python3 check_hotword_rules.py --vocab <dump> --hotwords <file>
"""
import argparse
import re
import sys


def load_vocab(path):
    toks = set()
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        # PF_DUMP_VOCAB prints "<id> <token>" with a space; a bare
        # one-token-per-line dump is also accepted. Splitting on the first
        # space keeps tokens that themselves contain nothing but this file is
        # only ever compared against `@@`-suffixed pieces, which never contain
        # a space.
        parts = line.split(" ", 1)
        toks.add(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return toks


def split_span(side):
    return side.split("+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--hotwords", required=True)
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    if not vocab:
        print("empty vocab dump")
        return 1

    bad = 0
    n_rule = 0
    for lineno, raw in enumerate(open(args.hotwords, encoding="utf-8"), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("R "):
            n_rule += 1
            m = re.match(r"^R\s+(\S+)\s+(.+?)\s*=>\s*(.+)$", line)
            if not m:
                print("%4d  malformed: %s" % (lineno, line))
                bad += 1
                continue
            margin, wrong, right = m.groups()
            try:
                float(margin)
            except ValueError:
                print("%4d  margin is not a number: %s" % (lineno, margin))
                bad += 1
            for label, side in (("wrong", wrong), ("right", right)):
                pieces = split_span(side)
                missing = [p for p in pieces if p not in vocab]
                if missing:
                    print("%4d  %s side has tokens not in vocab: %s"
                          % (lineno, label, missing))
                    print("      rule: %s" % line)
                    bad += 1
                # The last piece must be a word terminator. With the zh `@@`
                # convention that means it must NOT carry `@@`; with an
                # English `\u2581` convention the rule is different, so only the
                # `@@` convention is enforced here.
                if pieces and pieces[-1].endswith("@@"):
                    print("%4d  %s side ends with a continuation piece '%s'"
                          % (lineno, label, pieces[-1]))
                    print("      the last piece must close the word or the next "
                          "word is glued on")
                    bad += 1

    print("%d rules checked, %d problem(s)" % (n_rule, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())