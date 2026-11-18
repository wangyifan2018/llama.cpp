#!/usr/bin/env python3
"""Grade a mixed-language (中英混说) transcript run by command field.

Character CER is the wrong instrument for mixed utterances. `导航到 Beijing 机场`
recognised as `导航到北京机场` is correct and CER scores it as seven errors; a
hotword profile tuned to lower CER would be tuned against a metric that is
actively wrong. What the actuator needs is the field tuple
(device, position, state, value), so that is what this script compares.

The parser is deliberately reused from `examples/x-asr/tools/eval_vehicle.py`
rather than reimplemented: a private copy would drift and the two engines would
stop being comparable. Only the reference-side annotation lives here, because
the mixed fixture set is Paraformer's own.

    python3 tools/eval_mixed.py --run run.tsv
    python3 tools/eval_mixed.py --run base=run.tsv --run hot=run-hot.tsv

`--run <label>=<tsv>` takes a run file with `<wav>\t<transcript>` lines, i.e.
exactly what `llama-paraformer ... > run.tsv` produces.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "x-asr", "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from eval_vehicle import parse, split_clauses  # noqa: E402

# What each clip must produce downstream. field grammar:
#   device | position | state | value     (state: on/off/?; value: - if none)
# Position is only annotated when the utterance actually names one, because a
# missing position is a different defect from a wrong position.
MIXED = {
    "mx_zh01": "window|fl|on|-",
    "mx_zh02": "temp|-|on|21",
    "mx_zh03": "seat_heat|-|off|-",
    "mx_zh04": "ambient|-|on|warm_white",
    "mx_zh05": "reading|rr|on|-",
    "mx_zh06": "route|-|on|-",          # 导航: no cabin device; see NOTES
    "mx_zh07": "media|-|on|-",          # 播放: no cabin device
    "mx_zh08": "footwell|-|on|40",
    "mx_en01": "recirc|-|on|-",
    "mx_en02": "temp|fl|on|21",
    "mx_en03": "window|rl|on|-",
    "mx_zh09": "call|-|on|-",           # 打电话: no cabin device
    "mx_zh10": "volume|-|on|-",         # 音量: no cabin device
    "mx_zh11": "carplay|-|on|-",        # CarPlay: no cabin device
    "mx_zh12": "temp|-|on|22",
}

# mx_zh06/07/09/10/11 command no cabin actuator. eval_vehicle.parse() returns
# no clause for them by construction, so they can only be graded on whether the
# transcript is textually right. They are excluded from the field score and
# reported separately as `nocab`; folding them in would put a constant zero into
# the denominator and make every profile look equally bad.
NOCAB = {"mx_zh06", "mx_zh07", "mx_zh09", "mx_zh10", "mx_zh11"}


def load(path, clip_id):
    """Read a run file into {clip_id: transcript}."""
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            name = os.path.basename(fields[0])
            key = name[:-4] if name.endswith(".wav") else name
            # llama-paraformer writes <wav>\t<stats>\t<text>; take the last
            # field so both the graded run and a plain "<wav>\t<text>" work.
            rows[key] = fields[-1].strip()
    return rows


# Mixed utterances come out of the model with no spaces around the Chinese
# runs (`打开front left window`), because the zh vocabulary's `@@` joints only
# mark Latin word boundaries. Two instrument defects follow, both of which
# would otherwise be scored as ASR errors:
#
#   1. detect_device() with lang="zh" matches against a whitespace-stripped
#      string, so the multi-word English patterns (`front left`, `seat heater`)
#      can never fire. It also strips the spaces that the English STATE_RULES
#      (`\bturn on\b`, `\bset\b`) need, so a correct `turn on` grades as `?`.
#   2. Neither path reads an English numeral before a unit (`twenty one度`).
#
# Fixing this needs no change to the shared parser: the en path keeps the
# spaces and gets (1) right, and (2) is the same ITN that Whisper and
# SenseVoice already do, so grading without it would flatter the comparison.
_EN_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_TENS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
_UNITS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}


def itn(text):
    """Turn English numerals into digits and `degrees` into 度.

    Applied to the hypothesis only, and only to numerals, so it cannot hide a
    word error — `twenty one` and `21` are the same number said two ways, which
    is the whole point of comparing against engines that emit digits.
    """
    out = text.lower()

    # `\b` is unusable here: Python's \w covers CJK, so in `twenty one度` the
    # `one`/`度` joint has no word boundary and `\bone\b` never matches. A
    # lookaround on ASCII letters only is what the caller actually means.
    B0, B1 = r"(?<![A-Za-z])", r"(?![A-Za-z])"

    def tens(m):
        return " %d " % (_EN_NUM[m.group(1)] + _EN_NUM[m.group(2)])

    out = re.sub(B0 + r"(" + "|".join(_TENS) + r")[ -](" + "|".join(_UNITS) + r")" + B1,
                 tens, out)
    for word, digit in _EN_NUM.items():
        out = re.sub(B0 + word + B1, " %d " % digit, out)
    # `22 degrees` would otherwise satisfy neither reader: en2int() cannot read
    # a digit and the zh regex cannot read a Latin unit. Collapsing the unit
    # keeps one code path instead of two.
    out = re.sub(r"\s*(?<![A-Za-z])degrees?(?![A-Za-z])", "度", out)
    return re.sub(r"[ \t]+", " ", out).strip()


def fields_of(transcript):
    """Parse a mixed transcript into a list of field tuples.

    Both languages are tried and the better parse wins: more clauses, then more
    resolved fields. Clause count alone is not enough because a mixed single
    command ties at one clause and the two paths then disagree about position —
    the zh path strips the spaces that `front left` needs, so it loses `fl`
    while still finding the device. Counting resolved fields breaks that tie in
    favour of the path that actually matched a position pattern, which is a
    fact from the text rather than a preference.
    """
    best = []
    best_key = None
    for lang in ("zh", "en"):
        clauses = parse(itn(transcript), lang)
        resolved = sum(1 for c in clauses
                       for v in (c["device"], c["state"], c["value"])
                       if v not in (None, "?")) + sum(1 for c in clauses if c["position"] != "-")
        key = (len(clauses), resolved)
        if best_key is None or key > best_key:
            best, best_key = clauses, key
    return [(c["device"], c["position"], c["state"], c["value"]) for c in best]


def grade_one(key, transcript):
    want = MIXED[key]
    got = fields_of(transcript)
    if not got:
        return {"device": False, "position": False, "state": False,
                "value": False, "clause": False, "got": []}
    wdev, wpos, wstate, wval = want.split("|")
    # One clause per clip in this fixture set, so the first clause is the one.
    gdev, gpos, gstate, gval = got[0]
    wval = None if wval == "-" else wval
    dev_ok = gdev == wdev
    pos_ok = dev_ok and (wpos == "-" or gpos == wpos)
    state_ok = dev_ok and gstate == wstate
    val_ok = dev_ok and (wval is None or str(gval) == str(wval))
    return {"device": dev_ok, "position": pos_ok, "state": state_ok,
            "value": val_ok, "clause": dev_ok and pos_ok and state_ok and val_ok,
            "got": got}


def report(label, rows):
    names = [k for k in MIXED if k not in NOCAB]
    totals = {"device": 0, "position": 0, "state": 0, "value": 0, "clause": 0}
    misses = []
    for key in names:
        if key not in rows:
            misses.append((key, "MISSING", []))
            continue
        res = grade_one(key, rows[key])
        for field in totals:
            totals[field] += int(res[field])
        if not res["clause"]:
            misses.append((key, rows[key], res["got"]))
    n = len(names)
    line = "  ".join("%s %3d/%d %5.1f%%" % (f, totals[f], n, 100.0 * totals[f] / n)
                     for f in ("device", "position", "state", "value", "clause"))
    print("%-8s %s" % (label, line))
    for key, text, got in misses:
        print("         miss %-8s want %-28s got %s  | %s"
              % (key, MIXED[key], got, text))
    return totals, n, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="<label>=<tsv>; repeatable for A/B")
    ap.add_argument("--baseline", default=None,
                    help="label to compare the others against (regression gate)")
    args = ap.parse_args()

    results = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(spec), spec
        results[label] = report(label, load(path, None))

    if args.baseline and args.baseline in results:
        base_tot, base_n, _ = results[args.baseline]
        print("\nregression vs %s:" % args.baseline)
        clean = True
        for label, (tot, n, _) in results.items():
            if label == args.baseline:
                continue
            for field in ("clause", "device", "position", "state", "value"):
                if tot[field] < base_tot[field]:
                    clean = False
                    print("  %s %s %d -> %d  REGRESSION"
                          % (label, field, base_tot[field], tot[field]))
        print("  clean" if clean else "  FAILED")
        return 0 if clean else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())