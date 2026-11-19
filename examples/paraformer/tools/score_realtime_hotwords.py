#!/usr/bin/env python3
"""Score required and forbidden terms in eval_realtime.py JSONL output.

Annotations are TSV with a header: wav, group, required, forbidden. List
multiple phrases with |. Include all active phrases as forbidden on unrelated
hard negatives when checking whether the bias can invent an entity.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    text = " ".join(text.split())
    return re.sub(r"(?<=[\u3400-\u9fff]) (?=[\u3400-\u9fff])", "", text)


def pattern(phrase):
    text = normalize(phrase)
    if not text:
        raise ValueError(f"empty phrase {phrase!r}")
    left = r"(?<![a-z0-9])" if re.fullmatch(r"[a-z0-9]", text[0]) else ""
    right = r"(?![a-z0-9])" if re.fullmatch(r"[a-z0-9]", text[-1]) else ""
    return re.compile(left + re.escape(text) + right)


def load_runs(path):
    clips = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        name = Path(row["wav"]).name
        if name in clips or "error" in row:
            raise ValueError(f"duplicate or failed clip {name}")
        clips[name] = normalize(row["text"])
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    clips = load_runs(args.run)
    groups = defaultdict(lambda: {"expected": 0, "hit": 0, "clips": 0, "false_trigger_clips": 0})
    grades = {}
    with args.annotations.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            name, group = row["wav"], row["group"]
            if name in grades:
                raise ValueError(f"duplicate annotation {name}")
            text = clips.pop(name)
            required = [term for term in row["required"].split("|") if term]
            forbidden = [term for term in row["forbidden"].split("|") if term]
            if not required and not forbidden:
                raise ValueError(f"no terms annotated for {name}")
            missing = [term for term in required if not pattern(term).search(text)]
            invented = [term for term in forbidden if pattern(term).search(text)]
            summary = groups[group]
            summary["expected"] += len(required)
            summary["hit"] += len(required) - len(missing)
            summary["clips"] += 1
            summary["false_trigger_clips"] += bool(invented)
            grades[name] = {"group": group, "missing": missing, "invented": invented}
    if clips or not grades:
        raise ValueError(f"unannotated results: {', '.join(clips)}")
    report = {"groups": groups, "clips": grades}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
