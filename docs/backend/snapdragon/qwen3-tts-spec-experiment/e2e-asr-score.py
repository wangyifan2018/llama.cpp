#!/usr/bin/env python3
"""Score a local ASR transcript as an intelligibility proxy."""
import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata


def tokens(text, language):
    text = unicodedata.normalize("NFKC", text).casefold()
    if language == "english":
        text = re.sub(r"\b(?:0?7[.:]30)\b", "seven thirty", text)
        return re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return [ch for ch in text if ch.isalnum()]


def edit_distance(source, target):
    row = list(range(len(target) + 1))
    for i, x in enumerate(source, 1):
        prev = row
        row = [i]
        for j, y in enumerate(target, 1):
            row.append(min(prev[j] + 1, row[-1] + 1, prev[j - 1] + (x != y)))
    return row[-1]


parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
args = parser.parse_args()
dest = args.results
with (Path(__file__).with_name("e2e-prompts.tsv")).open() as stream:
    prompts = {row["id"]: row for row in csv.DictReader(stream, delimiter="\t")}
with (dest / "paired.csv").open() as stream:
    pairs = list(csv.DictReader(stream))
with (dest / "asr.jsonl").open() as stream:
    events = [json.loads(line) for line in stream if line.strip()]
    finished = {event["id"]: event for event in events if event.get("event") == "finish"}
if set(finished) != set(range(len(pairs))):
    raise ValueError("ASR did not finish every pair")
with (dest / "asr-score.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("sample", "mode", "language", "reference", "transcript", "edits", "units", "error_rate"))
    for idx, pair in enumerate(pairs):
        item = prompts[pair["sample"]]
        ref, hyp = item["text"], finished[idx]["text"].strip()
        a, b = tokens(ref, item["lang"]), tokens(hyp, item["lang"])
        edits = edit_distance(a, b)
        writer.writerow((pair["sample"], pair["mode"], item["lang"], ref, hyp, edits, len(a), edits / len(a)))
        print(pair["sample"], pair["mode"], f"{edits}/{len(a)}", hyp)
