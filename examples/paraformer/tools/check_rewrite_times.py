#!/usr/bin/env python3
"""Check Paraformer's growing and shrinking rewrite fixtures and optional long decode."""

import argparse
from pathlib import Path


def read_tsv(path):
    rows = {}
    for line in path.read_text().splitlines():
        wav, *fields, transcript = line.split("\t")
        meta = dict(field.split("=", 1) for field in fields)
        tokens = meta["tok"].split("|")
        times = list(map(int, meta["ts_cs"].split(",")))  # Legacy CLI column is in ms.
        assert len(tokens) == len(times) == int(meta["n_tokens"]), line
        rows[Path(wav).name] = (tokens, times, transcript, float(meta["audio_ms"]))
    return rows


def interval(tokens, spell):
    matches = [i for i in range(len(tokens)) if tokens[i:i + len(spell)] == spell]
    assert len(matches) == 1, (spell, tokens)
    return matches[0], matches[0] + len(spell)


def check_rewrite(baseline, biased, name, wrong, correct):
    raw_ids, raw_times, _, _ = baseline[name]
    ids, times, transcript, _ = biased[name]
    a, b = interval(raw_ids, wrong)
    c, d = interval(ids, correct)
    assert raw_ids[:a] == ids[:c] and raw_ids[b:] == ids[d:], transcript
    assert raw_times[:a] == times[:c] and raw_times[b:] == times[d:], (raw_times, times)
    assert raw_times[a] == times[c] and raw_times[b - 1] == times[d - 1], (raw_times, times)
    print(f"{name}: {len(wrong)} -> {len(correct)} pieces, source span {times[c]}-{times[d - 1]} ms, suffix unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("biased", type=Path)
    parser.add_argument("--long", type=Path)
    args = parser.parse_args()
    baseline, biased = read_tsv(args.baseline), read_tsv(args.biased)
    check_rewrite(baseline, biased, "light_en03.wav", ["li@@", "kes"], ["li@@", "gh@@", "ts"])
    check_rewrite(baseline, biased, "light_en04.wav", ["f@@", "oo@@", "t@@", "w@@", "all"], ["f@@", "oo@@", "t@@", "well"])
    for name, (_, times, _, duration) in biased.items():
        assert times == sorted(times) and 0 <= times[0] <= times[-1] <= duration, name
    if args.long:
        for name, (_, times, transcript, duration) in read_tsv(args.long).items():
            assert times == sorted(times) and 0 <= times[0] <= times[-1] <= duration, name
            assert times[-1] > 15000 and times[-1] > duration - 2500, (name, times[-1])
            assert transcript.count("ambient lights") >= 2 and transcript.count("footwell") >= 2, transcript
            print(f"{name}: monotonic windowed timestamps through {times[-1]} ms")


if __name__ == "__main__":
    main()
