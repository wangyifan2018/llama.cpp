#!/usr/bin/env python3
"""Synthesize the cabin fixture set in a new voice with the macOS voice engine.

Each output clip is 16 kHz mono 16-bit PCM, which is what the service accepts.
The text set is the bundled cabin_commands.tsv, the same one used for the
existing voices, so a new voice stays comparable across a regression.

    python3 gen_voice_fixtures.py --voice Eddy --out /tmp/voice-eddy

A voice name may carry several language variants. `say -v Eddy` resolves to the
English variant, which renders Mandarin as 5 KB of silence and still exits 0,
so the script picks the variant per row from the language column and refuses an
empty result instead of writing it out.

Voices are TTS: they add speaker variety, not channel variety. There is no far
field, no road noise and no accent, so results measure how much of the accuracy
comes from the speaker rather than the acoustics. Real recordings are still
required before any release threshold.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import wave


def read_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise SystemExit(f"bad manifest row: {line!r}")
        rows.append((fields[0], fields[1], fields[2]))
    return rows


def list_voices():
    """Map (base name, language tag) to the exact name `say` needs."""
    out = subprocess.run(["say", "-v", "?"], check=True, capture_output=True, text=True).stdout
    voices = {}
    for line in out.splitlines():
        m = re.match(r"^(.*?)\s{2,}([a-z]{2}_[A-Z]{2})\s", line)
        if not m:
            continue
        voices[(m.group(1).strip(), m.group(2))] = m.group(1).strip()
    return voices


def pick_voice(name, lang):
    """Resolve one voice name to the variant that speaks `lang`.

    Names with several variants appear as "Eddy (中文（中国大陆）)" and match on
    their prefix; plain names match exactly.
    """
    out = subprocess.run(["say", "-v", "?"], check=True, capture_output=True, text=True).stdout
    want = {"zh": "zh_CN", "en": "en_US"}.get(lang)
    if not want:
        raise SystemExit(f"unsupported language column: {lang!r}")
    exact, prefixed = [], []
    for line in out.splitlines():
        m = re.match(r"^(.*?)\s{2,}([a-z]{2}_[A-Z]{2})\s", line)
        if not m:
            continue
        full, tag = m.group(1).strip(), m.group(2)
        if tag != want:
            continue
        if full == name:
            exact.append(full)
        elif full.startswith(name + " ("):
            prefixed.append(full)
    if exact:
        return exact[0]
    if prefixed:
        return prefixed[0]
    # A single-variant voice (Tingting is zh_CN only) still has to speak the
    # other language column; let it try rather than failing the whole set, and
    # let the silence check below catch it if it cannot.
    for line in out.splitlines():
        m = re.match(r"^(.*?)\s{2,}([a-z]{2}_[A-Z]{2})\s", line)
        if not m:
            continue
        full = m.group(1).strip()
        if full == name or full.startswith(name + " ("):
            return full
    return None


def trim_to_pcm16(source, target, lead_pad=0.15, gate_ratio=0.01):
    """Rewrite a rendered clip as trimmed 16 kHz mono PCM16."""
    with wave.open(str(source), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"unexpected render format in {source}: "
                             f"{w.getframerate()} Hz, {w.getnchannels()} ch, {w.getsampwidth() * 8} bit")
        frames = w.readframes(w.getnframes())

    samples = memoryview(frames).cast("h")
    if not samples:
        return 0.0

    # say leaves a long lead-in and tail. Trim both against the clip peak so
    # clip lengths stay close to the recorded set and nothing starts late.
    peak = max(abs(s) for s in samples)
    gate = peak * gate_ratio
    start = 0
    while start < len(samples) and abs(samples[start]) < gate:
        start += 1
    end = len(samples)
    while end > start and abs(samples[end - 1]) < gate:
        end -= 1
    if end <= start:
        return 0.0
    pad = int(lead_pad * 16000)
    start = max(0, start - pad)
    end = min(len(samples), end + pad)

    with wave.open(str(target), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(memoryview(frames)[start * 2:end * 2].tobytes())
    return (end - start) / 16000.0


def render(voice, text, target, rate):
    raw = target.with_suffix(".raw.wav")
    cmd = ["say", "-v", voice, "-o", str(raw), "--data-format=LEF32@16000"]
    if rate:
        cmd += ["-r", str(rate)]
    cmd.append(text)
    subprocess.run(cmd, check=True, capture_output=True)
    try:
        # say writes IEEE float in a RIFF container, which the service rejects
        # and the wave module cannot read back, so convert through CoreAudio.
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        str(raw), str(target)], check=True, capture_output=True)
        return trim_to_pcm16(target, target)
    finally:
        raw.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="base voice name from `say -v '?'`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="tab separated name/language/text; defaults to the bundled set")
    ap.add_argument("--rate", type=int, default=0, help="words per minute, 0 keeps the voice default")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N rows")
    args = ap.parse_args()

    if args.manifest:
        rows = read_manifest(args.manifest)
    else:
        rows = read_manifest(Path(__file__).resolve().parent / "cabin_commands.tsv")
    if args.limit:
        rows = rows[:args.limit]

    resolved = {}
    for lang in sorted({lang for _, lang, _ in rows}):
        voice = pick_voice(args.voice, lang)
        if not voice:
            raise SystemExit(f"voice {args.voice!r} has no variant for {lang!r}")
        resolved[lang] = voice
        print(f"language {lang} -> {voice}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    durations = []
    for name, lang, text in rows:
        seconds = render(resolved[lang], text, args.out / name, args.rate)
        if seconds <= 0.0:
            raise SystemExit(f"{resolved[lang]!r} produced silence for {name}; "
                             f"the voice probably cannot speak {lang!r}")
        durations.append(seconds)
        print(f"{name}\t{lang}\t{seconds:.2f}s\t{text}")

    (args.out / "manifest.tsv").write_text(
        "".join(f"{n}\t{l}\t{t}\n" for n, l, t in rows), encoding="utf-8")
    mid = sorted(durations)[len(durations) // 2]
    print(f"\n{len(durations)} clips, {sum(durations):.1f}s total, median {mid:.2f}s -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()