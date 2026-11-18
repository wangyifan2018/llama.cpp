#!/usr/bin/env python3
"""Build long-audio fixtures and score a server's handling of them.

The cabin fixtures are 2-4 s clips. Every accuracy gate in this repo runs on
those, and none of them can see what happens past roughly 15 s - which is where
this model starts getting worse. This script makes the long audio, states the
expected transcript from the per-clip references, and scores whatever came back.

Two failure modes it exists to pin down, both reproducible on the host CPU and
both invisible to the short-clip gates:

1. **Repetition after ~17 s of continuous speech.** Concatenating different
   commands with 300 ms of silence between them is transcribed exactly up to
   5 clips (13.6 s) and then gains spurious syllables at fixed positions. The
   errors do not move as more audio is appended, so it is not drift that
   accumulates - the same early boundary is wrong in the 16.8, 19.5 and 21.4 s
   versions. The encoder attends in both directions, so adding tail audio
   changes the decision made about an earlier boundary.
2. **Hallucination on trailing silence.** 10.9 s of speech followed by 5 s of
   digital silence emits "没有没有没有" after the real transcript. At 4 s of
   silence it is clean. Real recordings that end on a hold or a closed mic have
   plenty of that, and a shorter clip does not: 2.3 s of speech plus 5 s of
   silence is fine, so the amount of speech already in the graph matters too.

Usage:

    # build the fixtures and print the expected transcripts
    python3 tools/gen_long_fixtures.py --model MODEL.gguf --out /tmp/pf-long

    # score a directory of transcripts, one line per file, "<name>\\t<text>"
    python3 tools/gen_long_fixtures.py --model MODEL.gguf --out /tmp/pf-long --score /tmp/pf-long/out.tsv

The scorer compares against the per-clip reference the model produces for each
source clip *in isolation*. Comparing against a hand-written sentence would
confound "long audio broke it" with "this model never got that word right",
which is the whole question.
"""

import argparse
import glob
import os
import wave

# Eight distinct cabin commands. Distinct on purpose: repeating one command
# hides a dropout, because the neighbouring copy supplies the missing syllables
# and the joined text still looks right.
SEQUENCE = [
    "hvac_zh01", "light_zh03", "seat_zh02", "win_zh04",
    "hvac_zh02", "light_zh04", "seat_zh03", "win_zh02",
]

GAP_S = 0.3          # silence between concatenated commands
STEPS = list(range(2, 9))   # how many commands in each fixture


def read_pcm(path):
    with wave.open(path) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 16000:
            raise SystemExit("%s is not 16 kHz mono PCM16" % path)
        return w.readframes(w.getnframes())


def write_pcm(path, data):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(data)


def silence(seconds):
    return b"\x00\x00" * int(16000 * seconds)


def build(fixdir, names, model_refs):
    """Write the staircase, the silence-tail case and the seam case.

    Returns (gated, recorded): gated fixtures must match exactly; recorded ones
    document behaviour that is known to be wrong, so they are reported but do
    not fail the run. Expected text for the gated set is the model's own
    per-clip references, so any mismatch is attributable to length alone.
    """
    os.makedirs(fixdir, exist_ok=True)
    expected, recognised = {}, {}
    gap = silence(GAP_S)

    pieces = []
    for n in names:
        raw = read_pcm(n)
        pieces.append(raw)
        # Per-clip reference with the same trailing gap the staircase uses, so
        # the reference and the fixture differ only in what is attached after.
        write_pcm(os.path.join(fixdir, "pc_%s.wav" % os.path.basename(n)[:-4]), raw + gap)

    for k in STEPS:
        data = b"".join(p + gap for p in pieces[:k])
        name = "step%02d" % k
        write_pcm(os.path.join(fixdir, name + ".wav"), data)
        expected[name] = "".join(model_refs[i] for i in range(k))
        print("%-8s %5.2f s  %d command(s)" % (name, len(data) / 2 / 16000, k))

    # Trailing silence: the same speech, 0-6 s of digital silence attached.
    base = b"".join(p + gap for p in pieces[:4])
    for s in (0, 1, 2, 3, 4, 5, 6):
        name = "tail%d" % s
        write_pcm(os.path.join(fixdir, name + ".wav"), base + silence(s))
        expected[name] = "".join(model_refs[i] for i in range(4))
        print("%-8s %5.2f s  speech + %d s silence" % (name, len(base) / 2 / 16000 + s, s))

    # Seam: split the first 5 commands at a boundary with 0.5 s of context on
    # each side. This is recorded, not gated: cutting 0.5 s either side of a
    # silence midpoint still duplicates the syllables that straddle it ("把副"
    # at the end of the first half, "一把" at the start of the second), because
    # the encoder attends both ways and each half sees a different continuation
    # of the same word. A splitter needs more context than this, or it needs to
    # cut where the CIF has already closed a token, and neither is implemented
    # yet - the point of keeping the case is that a future splitter has to move
    # these two rows or explain why it did not.
    full = b"".join(p + gap for p in pieces[:5])
    boundary = len(b"".join(p + gap for p in pieces[:4]))
    ctx = int(16000 * 0.5) * 2
    write_pcm(os.path.join(fixdir, "seam_a.wav"), full[:boundary + ctx])
    write_pcm(os.path.join(fixdir, "seam_b.wav"), full[boundary - ctx:])
    recognised["seam_a"] = "把主驾空调调到二十一度别关氛围灯把顶灯关掉关闭主驾座椅加热把副驾驶车窗降到三分之一把副"
    recognised["seam_b"] = "一把副驾空调调到二十二度"
    print("%-8s %5.2f s  (first half, 0.5 s context)" % ("seam_a", (boundary + ctx) / 2 / 16000))
    print("%-8s %5.2f s  (second half, 0.5 s context)" % ("seam_b", (len(full) - boundary + ctx) / 2 / 16000))

    return expected, recognised


def transcribe(binary, model, paths):
    """Run the CLI over a list of WAVs, returning {basename: text}.

    One process for all files: init is ~82 ms and would otherwise dominate the
    wall clock of a 2 s clip, which is not what this script measures.
    """
    import subprocess
    out = subprocess.run([binary, model] + paths + ["--device", "CPU", "--threads", "4", "--quiet"],
                         capture_output=True, text=True)
    text = {}
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].endswith(".wav"):
            text[os.path.basename(parts[0])] = parts[-1]
    return text


def score(expected, actual):
    """Exact string match, plus a character-level diff for near misses.

    Exact rather than CER: the failures being tracked are one duplicated or one
    dropped syllable inside an otherwise perfect transcript, which CER reports
    as 4 percent and a reader skims past. A boolean per fixture does not have
    that problem.
    """
    rows = []
    for name in sorted(expected):
        want = expected[name]
        got = actual.get(name + ".wav", "")
        rows.append((name, want == got, want, got))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="paraformer GGUF")
    ap.add_argument("--clips", default="", help="directory of 16 kHz mono clips (default: beside the model)")
    ap.add_argument("--binary", default="", help="llama-paraformer binary")
    ap.add_argument("--out", required=True, help="fixture output directory")
    ap.add_argument("--score", default="", help="score an existing transcripts TSV instead of building")
    ap.add_argument("--gate", action="store_true", help="exit nonzero if any fixture is wrong")
    args = ap.parse_args()

    model = os.path.abspath(args.model)
    clips = args.clips or os.path.join(os.path.dirname(model), "tingting")
    names = [os.path.join(clips, n + ".wav") for n in SEQUENCE]
    missing = [n for n in names if not os.path.exists(n)]
    if missing:
        raise SystemExit("missing clips: %s" % ", ".join(os.path.basename(m) for m in missing))

    binary = args.binary or os.path.join(os.path.dirname(model), "..", "bin", "llama-paraformer")
    if not os.path.exists(binary):
        import shutil
        binary = shutil.which("llama-paraformer") or binary

    # References come from the model on each clip alone. Without this the
    # comparison would mix two questions together.
    print("== per-clip references")
    refs = transcribe(binary, model, [n for n in names])
    ordered = []
    for n in names:
        base = os.path.basename(n)
        ordered.append(refs.get(base, ""))
        print("  %-14s %s" % (base, refs.get(base, "<<missing>>")))
    if not all(ordered):
        raise SystemExit("the model produced no transcript for at least one clip")

    print()
    print("== fixtures")
    expected, recorded = build(args.out, names, ordered)

    print()
    print("== transcribe")
    paths = sorted(p for p in glob.glob(os.path.join(args.out, "*.wav")))
    actual = transcribe(binary, model, paths)

    print()
    print("== result")
    rows = score(expected, actual)
    bad = 0
    for name, ok, want, got in rows:
        if ok:
            print("  ok   %-8s" % name)
            continue
        bad += 1
        print("  FAIL %-8s" % name)
        print("       want: %s" % want)
        print("       got : %s" % got)

    print()
    print("== recorded (not gated)")
    for name in sorted(recorded):
        got = actual.get(name + ".wav", "")
        same = got == recorded[name]
        print("  %-8s %s" % (name, "as expected" if same else "CHANGED: %s" % got))

    if args.score:
        with open(args.score, "w") as f:
            for name, _, _, got in rows:
                f.write("%s\t%s\n" % (name, got))
        print("\nwrote %s" % args.score)

    print("\n%d/%d gated fixtures match" % (len(rows) - bad, len(rows)))
    if args.gate and bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())