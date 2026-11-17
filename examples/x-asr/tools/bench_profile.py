#!/usr/bin/env python3
"""Per-stage and per-operator budget for one llama-x-asr run on the phone.

Two modes:

  timing  (default) reads the `xasr-timing` line each stream prints, so the
          stages inside one chunk are separated: feature extraction, upload,
          mask, encoder submit, encoder wait, readback, greedy, detokenise.
  ops     also sets GGML_HEXAGON_PROFILE=1, pulls the per-operator log and ranks
          operators by accumulated DSP microseconds, including a fixed-versus-
          per-element fit.

The per-operator numbers are DSP-side wall time for the operator body only. They
exclude the host submit and the batch boundaries, so their sum is slightly below
compute_ms; compare shares, not absolute totals.

Usage:
  python3 bench_profile.py --wav en-20.wav
  python3 bench_profile.py --wav en-20.wav --mode ops --device CPU
  python3 bench_profile.py --wav en-20.wav --mode ops --reps 2 --out /tmp/prof
"""

import argparse
import json
import os
import re
import subprocess
import sys

ADB = os.environ.get("ADB", os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"))
REMOTE = "/data/local/tmp/llama.cpp"
MODEL = "x-asr-zh-en-q8_0.gguf"

DEVICE_ENVS = {
    "HTP": ("GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 "
            "GGML_HEXAGON_OPPOLL=1"),
    "CPU": "",
}


def sh(argv, check=True, timeout=1800):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL, check=check)


def device_env(device, profile):
    base = "LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib XASR_TIMING=1"
    extra = DEVICE_ENVS["HTP"] if device.startswith("HTP") else DEVICE_ENVS["CPU"]
    if profile:
        extra = (extra + " GGML_HEXAGON_PROFILE=1 GGML_LOG_LEVEL=debug").strip()
    return (base + " " + extra).strip()


def run_stream(wav, device, profile, log_path, extra_env):
    log = "/data/local/tmp/xasr-prof.log"
    cmd = ("cd %s && export %s %s && bin/llama-x-asr %s %s %s > %s 2>&1"
           % (REMOTE, device_env(device, profile), extra_env, MODEL, device, wav, log))
    sh([ADB, "shell", "-T", cmd])
    return log


def parse_timing(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("xasr-timing "):
            continue
        rows.append(json.loads(line[len("xasr-timing "):]))
    return rows


def parse_wav_lines(text):
    out = []
    for line in text.splitlines():
        if ".wav\t" in line:
            out.append(line.rstrip())
    return out


def pct(values, q):
    if not values:
        return float("nan")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


STAGES = [
    ("fbank_ms", "fbank (host)"),
    ("feat_up_ms", "feature upload"),
    ("mask_build_ms", "mask build"),
    ("mask_up_ms", "mask upload"),
    ("submit_ms", "encoder submit"),
    ("wait_ms", "encoder wait"),
    ("readback_ms", "readback"),
    ("greedy_ms", "greedy (predictor+joiner)"),
    ("detok_ms", "detokenise"),
]


def timing_table(rows, label):
    """One row per stream: stage totals, and the same divided by chunk count."""
    out = []
    out.append("| Stage | ms | ms/chunk | share of compute_ms |")
    out.append("|---|---:|---:|---:|")
    for key, name in STAGES:
        total = sum(r.get(key, 0.0) for r in rows)
        chunks = sum(r.get("chunks", 0) for r in rows) or 1
        compute = sum(r.get("compute_ms", 0.0) for r in rows) or 1.0
        if key in ("fbank_ms", "feat_up_ms", "mask_build_ms", "mask_up_ms", "readback_ms", "greedy_ms", "detok_ms"):
            # These are outside compute_ms; express them against the same base so
            # the column stays comparable.
            share = 100.0 * total / compute
        else:
            share = 100.0 * total / compute
        out.append("| %s | %.1f | %.2f | %.1f%% |" % (name, total, total / chunks, share))
    chunks = sum(r.get("chunks", 0) for r in rows)
    compute = sum(r.get("compute_ms", 0.0) for r in rows)
    append = sum(r.get("append_wall_ms", 0.0) for r in rows)
    out.append("")
    out.append("`%s`: %d chunks over %d stream(s), compute %.1f ms (%.2f ms/chunk), "
               "append wall %.1f ms, DSP batches %d (%.2f per chunk), pad frames %d."
               % (label, chunks, len(rows), compute, compute / max(chunks, 1), append,
                  sum(r.get("dsp_batches", 0) for r in rows),
                  sum(r.get("dsp_batches", 0) for r in rows) / max(chunks, 1),
                  sum(r.get("pad_frames", 0) for r in rows)))
    return out


def parse_ops(path):
    """Return (per-op totals, per-op (elements, usec) samples, per-batch sizes)."""
    agg = {}
    samples = {}
    batches = []
    for line in open(path, errors="replace"):
        if "profile-op" not in line:
            continue
        if "OPBATCH" in line:
            m = re.search(r"n-ops (\d+).*usec (\d+)", line)
            if m:
                batches.append((int(m.group(1)), int(m.group(2))))
            continue
        parts = line.split("profile-op ", 1)[1].split("|")
        if len(parts) < 3:
            continue
        m = re.search(r"usec (\d+)", line)
        if not m:
            continue
        name = parts[0].strip()
        usec = int(m.group(1))
        rec = agg.setdefault(name, [0, 0])
        rec[0] += usec
        rec[1] += 1
        try:
            out = parts[2].split("->")[-1].strip()
            n = 1
            for d in out.split(":"):
                n *= int(d)
            samples.setdefault(name, []).append((n, usec))
        except ValueError:
            pass
    return agg, samples, batches


def fit_fixed(samples):
    """Least squares usec = fixed + per_element * n. Returns (fixed_us, ns/elem)."""
    if len(samples) < 12:
        return None
    nn = len(samples)
    sx = sum(p[0] for p in samples)
    sy = sum(p[1] for p in samples)
    sxx = sum(p[0] * p[0] for p in samples)
    sxy = sum(p[0] * p[1] for p in samples)
    den = nn * sxx - sx * sx
    if den == 0:
        return None
    b = (nn * sxy - sx * sy) / den
    a = (sy - b * sx) / nn
    return a, b * 1e3


def op_table(path, chunks):
    agg, samples, batches = parse_ops(path)
    total = sum(v[0] for v in agg.values())
    out = []
    out.append("| Operator | Calls | ms | ms/chunk | µs/call | fixed µs | ns/elem |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, (usec, count) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:18]:
        fit = fit_fixed(samples.get(name, []))
        fixed = "%6.1f" % fit[0] if fit else "-"
        slope = "%6.2f" % fit[1] if fit else "-"
        out.append("| `%s` | %d | %.1f | %.2f | %.1f | %s | %s |"
                   % (name, count, usec / 1000.0, usec / 1000.0 / max(chunks, 1),
                      usec / count, fixed, slope))
    if batches:
        widths = {}
        for n, _ in batches:
            widths[n] = widths.get(n, 0) + 1
        out.append("")
        out.append("DSP batches: %d, widths %s, batch wall %.1f ms."
                   % (len(batches), ", ".join("%d x %d ops" % (c, n) for n, c in sorted(widths.items())),
                      sum(u for _, u in batches) / 1000.0))
    out.append("")
    out.append("Operator time sums to %.1f ms across %d chunks (%.2f ms/chunk); the "
               "fixed column is the per-call cost that does not scale with tensor size, "
               "so it is the floor a graph with this many operators pays." %
               (total / 1000.0, chunks, total / 1000.0 / max(chunks, 1)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", default="en-20.wav", help="fixture inside the device directory")
    p.add_argument("--device", default="HTP0:0")
    p.add_argument("--mode", choices=["timing", "ops"], default="timing")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--env", default="", help="extra device environment, e.g. XASR_CACHE_FOLD=0")
    p.add_argument("--out", default=None, help="directory for the raw log and markdown")
    args = p.parse_args()

    if not os.path.exists(ADB):
        sys.exit("adb not found at %s (set ADB)" % ADB)

    out_dir = args.out or "/tmp/xasr-profile"
    os.makedirs(out_dir, exist_ok=True)
    profile = args.mode == "ops"

    timing_rows = []
    wav_lines = []
    log_remote = None
    for rep in range(args.reps):
        log_remote = run_stream(args.wav, args.device, profile, None, args.env)
        text = sh([ADB, "shell", "-T", "cat %s" % log_remote]).stdout
        timing_rows += parse_timing(text)
        wav_lines += parse_wav_lines(text)
        print("rep %d: %d stream(s)" % (rep, len(parse_timing(text))), file=sys.stderr)

    chunks = sum(r.get("chunks", 0) for r in timing_rows)
    lines = []
    lines.append("# x-asr profile: `%s` on %s" % (args.wav, args.device))
    lines.append("")
    lines.append("%d stream(s), %d chunks total. `env` extra: `%s`." % (len(timing_rows), chunks, args.env or "-"))
    lines.append("")
    lines.append("## Per-stage budget")
    lines.append("")
    lines += timing_table(timing_rows, args.wav)
    lines.append("")

    if profile:
        local = os.path.join(out_dir, "ops.log")
        sh([ADB, "pull", log_remote, local])
        lines.append("## Per-operator budget")
        lines.append("")
        lines += op_table(local, chunks)
        lines.append("")

    if wav_lines:
        lines.append("## Wall and RTF")
        lines.append("")
        lines.append("| Run | audio ms | wall ms | flush ms | RTF |")
        lines.append("|---|---:|---:|---:|---:|")
        for line in wav_lines:
            fields = dict(re.findall(r"(\w+)=([0-9.]+)", line))
            lines.append("| `%s` | %s | %s | %s | %s |" % (
                line.split("\t")[0], fields.get("audio_ms", "?"), fields.get("wall_ms", "?"),
                fields.get("flush_ms", "?"), fields.get("rtf", "?")))
        lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(out_dir, "profile.md"), "w") as fh:
        fh.write(text)
    print("written to %s" % os.path.join(out_dir, "profile.md"), file=sys.stderr)


if __name__ == "__main__":
    main()
