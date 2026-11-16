#!/usr/bin/env python3
"""Run the X-ASR streaming benchmark matrix on one Android device.

The four configurations share the WAV files, the 20 ms packet pacing, the
1,200 ms tail padding, the greedy decoder and the text normalisation. See
examples/x-asr/BENCHMARK.md for the fixture hashes and the recorded results.
"""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time

CONFIGS = {
    "onnx-cpu":      {"backend": "onnx"},
    "gguf-cpu":      {"backend": "gguf", "device": "CPU", "batch": 2048, "poll": 0},
    "gguf-htp-wait": {"backend": "gguf", "device": "HTP0:0", "batch": 2048, "poll": 0},
    "gguf-htp-poll": {"backend": "gguf", "device": "HTP0:0", "batch": 2048, "poll": 1},
}


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, text=True, **kwargs)


ADB = "adb"


def adb_shell(serial, script, capture=True):
    cmd = [ADB]
    if serial:
        cmd += ["-s", serial]
    cmd += ["shell", "-T", script]
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def pct(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def parse_kv(line):
    out = {}
    for field in line.split("\t")[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            out[key] = value
    return out


def bench_gguf(args, cfg, remote_path):
    device = cfg["device"]
    env = (
        f"export LD_LIBRARY_PATH={remote_path}/lib ADSP_LIBRARY_PATH={remote_path}/lib && "
        f"export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH={cfg['batch']} "
        f"GGML_HEXAGON_OPPOLL={cfg['poll']} GGML_HEXAGON_POWER_MODE={args.power_mode} "
        f"GGML_HEXAGON_PROFILE=0 XASR_TIMING=1 && "
    )
    waves = " ".join(shlex.quote(w) for w in args.wavs)
    script = (
        f"cd {remote_path} && {env}"
        f"bin/llama-x-asr {shlex.quote(args.gguf)} {device} {waves} --realtime"
    )
    proc = adb_shell(args.serial, script)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    timing = [line[len("xasr-timing "):] for line in proc.stderr.splitlines() if line.startswith("xasr-timing ")]
    placement = [line for line in proc.stderr.splitlines() if "encoder " in line]
    init = [line for line in proc.stderr.splitlines() if line.startswith("xasr: init")]
    rows = []
    for index, line in enumerate(lines):
        row = {"wave": line.split("\t")[0]}
        row.update(parse_kv(line))
        row["text"] = line.split("\t")[-1]
        rows.append(row)
    if len(timing) != len(rows):
        raise SystemExit(f"{args.config}: {len(rows)} transcripts but {len(timing)} timing lines")
    for row, blob in zip(rows, timing):
        row.update(json.loads(blob))
    return rows, {"placement": placement, "init": init, "stderr": proc.stderr}


def bench_onnx(args, cfg, remote_path):
    waves = " ".join(shlex.quote(w) for w in args.wavs)
    script = (
        f"LD_LIBRARY_PATH={remote_path}/onnx/bin {remote_path}/onnx/bin/onnx-bench "
        f"{shlex.quote(args.onnx_dir)} {args.onnx_threads} 1 {waves}"
    )
    proc = adb_shell(args.serial, script)
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    init = [line for line in proc.stderr.splitlines() if line.startswith("{")]
    return rows, {"placement": [], "init": init, "stderr": proc.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default=os.environ.get("ADB", "adb"))
    parser.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL", ""))
    parser.add_argument("--remote", default="/data/local/tmp/llama.cpp")
    parser.add_argument("--gguf", default="x-asr-zh-en-q8_0.gguf")
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--onnx-threads", type=int, default=2)
    parser.add_argument("--wavs", nargs="+", required=True)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--power-mode", default="burst")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()
    global ADB
    ADB = args.adb
    for name in args.configs:
        if name not in CONFIGS:
            raise SystemExit(f"unknown config {name}")
    log_dir = args.log_dir or time.strftime("/tmp/xasr-ab-%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(log_dir, "stderr"), exist_ok=True)
    with open(os.path.join(log_dir, "commands.txt"), "w") as fh:
        fh.write(" ".join(shlex.quote(a) for a in sys.argv) + "\n")

    info = adb_shell(args.serial, "getprop ro.soc.model; getprop ro.build.version.release").stdout
    with open(os.path.join(log_dir, "device.txt"), "w") as fh:
        fh.write(info)
    raw = open(os.path.join(log_dir, "raw.jsonl"), "w")
    records = []
    for round_index in range(args.rounds):
        order = list(args.configs)
        if round_index % 2:
            order.reverse()
        for name in order:
            cfg = CONFIGS[name]
            if cfg["backend"] == "gguf":
                rows, extra = bench_gguf(args, cfg, args.remote)
            else:
                rows, extra = bench_onnx(args, cfg, args.remote)
            with open(os.path.join(log_dir, "stderr", f"{name}-r{round_index}.log"), "w") as fh:
                fh.write(extra["stderr"])
            for row in rows:
                row["config"] = name
                row["round"] = round_index
                records.append(row)
                raw.write(json.dumps(row) + "\n")
            print(f"round {round_index} {name}: {len(rows)} runs", flush=True)
    raw.close()

    summary = build_summary(records, args)
    with open(os.path.join(log_dir, "summary.md"), "w") as fh:
        fh.write(summary)
    print(summary)
    print(f"logs: {log_dir}")


def fixture_key(wave):
    return os.path.basename(wave)


def build_summary(records, args):
    by_config = {}
    for row in records:
        by_config.setdefault(row["config"], []).append(row)
    lines = ["# X-ASR benchmark summary", ""]
    lines.append(f"rounds={args.rounds} configs={','.join(args.configs)} threads: onnx={args.onnx_threads}")
    lines.append("")
    lines.append("## Endpoint to final")
    lines.append("")
    lines.append("| config | fixture | N | tail p50 | tail p95 | commit p50 | chunks | commit chunks | chunk p50 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for config in args.configs:
        rows = by_config.get(config, [])
        for wave in args.wavs:
            subset = [r for r in rows if fixture_key(r["wave"]) == fixture_key(wave)]
            if not subset:
                continue
            tail = [float(r.get("tail_ms", "nan")) for r in subset]
            commit = [float(r.get("commit_ms", r.get("flush_ms", "nan"))) for r in subset]
            chunks = [int(r.get("chunks", 0)) for r in subset]
            cchunks = [int(r.get("chunks_commit", 0)) for r in subset]
            chunk_all = [v for r in subset for v in r.get("chunk_ms", [])]
            lines.append(
                f"| {config} | {fixture_key(wave)} | {len(subset)} | {pct(tail,50):.1f} | {pct(tail,95):.1f} | "
                f"{pct(commit,50):.1f} | {statistics.median(chunks):.0f} | {statistics.median(cchunks):.0f} | "
                f"{pct(chunk_all,50):.1f} |"
            )
    lines.append("")
    lines.append("## Normalised cost per encoder chunk and per second of audio")
    lines.append("")
    lines.append("Summed over all runs of a configuration, so long clips carry their own weight.")
    lines.append("")
    lines.append("| config | chunk ms | greedy ms/frame | fbank ms/s | submit ms/batch | wait ms/batch | cpu ms/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for config in args.configs:
        rows = by_config.get(config, [])
        if not rows:
            continue
        def total(key):
            return sum(float(r[key]) for r in rows if key in r and r[key] not in (None, ""))
        chunks = sum(int(r.get("chunks", 0)) for r in rows)
        frames = chunks * 12
        audio_s = sum(float(r.get("audio_ms", 0)) for r in rows) / 1000.0
        dsp_batches = total("dsp_batches")
        greedy = f"{total('greedy_ms') / max(frames, 1):.3f}" if "greedy_ms" in rows[0] else "n/a"
        fbank = f"{total('fbank_ms') / max(audio_s, 1e-9):.2f}" if "fbank_ms" in rows[0] else "n/a"
        submit = f"{total('submit_ms') / dsp_batches:.2f}" if dsp_batches else "n/a"
        wait = f"{total('wait_ms') / dsp_batches:.2f}" if dsp_batches else "n/a"
        cpu = f"{total('cpu_ms') / max(audio_s, 1e-9):.0f}" if "cpu_ms" in rows[0] else "n/a"
        lines.append(
            f"| {config} | {total('compute_ms') / max(chunks, 1):.1f} | {greedy} | {fbank} | "
            f"{submit} | {wait} | {cpu} |"
        )
    lines.append("")
    lines.append("## Stage breakdown, median over all runs")
    lines.append("")
    lines.append("| config | compute | submit | wait | batches | greedy/onnx decode | fbank | feat up | mask build | mask up | readback | detok | cpu_ms | rss |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for config in args.configs:
        rows = by_config.get(config, [])
        if not rows:
            continue
        def med(key):
            values = [float(r[key]) for r in rows if key in r and r[key] not in (None, "")]
            return statistics.median(values) if values else float("nan")
        lines.append(
            f"| {config} | {med('compute_ms'):.1f} | {med('submit_ms'):.1f} | {med('wait_ms'):.1f} | "
            f"{med('dsp_batches'):.1f} | {med('greedy_ms'):.1f} | {med('fbank_ms'):.1f} | "
            f"{med('feat_up_ms'):.2f} | {med('mask_build_ms'):.2f} | {med('mask_up_ms'):.2f} | "
            f"{med('readback_ms'):.2f} | {med('detok_ms'):.2f} | {med('cpu_ms'):.0f} | {med('rss_kib'):.0f} |"
        )
    lines.append("")
    lines.append("## Transcripts")
    lines.append("")
    reference = {}
    for row in records:
        key = fixture_key(row["wave"])
        reference.setdefault(key, set()).add(row.get("text", ""))
    for key, texts in reference.items():
        lines.append(f"- {key}: " + " | ".join(sorted(texts)))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
