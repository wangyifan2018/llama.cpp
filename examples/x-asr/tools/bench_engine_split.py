#!/usr/bin/env python3
"""Four concurrent VAD segments, split across the HTP and the CPU engines.

The encoder batch probe shows that batching saturates: one chunk costs
`25.5 ms + 48.9 ms` per lane, so four lanes on the HTP are about 2.9 single
streams, not one. The HTP is a single engine, so four lanes there queue. The CPU
is a second engine with its own threads, and its per-chunk cost is close to the
HTP's, so the tail should fall when a lane moves off the HTP even though the
moved lane itself is slower.

This driver measures exactly that on one arrival pattern: whole segments
committed at the same instant, which is the worst case for `endpoint -> final`.

Each arm is one server command line. Rounds are interleaved and the arm order is
rotated per round, so a drift in device state cannot favour one arm.

Usage:
  python3 bench_engine_split.py --rounds 5
  python3 bench_engine_split.py --rounds 3 --arms htp4,htp3+1
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_load_matrix import Device, run, start_server, stop_server  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/Library/Caches/qwen3-asr-htp")


def htp_env(extra=""):
    return ("LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib "
            "GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 "
            "GGML_HEXAGON_OPPOLL=1 XASR_TIMING=1" + extra).strip()


def arm_specs(remote, port):
    """{name: (engine, argv, env, batch cap)}."""
    gguf = remote + "/x-asr-zh-en-q8_0.gguf"
    common = ["bin/llama-x-asr-server", gguf, "HTP0:0", "--port", str(port)]

    def arm(htp_streams, batch, cpu_streams=0, threads=2, tail=()):
        argv = common + ["--streams", str(htp_streams)]
        if batch:
            argv += ["--encoder-batch", str(batch)]
        if cpu_streams:
            argv += ["--cpu-streams", str(cpu_streams), "--threads", str(threads)]
        return ("gguf-htp", argv + list(tail), htp_env(), batch)

    return {
        # Reference: four lanes, one batched HTP step each.
        "htp4": arm(4, 4),
        # The same four slots without batching, to price batching end to end.
        "htp4-b1": arm(4, None),
        # One lane off the HTP. Batch cap 2 because only three HTP lanes remain
        # and a width of 4 could never form.
        "htp3+1": arm(3, 2, cpu_streams=1),
        # Two lanes each: the most even split the two engines allow.
        "htp2+2": arm(2, 2, cpu_streams=2),
        # The documented hybrid, letting the router decide where the fourth
        # request goes.
        "htp4+1": arm(4, 4, cpu_streams=1),
        # The shipped default deployment: four HTP slots, two CPU slots.
        "hybrid": arm(4, 4, cpu_streams=2),
        # 2+2 with four encoder threads per CPU slot: the CPU lanes are the
        # stragglers at two threads, and the phone has eight cores.
        "htp2+2-t4": arm(2, 2, cpu_streams=2, threads=4),
    }


def percentile(values, q):
    if not values:
        return float("nan")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def client(args, wavs, port, sessions):
    """One asr_client invocation; returns its JSON rows."""
    cmd = [sys.executable, os.path.join(HERE, "asr_client.py"),
           "--host", "127.0.0.1", "--port", str(port),
           "--sessions", str(sessions), "--wavs", ",".join(wavs),
           "--segment", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          stdin=subprocess.DEVNULL)
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    if not rows:
        raise RuntimeError("client produced no rows:\n%s%s" % (proc.stdout[-800:], proc.stderr[-800:]))
    # A session thread that dies leaves a null row behind with no timing field,
    # which is a transport failure, not a slow request.
    bad = [row for row in rows if "endpoint_to_final_ms" not in row]
    if bad:
        raise RuntimeError("session failed (%d of %d):\n%s%s" % (
            len(bad), len(rows), json.dumps(bad)[:600], proc.stderr[-600:]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default=os.environ.get(
        "ADB", os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")))
    parser.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL", ""))
    parser.add_argument("--remote", default=os.environ.get("XASR_REMOTE", "/data/local/tmp/llama.cpp"))
    parser.add_argument("--port", type=int, default=18770)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--arms", default="htp4,htp4-b1,htp3+1,htp2+2,htp4+1")
    parser.add_argument("--wavs", default=",".join(
        CACHE + "/xasr-multistream/vehicle-baseline.QMMIN2/tingting/" + name
        for name in ("hvac_en01.wav", "seat_zh03.wav", "win_zh01.wav", "light_en02.wav")))
    parser.add_argument("--sessions", type=int, default=4,
                        help="concurrent whole segments, one --wavs entry each")
    parser.add_argument("--warmup", default="",
                        help="clip for the one warm-up request per arm (default: first --wavs entry)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    wavs = args.wavs.split(",")
    if len(wavs) != args.sessions:
        parser.error("--wavs needs one clip per session (%d)" % args.sessions)
    for path in wavs:
        if not os.path.exists(path):
            parser.error("missing clip: %s" % path)
    warmup = args.warmup or wavs[0]

    dev = Device(args.adb, args.serial, args.remote)
    run([args.adb] + (["-s", args.serial] if args.serial else []) +
        ["forward", "tcp:%d" % args.port, "tcp:%d" % args.port], check=False)

    specs = arm_specs(args.remote, args.port)
    names = [n for n in args.arms.split(",") if n]
    for name in names:
        if name not in specs:
            parser.error("unknown arm %s (have %s)" % (name, ",".join(sorted(specs))))

    out_dir = args.out or time.strftime("~/Library/Caches/qwen3-asr-htp/xasr-split-%Y%m%d-%H%M%S")
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    config = {"git_rev": subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip(),
              "wavs": wavs, "rounds": args.rounds, "port": args.port,
              "sessions": args.sessions,
              "arms": {n: {"argv": specs[n][1], "env": specs[n][2]} for n in names}}
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    rows = {n: [] for n in names}
    logs = {n: [] for n in names}
    try:
        for rnd in range(args.rounds):
            order = names[rnd % len(names):] + names[:rnd % len(names)]
            for name in order:
                handle, text = start_server(dev, specs[name], "/data/local/tmp/split-%s.log" % name)
                try:
                    client(args, [warmup], args.port, 1)
                    got = client(args, wavs, args.port, args.sessions)
                    for row in got:
                        row["round"] = rnd + 1
                        row["arm"] = name
                    rows[name] += got
                    print("round %d %-8s %6.0f ms (max)" % (
                        rnd + 1, name, max(r["endpoint_to_final_ms"] for r in got)), flush=True)
                finally:
                    stop_server(dev, handle)
                    logs[name].append(dev.shell(
                        "cat /data/local/tmp/split-%s.log 2>/dev/null || true" % name,
                        check=False).stdout)
    finally:
        run([args.adb] + (["-s", args.serial] if args.serial else []) +
            ["forward", "--remove", "tcp:%d" % args.port], check=False)
        dev.kill_servers()

    with open(os.path.join(out_dir, "requests.jsonl"), "w") as fp:
        for name in names:
            for row in rows[name]:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in names:
        with open(os.path.join(out_dir, "server-%s.log" % name), "w") as fp:
            fp.write("\n".join(logs.get(name, [])))

    # Parity: every arm must return the same text for the same clip, otherwise a
    # latency difference is not a fair comparison.
    reference = {}
    for row in rows[names[0]]:
        reference[os.path.basename(row["wav"])] = row["text"]
    diffs = []
    for name in names:
        for row in rows[name]:
            base = os.path.basename(row["wav"])
            if reference.get(base) != row["text"]:
                diffs.append((name, base, reference.get(base), row["text"]))

    lines = ["| arm | request | p50 | p95 | p99 | max |", "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        values = [r["endpoint_to_final_ms"] for r in rows[name]]
        lines.append("| `%s` | %d | %.0f | %.0f | %.0f | %.0f |" % (
            name, len(values), percentile(values, 50), percentile(values, 95),
            percentile(values, 99), max(values)))
    lines.append("")
    for name in names:
        text = "\n".join(logs.get(name, []))
        batches = [line for line in text.splitlines() if "xasr-batch" in line]
        widths = {}
        for line in batches:
            w = line.split("size=")[1].split()[0]
            widths[w] = widths.get(w, 0) + 1
        routes = [line.split("device=")[1].split()[0]
                  for line in text.splitlines() if "xasr-route" in line and "device=" in line]
        placement = {}
        for dev_name in routes:
            placement[dev_name] = placement.get(dev_name, 0) + 1
        lines.append("- `%s`: batch steps %s, placements %s" % (
            name, widths or "-", placement or "-"))
    lines.append("")
    lines.append("transcript parity: %s" % (
        "identical across all arms" if not diffs else "%d mismatches" % len(diffs)))
    for item in diffs[:8]:
        lines.append("  - %s %s ref=%r got=%r" % item)

    text = "\n".join(lines)
    open(os.path.join(out_dir, "summary.md"), "w").write(text + "\n")
    print(text)
    print("\nraw: %s" % out_dir)
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
