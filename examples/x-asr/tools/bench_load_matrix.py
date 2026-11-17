#!/usr/bin/env python3
"""One command for the same-arrival comparison of ONNX CPU against GGUF HTP.

Starts one server per arm and round on the phone, drives the same arrival
scenarios through an ADB-forwarded port, records one JSON record per request,
then writes `summary.md` with latency percentiles, encoder batch widths, graph
placement and transcript parity.

    python3 examples/x-asr/tools/bench_load_matrix.py --rounds 3

Everything the run depended on is captured in `config.json`: git revision,
device, input hashes, binary hashes and each arm's exact command line, so a
number in the summary can be traced back to one build.

    --arms onnx-cpu,htp-b1,htp-b2,htp-b4
    --htp-args "--cpu-streams 2 --eta-router"   # one-factor experiment on every
                                                # HTP arm, reflected in the name
"""

import argparse
import json
import os
import subprocess
import sys
import time
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = ("for n in llama-x-asr-server onnx-server; do p=$(pidof $n); "
         "[ -n \"$p\" ] && kill $p; done")


def sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, check=True, timeout=300):
    # stdin must be closed: `adb shell` forwards stdin and would otherwise wait
    # for the caller's stdin to reach EOF before returning.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
    if check and proc.returncode != 0:
        raise RuntimeError("%s failed (%d): %s%s" % (
            " ".join(cmd), proc.returncode, proc.stdout, proc.stderr))
    return proc


class Device(object):
    def __init__(self, adb, serial, remote):
        self.adb = adb
        self.serial = serial
        self.remote = remote

    def shell_cmd(self, script):
        cmd = [self.adb]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd + ["shell", "-T", script]

    def shell(self, script, check=True, timeout=300):
        return run(self.shell_cmd(script), check=check, timeout=timeout)

    def prop(self, name):
        return self.shell("getprop %s" % name, check=False).stdout.strip()

    def hash_remote(self, paths):
        out = self.shell("sha256sum " + " ".join(paths), check=False).stdout
        return {os.path.basename(line.split()[1]): line.split()[0]
                for line in out.splitlines() if len(line.split()) == 2}

    def kill_servers(self):
        self.shell(GUARD + "; sleep 1", check=False)


def arm_specs(args):
    """Return {name: (engine, server argv, env dict, batch cap)}."""
    remote = args.remote
    port = str(args.port)
    gguf = remote + "/x-asr-zh-en-q8_0.gguf"
    common = ["--port", port]
    extra = args.htp_args.split()
    htp_env = ("LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib "
               "GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 "
               "GGML_HEXAGON_OPPOLL=1 XASR_TIMING=1")

    def htp(batch, tail):
        argv = ["bin/llama-x-asr-server", gguf, "HTP0:0"] + common + ["--streams", "4"]
        if batch:
            argv += ["--encoder-batch", str(batch)]
        return ("gguf-htp", argv + tail, htp_env, batch)

    specs = {
        "onnx-cpu": ("onnx", ["onnx/bin/onnx-server", remote + "/onnx/model",
                              str(args.onnx_threads)] + common + ["--streams", "4"],
                     "LD_LIBRARY_PATH=$PWD/onnx/bin ONNX_TIMING=1", None),
        "htp-b1": htp(None, extra),
        "htp-b2": htp(2, extra),
        "htp-b4": htp(4, extra),
        "htp-hybrid": ("gguf-htp", ["bin/llama-x-asr-server", gguf, "HTP0:0"] + common +
                       ["--streams", "2", "--cpu-streams", "2"] + extra,
                       htp_env, None),
        "htp-b4-hybrid": ("gguf-htp", ["bin/llama-x-asr-server", gguf, "HTP0:0"] + common +
                          ["--streams", "2", "--cpu-streams", "2", "--encoder-batch", "4"] +
                          extra, htp_env, 4),
    }
    return specs


def start_server(dev, spec, log_name, wait_s=180):
    """Start the server and return (handle, log text once it is listening).

    The adb channel is kept open on purpose: the server replaces the remote
    shell with `exec`, so closing this handle is what stops it. Backgrounding on
    the device (`nohup ... &`) is unreliable here, the shell does not always
    return while the child holds the channel.
    """
    engine, argv, env, _ = spec
    script = "cd %s && export %s && exec %s > %s 2>&1" % (
        dev.remote, env, " ".join(argv), log_name)
    handle = subprocess.Popen(dev.shell_cmd(script), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + wait_s
    text = ""
    while time.time() < deadline:
        time.sleep(1.0)
        text = dev.shell("cat %s 2>/dev/null || true" % log_name, check=False).stdout
        if "listening on" in text:
            return handle, text
        if "cannot" in text or "failed to start" in text:
            handle.terminate()
            raise RuntimeError("server failed to start:\n" + text[-2000:])
    handle.terminate()
    raise RuntimeError("server did not report listening within %d s:\n%s" % (wait_s, text[-2000:]))


def stop_server(dev, handle):
    if handle is not None:
        handle.terminate()
        try:
            handle.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.kill()
    dev.kill_servers()
    deadline = time.time() + 10
    while time.time() < deadline:
        if not dev.shell("pidof llama-x-asr-server onnx-server", check=False).stdout.strip():
            return
        dev.kill_servers()
    raise RuntimeError("server still running after stop")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default=os.environ.get(
        "ADB", os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")))
    parser.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL", ""))
    parser.add_argument("--remote", default=os.environ.get("XASR_REMOTE", "/data/local/tmp/llama.cpp"))
    parser.add_argument("--cache", default=os.path.expanduser("~/Library/Caches/qwen3-asr-htp"))
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--arms", default="onnx-cpu,htp-b1,htp-b2,htp-b4")
    parser.add_argument("--scenarios", default="single,aligned4,stagger4,mixed4,burst4,vadburst")
    parser.add_argument("--htp-args", default="", help="extra server arguments for every HTP arm")
    parser.add_argument("--onnx-threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1, help="discarded single-stream runs per arm")
    parser.add_argument("--interleave", action="store_true",
                        help="run round by round across arms instead of arm by arm")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-parity", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.adb):
        sys.exit("adb not found: %s (use --adb)" % args.adb)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or "/tmp/xasr-load-%s" % run_id
    os.makedirs(out, exist_ok=True)

    fixtures = {
        "zh.wav": os.path.join(args.cache, "zh.wav"),
        "en.wav": os.path.join(args.cache, "en.wav"),
        "zh-10.wav": os.path.join(args.cache, "cabin-asr-probe/long-perf/zh-10.wav"),
        "en-20.wav": os.path.join(args.cache, "cabin-asr-probe/long-perf/en-20.wav"),
    }
    for name, path in fixtures.items():
        if not os.path.exists(path):
            sys.exit("missing fixture %s" % path)

    dev = Device(args.adb, args.serial, args.remote)
    specs = arm_specs(args)
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        if arm not in specs:
            sys.exit("unknown arm %s; known: %s" % (arm, ", ".join(sorted(specs))))

    # Preflight: the device must hold the exact model the summary quotes.
    remote_hashes = dev.hash_remote(
        [args.remote + "/x-asr-zh-en-q8_0.gguf", args.remote + "/bin/llama-x-asr-server",
         args.remote + "/onnx/bin/onnx-server"])
    model_hash = remote_hashes.get("x-asr-zh-en-q8_0.gguf")
    if not model_hash:
        sys.exit("model missing under %s" % args.remote)
    if model_hash != "1ca120084a1517cf02d96e44cdd9a9544f0c887f6d0149c9f85d151be6833a61":
        print("WARNING: model hash %s differs from the documented Q8_0 file" % model_hash)

    git_rev = run(["git", "rev-parse", "HEAD"], check=False).stdout.strip()
    status = run(["git", "status", "--porcelain"], check=False).stdout.strip()

    config = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_rev": git_rev,
        "git_dirty": [l[2:].strip() for l in status.splitlines()] if status else [],
        "device": {"serial": dev.prop("ro.serialno") or args.serial,
                   "soc": dev.prop("ro.soc.model"),
                   "android": dev.prop("ro.build.version.release")},
        "remote": args.remote,
        "port": args.port,
        "rounds": args.rounds,
        "interleaved": args.interleave,
        "scenarios": args.scenarios.split(","),
        "model": {"path": args.remote + "/x-asr-zh-en-q8_0.gguf", "sha256": model_hash},
        "fixtures": {name: {"path": path, "sha256": sha256(path)}
                     for name, path in fixtures.items()},
        "binaries": {"llama-x-asr-server": {"sha256": remote_hashes.get("llama-x-asr-server", "?")},
                     "onnx-server": {"sha256": remote_hashes.get("onnx-server", "?")}},
        "arms": [],
    }

    for arm in arms:
        engine, argv, env, batch = specs[arm]
        config["arms"].append({"name": arm, "engine": engine, "argv": " ".join(argv),
                               "env": env, "batch_cap": batch})
        stale = os.path.join(out, arm + ".jsonl")
        if os.path.exists(stale):
            os.remove(stale)

    handle = None
    try:
        stop_server(dev, None)
        run([args.adb] + (["-s", args.serial] if args.serial else []) +
            ["forward", "tcp:%d" % args.port, "tcp:%d" % args.port], check=False)

        def one_round(arm, rnd, warmup):
            log_name = "/data/local/tmp/xasr-arm-%s-r%d.log" % (arm, rnd)
            handle, header = start_server(dev, specs[arm], log_name)
            if warmup:
                print("\n".join(l for l in header.splitlines()
                                if "listening" in l or "engine " in l or "threads," in l), flush=True)
                for _ in range(warmup):
                    run([sys.executable, os.path.join(HERE, "asr_client.py"),
                         "--port", str(args.port), "--sessions", "1", "--reps", "1",
                         "--wav", fixtures["zh.wav"], "--paced", "--json"], check=False)
            measured = run([sys.executable, os.path.join(HERE, "bench_arrival.py"),
                            "--port", str(args.port), "--label", arm, "--reps", "1",
                            "--round-base", str(rnd), "--append",
                            "--short", fixtures["zh.wav"], "--short-en", fixtures["en.wav"],
                            "--long", fixtures["zh-10.wav"], "--long-en", fixtures["en-20.wav"],
                            "--scenarios", args.scenarios, "--out", os.path.join(out, arm + ".jsonl")])
            print(measured.stdout.rstrip(), flush=True)
            with open(os.path.join(out, "%s-r%d.log" % (arm, rnd)), "w") as fh:
                fh.write(dev.shell("cat %s" % log_name, check=False).stdout)
            return handle

        # Interleaved: round by round across arms, so device drift hits every arm.
        if args.interleave:
            order = [(arm, rnd, rnd == 0) for rnd in range(args.rounds) for arm in arms]
        else:
            order = [(arm, rnd, rnd == 0) for arm in arms for rnd in range(args.rounds)]
        for i, (arm, rnd, warm) in enumerate(order):
            print("=== %s round %d ===" % (arm, rnd), flush=True)
            handle = one_round(arm, rnd, args.warmup if warm else 0)
            stop_server(dev, handle)
            handle = None
    finally:
        stop_server(dev, handle)

    # One log per arm for the summary: concatenate that arm's rounds in order.
    for arm in arms:
        parts = []
        for rnd in range(args.rounds):
            path = os.path.join(out, "%s-r%d.log" % (arm, rnd))
            if os.path.exists(path):
                parts.append(open(path).read())
                os.remove(path)
        with open(os.path.join(out, arm + ".log"), "w") as fh:
            fh.write("".join(parts))
        run([args.adb] + (["-s", args.serial] if args.serial else []) +
            ["forward", "--remove", "tcp:%d" % args.port], check=False)

    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    print("\nrun directory:", out)
    if not args.no_parity:
        run([sys.executable, os.path.join(HERE, "bench_parity.py"), "--dir", out], check=False)


if __name__ == "__main__":
    main()
