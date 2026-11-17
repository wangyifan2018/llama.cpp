#!/usr/bin/env python3
"""Compare cabin phrase bias arms on the phone, round by round.

Each arm is one `llama-x-asr` invocation over every clip of every pack, in a
fixed order, with a warm-up file first so the first measured clip does not pay
for the cold graph. Arms alternate inside a round so a thermal or scheduling
drift lands on both, and `summary.md` reports the pooled field-level result
from `eval_vehicle`, not a single pass.

    python3 bench_vehicle.py --rounds 3 --arms off,on

The device-side layout is the one `VEHICLE-EVAL.md` describes:
`/data/local/tmp/llama.cpp/hotword-eval/<pack>/{manifest.tsv,*.wav}`.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_vehicle as ev  # noqa: E402
from bench_load_matrix import Device, run, sha256  # noqa: E402

ENV = ("LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib "
       "GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 "
       "GGML_HEXAGON_OPPOLL=1 XASR_TIMING=1")


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def build_clip_list(dev, remote, packs):
    """Fixed clip order plus a warm-up file that is never graded."""
    files = [remote + "/zh.wav"]
    ids = {}
    for pack in packs:
        listing = dev.shell("ls %s/hotword-eval/%s/*.wav | sort" % (remote, pack), check=False)
        paths = [p for p in listing.stdout.split() if p.endswith(".wav")]
        if not paths:
            raise SystemExit("pack %s is empty on the device" % pack)
        files.extend(paths)
        for path in paths:
            ids[os.path.basename(path)[:-4]] = pack
    return files, ids


def pull_manifest(dev, remote, pack, out):
    path = os.path.join(out, "%s-manifest.tsv" % pack)
    text = dev.shell("cat %s/hotword-eval/%s/manifest.tsv" % (remote, pack)).stdout
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default=os.environ.get(
        "ADB", os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")))
    parser.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL", ""))
    parser.add_argument("--remote", default="/data/local/tmp/llama.cpp")
    parser.add_argument("--model", default="x-asr-zh-en-q8_0.gguf")
    parser.add_argument("--device-backend", default="HTP0:0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--arms", default="off,on")
    parser.add_argument("--sets", default="tingting,holdout,controls,hardneg")
    parser.add_argument("--hotwords", default=os.path.join(HERE, "vehicle_hotwords.txt"))
    parser.add_argument("--semantics", default=os.path.join(HERE, "vehicle_semantics.tsv"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(
        os.path.expanduser("~/Library/Caches/qwen3-asr-htp"),
        "xasr-vehicle-%s" % time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)

    dev = Device(args.adb, args.serial, args.remote)
    remote_hotwords = args.remote + "/hotword-eval/vehicle_hotwords.txt"
    dev.shell("mkdir -p %s/hotword-eval" % args.remote)
    run([args.adb] + (["-s", args.serial] if args.serial else []) +
        ["push", args.hotwords, remote_hotwords])

    packs = [p for p in args.sets.split(",") if p]
    manifests = {pack: pull_manifest(dev, args.remote, pack, out) for pack in packs}
    files, pack_of = build_clip_list(dev, args.remote, packs)
    semantics = ev.load_semantics(args.semantics)

    hashes = dev.hash_remote([args.remote + "/" + args.model,
                              args.remote + "/bin/llama-x-asr", remote_hotwords])
    config = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_rev": run(["git", "rev-parse", "HEAD"], check=False).stdout.strip(),
        "git_dirty": [l[2:].strip() for l in run(
            ["git", "status", "--porcelain"], check=False).stdout.splitlines()],
        "device": {"serial": dev.prop("ro.serialno") or args.serial,
                   "soc": dev.prop("ro.soc.model")},
        "env": ENV,
        "rounds": args.rounds,
        "arms": args.arms.split(","),
        "packs": packs,
        "model": {"path": args.model, "sha256": hashes.get(args.model)},
        "binary": {"sha256": hashes.get("llama-x-asr")},
        "hotwords": {"path": remote_hotwords, "sha256": hashes.get("vehicle_hotwords.txt"),
                     "local_sha256": sha256(args.hotwords)},
        "clips": {os.path.basename(p)[:-4]: {"path": p, "pack": pack_of.get(os.path.basename(p)[:-4])}
                  for p in files[1:]},
    }
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=1, ensure_ascii=False)

    run_files = {}
    for rnd in range(1, args.rounds + 1):
        for arm in config["arms"]:
            log = "/data/local/tmp/xasr-vehicle-%s-r%d.log" % (arm, rnd)
            tsv = "/data/local/tmp/xasr-vehicle-%s-r%d.tsv" % (arm, rnd)
            extra = ""
            if arm == "on":
                extra = " --hotwords %s" % remote_hotwords
            script = ("cd %s && export %s && exec bin/llama-x-asr %s %s %s%s > %s 2> %s" % (
                args.remote, ENV, args.model, args.device_backend,
                " ".join(files), extra, tsv, log))
            started = time.time()
            run(dev.shell_cmd(script), timeout=1800)
            local = os.path.join(out, "%s-r%d.tsv" % (arm, rnd))
            text = dev.shell("cat %s" % tsv).stdout
            with open(local, "w", encoding="utf-8") as fh:
                fh.write(text)
            joiner = dev.shell("grep -c 'joiner.*7/7' %s || true" % log, check=False).stdout.strip()
            print("%s round %d: %d rows, %.0f s, joiner-on-htp lines %s" % (
                arm, rnd, len(text.splitlines()), time.time() - started, joiner), flush=True)
            run_files.setdefault(arm, []).append(local)

    # Grade every round, then pool. Clips keep their annotation and language
    # from the manifest that produced them.
    clips = {}
    for pack, manifest in manifests.items():
        for key, row in ev.load_manifest(manifest, pack=pack).items():
            row["pack"] = pack
            clips[key] = row

    report = {"arms": {}, "rows": {}}
    for arm, paths in run_files.items():
        rows = []
        for path in paths:
            transcripts = ev.load_transcripts(path)
            missing = [key for key in clips if key not in transcripts]
            if missing:
                raise SystemExit("%s: %d clips had no transcript, first is %s" % (
                    os.path.basename(path), len(missing), missing[0]))
            for key, clip in clips.items():
                ref = semantics.get(clip["id"])
                if ref is None:
                    raise SystemExit("no annotation for %s" % key)
                hyp = transcripts[key]["text"]
                graded = ev.grade(ref, ev.parse(hyp, clip["lang"]))
                rows.append({"id": clip["id"], "key": key, "pack": clip["pack"],
                             "lang": clip["lang"], "text": clip["text"], "hyp": hyp,
                             "grade": graded, "round": os.path.basename(path),
                             "wall_ms": transcripts[key]["meta"].get("wall_ms"),
                             "audio_ms": transcripts[key]["meta"].get("audio_ms"),
                             "value_total": sum(1 for c in ref if c["value"] is not None)})
        report["rows"][arm] = rows
        by_pack = {}
        for pack in packs:
            pack_rows = [r for r in rows if r["pack"] == pack]
            summary = ev.summarise(pack_rows)
            summary["value_total"] = sum(r["value_total"] for r in pack_rows)
            rtf = [r["wall_ms"] / r["audio_ms"] for r in pack_rows
                   if r.get("wall_ms") and r.get("audio_ms")]
            summary["rtf_median"] = percentile(rtf, 50)
            by_pack[pack] = summary
        report["arms"][arm] = by_pack

    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)

    # Transcript stability: a clip is stable when every round of every arm
    # produced the same string. Anything else means the pipeline is not
    # deterministic on this input, and per-clip deltas need a closer look.
    per_clip = {}
    for arm, rows in report["rows"].items():
        for row in rows:
            per_clip.setdefault(row["key"], {}).setdefault(arm, []).append(row["hyp"])
    unstable = [cid for cid, arms in per_clip.items()
                if len({h for hs in arms.values() for h in hs}) > 1]

    lines = ["# Cabin command bias, %s" % config["started_at"], "",
             "Device %s (%s), %d rounds, arms %s, packs %s." % (
                 config["device"]["serial"], config["device"]["soc"], args.rounds,
                 ", ".join(config["arms"]), ", ".join(packs)),
             "Binary %s, model %s, profile %s." % (
                 (config["binary"]["sha256"] or "?")[:12],
                 (config["model"]["sha256"] or "?")[:12],
                 (config["hotwords"]["sha256"] or "?")[:12]), "",
             "| arm | pack | clips | device | state | value | clause | command | wrong dir | unstated | false trig | near miss | zh CER | en WER |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, by_pack in report["arms"].items():
        for pack, s in by_pack.items():
            lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %d | %d | %d/%d | %d/%d | %.2f%% | %.2f%% |" % (
                arm, pack, s["clips"],
                ev.fmt(s["device"], s["total"]), ev.fmt(s["state"], s["total"]),
                ev.fmt(s["value"], s["value_total"]), ev.fmt(s["clause"], s["total"]),
                ev.fmt(s["command_ok"], s["command_clips"]),
                s["wrong_direction"], s["unstated"], s["false_trigger"], s["negative_clips"],
                s["near_miss"], s["negative_clips"],
                s["zh_cer"], s["en_wer"]))
    lines += ["", "| arm | pack | wall ms p50 | p95 | p99 | RTF median |",
              "|---|---|---:|---:|---:|---:|"]
    for arm, by_pack in report["arms"].items():
        for pack, s in by_pack.items():
            lat = s["latency"]
            lines.append("| %s | %s | %.0f | %.0f | %.0f | %.3f |" % (
                arm, pack, lat.get(50, 0), lat.get(95, 0), lat.get(99, 0), s["rtf_median"]))
    lines += ["", "Transcripts identical across all rounds and arms: %d/%d clips." % (
        len(per_clip) - len(unstable), len(per_clip))]
    if unstable:
        lines.append("Unstable: " + ", ".join(sorted(unstable)))

    lines += ["", "## Commands whose verdict differs between arms", ""]
    diffs = 0
    for key, clip in clips.items():
        verdict = {}
        for arm, rows in report["rows"].items():
            row = next((r for r in rows if r["key"] == key), None)
            if not row:
                continue
            g = row["grade"]
            verdict[arm] = (not g["triggered"]) if g["none"] else g["command"]
        if len(set(verdict.values())) > 1:
            diffs += 1
            sample = next(r for r in report["rows"][config["arms"][-1]] if r["key"] == key)
            lines.append("- %s %s ref=%s hyp=%s" % (key, verdict, sample["text"], sample["hyp"]))
    if not diffs:
        lines.append("None.")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
