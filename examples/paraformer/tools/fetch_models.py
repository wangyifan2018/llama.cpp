#!/usr/bin/env python3
"""Fetch and verify the Paraformer GGUFs this example is measured against.

The numbers in README.md and PRODUCTION.md are only meaningful for the exact
files listed in `model_manifest.txt`. A different conversion of the same source
checkpoint changes the logit gaps slightly, and `hotwords_vehicle.txt` was tuned
against those gaps, so an unverified weight file can make a correct profile look
broken. This script therefore refuses to leave an unverified file in place.

    python3 tools/fetch_models.py --out ~/models/paraformer          # all
    python3 tools/fetch_models.py --out DIR --only paraformer-zh-q8_0.gguf
    python3 tools/fetch_models.py --out DIR --verify-only            # no network

Download order is deliberate: the shipping Q8_0 file is first, so a run that is
interrupted still leaves a usable model.

If the download fails, that is reported as a failure and not as "skipped".
The upstream repository name comes from the earlier rounds of this port and has
not been re-checked from this machine, so treat a 404 as information about the
repository, not as a transient error: the checksum below is the authoritative
contract either way, and any mirror serving that exact file is acceptable.
"""

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "model_manifest.txt"

# Only the shipping checkpoint has a stable upstream name recorded in the
# earlier rounds. The others are derived: the en Q8_0 comes from the F32 file
# through tools/requant.cpp, and the Q4_K arm is a requant of the zh Q8_0.
BASE_URL = os.environ.get("PF_MODEL_BASE_URL", "https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main")


def load_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            raise SystemExit("bad manifest row: %r" % line)
        sha, size, name = fields[0], int(fields[1]), fields[2]
        rows.append({"sha256": sha, "bytes": size, "name": name,
                     "note": " ".join(fields[3:])})
    return rows


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify(path, want_sha, want_size):
    """Return (ok, message). Size is checked first because it is cheap."""
    if not path.exists():
        return False, "missing"
    got_size = path.stat().st_size
    if got_size != want_size:
        return False, "size %d, expected %d" % (got_size, want_size)
    got = sha256_file(path)
    if got != want_sha:
        return False, "sha256 %s..., expected %s..." % (got[:12], want_sha[:12])
    return True, "ok"


def download(url, dest, want_size):
    """Fetch to a .part file, then rename. Never leaves a partial in place."""
    part = dest.with_suffix(dest.suffix + ".part")
    print("  fetching %s" % url)
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_models.py"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
        total = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            total += len(block)
            if want_size:
                pct = 100.0 * total / want_size
                sys.stdout.write("\r  %5.1f%%  %d / %d MiB" % (pct, total >> 20, want_size >> 20))
                sys.stdout.flush()
    sys.stdout.write("\n")
    part.replace(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory holding the GGUF files")
    ap.add_argument("--only", action="append", default=None,
                    help="fetch just these file names (repeatable)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check what is already there; never touches the network")
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(MANIFEST)
    if args.only:
        wanted = set(args.only)
        rows = [r for r in rows if r["name"] in wanted]
        missing = wanted - {r["name"] for r in rows}
        if missing:
            raise SystemExit("not in manifest: %s" % ", ".join(sorted(missing)))

    failures = []
    for row in rows:
        dest = out / row["name"]
        ok, why = verify(dest, row["sha256"], row["bytes"])
        if ok:
            print("%-28s  ok" % row["name"])
            continue
        if args.verify_only:
            print("%-28s  %s" % (row["name"], why))
            failures.append(row["name"])
            continue
        print("%-28s  %s, fetching" % (row["name"], why))
        url = "%s/%s" % (BASE_URL.rstrip("/"), row["name"])
        try:
            download(url, dest, row["bytes"])
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            # Delete a failed download: an unsigned file in the model directory
            # is worse than no file, because the next run would use it.
            print("  failed: %s" % exc)
            failures.append(row["name"])
            continue
        ok, why = verify(dest, row["sha256"], row["bytes"])
        if not ok:
            print("  downloaded file failed verification: %s" % why)
            dest.unlink(missing_ok=True)
            failures.append(row["name"])
            continue
        print("%-28s  ok after fetch" % row["name"])

    if failures:
        print("\n%d file(s) not available: %s" % (len(failures), ", ".join(failures)))
        print("The checksum in tools/model_manifest.txt is the contract, not the URL:")
        print("any copy with matching sha256 works. Point PF_MODEL_BASE_URL at a mirror,")
        print("or place the files in %s yourself and re-run with --verify-only." % out)
        return 1
    print("\nall %d file(s) verified in %s" % (len(rows), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())