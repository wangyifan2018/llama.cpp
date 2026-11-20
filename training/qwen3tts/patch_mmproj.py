"""Apply a tensor patch to an existing mmproj GGUF in place (byte offsets, no re-serialisation).

The patch only replaces tensor payloads with identical element counts, so all metadata, alignment and
offsets stay valid. Every tensor is cast to the dtype the file already declares for it (f16 or f32),
which is what the runtime expects.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from gguf import GGUFReader, GGMLQuantizationType

DTYPE = {
    GGMLQuantizationType.F16: np.float16,
    GGMLQuantizationType.F32: np.float32,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="input mmproj gguf")
    ap.add_argument("patch", help="safetensors patch (f32 tensors, gguf names)")
    ap.add_argument("dst", help="output mmproj gguf")
    ap.add_argument("--verify", action="store_true", help="re-read dst and compare payloads")
    args = ap.parse_args()

    shutil.copyfile(args.src, args.dst)

    reader = GGUFReader(args.src)
    by_name = {t.name: t for t in reader.tensors}

    applied, skipped = 0, []
    with safe_open(args.patch, framework="pt") as f, open(args.dst, "r+b") as out:
        for name in f.keys():
            t = by_name.get(name)
            if t is None:
                skipped.append(name)
                continue
            arr = f.get_tensor(name).numpy().astype(np.float32)
            if arr.size != t.n_elements:
                print(f"{name}: element count {arr.size} != {t.n_elements}", file=sys.stderr)
                return 1
            dtype = DTYPE.get(t.tensor_type)
            if dtype is None:
                print(f"{name}: unsupported type {t.tensor_type}", file=sys.stderr)
                return 1
            buf = np.ascontiguousarray(arr.reshape(-1).astype(dtype)).tobytes()
            if len(buf) != t.n_bytes:
                print(f"{name}: {len(buf)} bytes != {t.n_bytes}", file=sys.stderr)
                return 1
            out.seek(t.data_offset)
            out.write(buf)
            applied += 1

    print(f"patched {applied} tensors into {args.dst}")
    if skipped:
        print(f"skipped {len(skipped)} tensors not in the gguf: {skipped[:4]}{' ...' if len(skipped) > 4 else ''}")

    if args.verify:
        chk = GGUFReader(args.dst)
        chk_by = {t.name: t for t in chk.tensors}
        bad = 0
        with safe_open(args.patch, framework="pt") as f:
            for name in f.keys():
                t = chk_by.get(name)
                if t is None:
                    continue
                want = f.get_tensor(name).numpy().astype(np.float32).reshape(-1)
                got = np.asarray(t.data, dtype=np.float32).reshape(-1)
                if want.size != got.size or not np.allclose(want, got, atol=1e-3):
                    bad += 1
                    print(f"verify failed: {name}", file=sys.stderr)
        print("verify ok" if bad == 0 else f"verify FAILED ({bad})")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
