"""End-to-end smoke test of the T1 pipeline: synthetic data -> train -> export -> patch.

Runs on CPU in about a minute. It does not test quality (the data is synthetic); it tests that the
format, the model shapes, the training loop, the metric and the GGUF patch all line up.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import datafmt


def make_dataset(out: Path, n: int, seed: int = 0) -> None:
    """hidden ~ N(0,1); the 15 codebooks are argmax of a fixed random projection of hidden,
    i.e. exactly the structure the model can represent, so the loss must fall."""
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((n, datafmt.N_EMBD)).astype(np.float32)
    codes = np.zeros((n, datafmt.N_CODES), dtype=np.int32)
    for g in range(15):
        w = rng.standard_normal((datafmt.CODE_VOCAB, datafmt.N_EMBD)).astype(np.float32) / np.sqrt(datafmt.N_EMBD)
        codes[:, g + 1] = (hidden @ w.T).argmax(-1)
    # 20% label noise so it is not a trivially perfect fit
    mask = rng.random((n, 15)) < 0.2
    codes[:, 1:][mask] = rng.integers(0, datafmt.CODE_VOCAB, mask.sum(), dtype=np.int32)
    codes[:, 0] = 0

    out.mkdir(parents=True, exist_ok=True)
    cut = int(n * 0.85)
    datafmt.write_t1(out / "train_0000.npz", hidden[:cut].astype(np.float16), codes[:cut])
    datafmt.write_t1(out / "eval_0000.npz", hidden[cut:].astype(np.float16), codes[cut:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--check-patch", action="store_true", help="also round-trip a real mmproj if present")
    ap.add_argument("--overfit", action="store_true", help="tiny set, many steps: asserts the loss actually falls")
    ap.add_argument("--kd", action="store_true", help="also run the KL distillation path")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="qwen3tts_smoke_"))
    print(f"workdir {tmp}")
    make_dataset(tmp / "data", args.n)
    print(f"synthetic shards: {args.n} frames, 15 books, 2048 entries")

    cmd = [sys.executable, str(HERE / "train_predictor.py"),
           "--shards", str(tmp / "data" / "train_*.npz"),
           "--eval-shards", str(tmp / "data" / "eval_*.npz"),
           "--out", str(tmp / "pred.safetensors"),
           "--s0", "--max-steps", str(args.steps), "--batch", str(args.batch), "--device", "cpu",
           "--epochs", str(args.epochs)] + (["--kd"] if args.kd else [])
    print("running:", " ".join(cmd[1:]))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr.strip(), file=sys.stderr)
        return 1

    losses = [float(l.split()[3]) for l in res.stdout.splitlines() if l.startswith("epoch")]
    if not losses:
        print("no epochs reported", file=sys.stderr)
        return 1
    if not all(np.isfinite(x) for x in losses):
        print(f"FAIL: non finite loss {losses}", file=sys.stderr)
        return 1
    print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f} over {len(losses)} epoch(s)")
    if args.overfit and losses[-1] >= losses[0] - 0.1:
        print("FAIL: the loop did not fit the tiny set", file=sys.stderr)
        return 1

    res = subprocess.run([sys.executable, str(HERE / "export_patch.py"),
                          str(tmp / "pred.safetensors"), str(tmp / "patch.safetensors")],
                         capture_output=True, text=True)
    print(res.stdout.strip().splitlines()[0] if res.stdout.strip() else res.stderr.strip())
    if res.returncode != 0:
        return 1

    if args.check_patch:
        src = Path("/tmp/q3tts-models/mmproj-qwen3-tts-0.6b-f16.gguf")
        if not src.exists():
            print(f"skip patch round-trip: {src} not found")
        else:
            dst = tmp / "patched.gguf"
            import os
            gguf_py = str(HERE.parent.parent / "gguf-py")  # the repo's vendored gguf reader
            env = {**os.environ, "PYTHONPATH": gguf_py + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
            r2 = subprocess.run([sys.executable, str(HERE / "patch_mmproj.py"), str(src),
                                 str(tmp / "patch.safetensors"), str(dst), "--verify"],
                                capture_output=True, text=True, env=env)
            print(r2.stdout.strip() or r2.stderr.strip())
            if r2.returncode != 0:
                return 1

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
