"""S0 gate and T1 training for the 1-pass code predictor.

Two modes:

  --s0     small subset, few epochs, only reports the gate metric (codes top-1 agreement).
           This is the 8 GPU-hour experiment that decides which variant to build.
  default  full training on the shards, saves the best checkpoint by agreement.

Data comes from prepare_pairs.py (see datafmt.py for the layout). Nothing here needs the reference
model, so it runs on any machine with torch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import datafmt
from model_predictor import MTPStylePredictor, agreement, book_entropy, per_book_loss
from model_ar_teacher import ARTeacher, kd_loss

GATE_AGREEMENT = 0.60   # >= this: full parallel (A). 0.50-0.60: grouped (B). < 0.50: distill (C)


def save_checkpoint(model, path: str, meta: dict) -> None:
    """safetensors for the weights, a json sidecar for the metrics: no torch zip writer involved."""
    from safetensors.torch import save_file
    tensors = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, path)
    Path(path + ".json").write_text(json.dumps({"config": {"layers": 5, "n_embd": 1024, "books": 15, "vocab": 2048},
                                                **meta}, indent=2))


def batches(shards: list[Path], batch: int, shuffle: bool, seed: int = 0):
    """Yield (hidden, codes) batches, streaming one shard at a time so 200 h fits in RAM."""
    order = list(range(len(shards)))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(order)
    for si in order:
        hidden, codes = datafmt.read_t1(shards[si])
        n = len(hidden)
        idx = rng.permutation(n) if shuffle else np.arange(n)
        for off in range(0, n - batch + 1, batch):
            sel = idx[off:off + batch]
            yield (torch.from_numpy(hidden[sel].astype(np.float32)),
                   torch.from_numpy(codes[sel].astype(np.int64)))


@torch.no_grad()
def evaluate(model, shards, device, batch=512, max_batches=40) -> dict:
    model.eval()
    acc, ent, n = np.zeros(15), np.zeros(15), 0
    for i, (h, c) in enumerate(batches(shards, batch, shuffle=False)):
        if i >= max_batches:
            break
        h, c = h.to(device), c.to(device)
        logits = model(h)
        acc += np.array(agreement(logits, c))
        ent += np.array(book_entropy(logits))
        n += 1
    model.train()
    if n == 0:
        return {"agreement": 0.0, "per_book": [0.0] * 15, "entropy": [0.0] * 15}
    return {"agreement": float((acc / n).mean()), "per_book": (acc / n).tolist(), "entropy": (ent / n).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob for training t1 shards")
    ap.add_argument("--eval-shards", default=None, help="glob for held-out t1 shards (default: 5%% of --shards)")
    ap.add_argument("--init", default=None, help="AR checkpoint (.safetensors) to start from")
    ap.add_argument("--out", default="predictor_1pass.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0 = full)")
    ap.add_argument("--s0", action="store_true", help="gate mode: tiny run, report agreement only")
    ap.add_argument("--kd", action="store_true", help="add KL self-distillation from the AR predictor")
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    shards = datafmt.list_shards(args.shards)
    shards = [p for p in shards if p.suffix == ".npz"]
    if not shards:
        print(f"no shards match {args.shards}", file=sys.stderr)
        return 1
    if args.eval_shards:
        eval_shards = [p for p in datafmt.list_shards(args.eval_shards) if p.suffix == ".npz"]
    else:
        cut = max(1, len(shards) // 20)
        eval_shards, shards = shards[-cut:], shards[:-cut] or shards[-cut:]

    if args.s0:
        args.epochs, args.batch = min(args.epochs, 10), min(args.batch, 256)
        args.max_steps = args.max_steps or 400

    sd = None
    if args.init:
        from safetensors.torch import load_file
        sd = load_file(args.init)
        print(f"initialized from {args.init}")

    model = MTPStylePredictor.from_ar_state_dict(sd) if sd is not None else MTPStylePredictor()
    model.to(args.device)

    teacher = None
    if args.kd:
        # the teacher is the original 15 step chain: same weights, frozen, used only for soft targets
        teacher = ARTeacher.from_ar_state_dict(sd) if sd is not None else ARTeacher()
        teacher.to(args.device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        n_t = sum(p.numel() for p in teacher.parameters())
        print(f"kd: teacher {n_t/1e6:.1f} M params, T={args.kd_temp}, weight={args.kd_weight}")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par/1e6:.1f} M params, device {args.device}, {len(shards)} train shards, {len(eval_shards)} eval shards")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = max(1, 200_000 // args.batch)  # informational only; the loop is shard driven
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total_steps, 1))

    best = -1.0
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        run_loss, run_kd, nb = 0.0, 0.0, 0
        for h, c in batches(shards, args.batch, shuffle=True, seed=epoch):
            h, c = h.to(args.device), c.to(args.device)
            logits = model(h)
            ce = per_book_loss(logits, c)
            loss = ce
            kdl = None
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(h, c)
                kdl = kd_loss(logits, t_logits, args.kd_temp)
                loss = ce + args.kd_weight * kdl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total_steps - 1:
                sched.step()
            run_loss += ce.item()
            if kdl is not None:
                run_kd += kdl.item()
            nb += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        ev = evaluate(model, eval_shards, args.device)
        dt = time.time() - t0
        kd_str = f"  kd {run_kd/max(nb,1):.3f}" if teacher is not None else ""
        print(f"epoch {epoch:2d}  loss {run_loss/max(nb,1):.3f}{kd_str}  agreement {ev['agreement']*100:5.1f}%  "
              f"({dt:6.1f}s, {step} steps)")
        if ev["agreement"] > best:
            best = ev["agreement"]
            save_checkpoint(model, args.out, {"metrics": ev, "agreement": ev["agreement"], "epoch": epoch})
        if args.max_steps and step >= args.max_steps:
            break

    ev = evaluate(model, eval_shards, args.device)
    print(f"final agreement {ev['agreement']*100:.1f}%   per-book " +
          " ".join(f"{x*100:.0f}" for x in ev["per_book"]))
    print(f"saved {args.out} (best {best*100:.1f}%)")

    if args.s0:
        a = max(ev["agreement"], best)
        verdict = "A (full parallel)" if a >= GATE_AGREEMENT else ("B (grouped 8+7)" if a >= 0.50 else "C (distill, keep AR)")
        print(f"S0 GATE: agreement {a*100:.1f}% -> build {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
