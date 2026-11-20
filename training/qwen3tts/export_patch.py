"""Map a trained 1-pass predictor checkpoint to GGUF tensor names.

The runtime reads the same names the converter writes, so the only job here is renaming plus the
transpose-free layout note: ggml stores ne[0] fastest, which equals a C-contiguous torch tensor whose
last dim is the fastest -- i.e. torch [out, in] and [books, vocab, embd] map byte for byte.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# (gguf name suffix, state dict key), per layer
LAYER_MAP = [
    ("ln1",       "ln1.weight"),
    ("ln2",       "ln2.weight"),
    ("attn_v",    "v_proj.weight"),
    ("attn_out",  "o_proj.weight"),
    ("ffn_gate",  "gate.weight"),
    ("ffn_up",    "up.weight"),
    ("ffn_down",  "down.weight"),
]

TOP_MAP = [
    ("a.gen.code.proj_in.weight",     "proj_in.weight"),
    ("a.gen.code.proj_in.bias",       "proj_in.bias"),
    ("a.gen.code.output_norm.weight", "norm.weight"),
    ("a.gen.code.head.weight",        "heads"),
]


def build_patch(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, key in TOP_MAP:
        if key not in state:  # e.g. proj_in on the 0.6B checkpoint
            continue
        out[name] = state[key].detach().float().contiguous()
    # layers: discover the count from the keys instead of hardcoding 5
    n_layer = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("layers."))
    for i in range(n_layer):
        for gguf_suffix, key in LAYER_MAP:
            out[f"a.gen.code.blk.{i}.{gguf_suffix}.weight"] = \
                state[f"layers.{i}.{key}"].detach().float().contiguous()
    return out


def main() -> int:
    ckpt, out = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("predictor_1pass.pt", "predictor_patch.safetensors")
    # sniff the container instead of trusting the extension: zip header = torch, otherwise safetensors
    with open(ckpt, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        blob = torch.load(ckpt, map_location="cpu")
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    else:
        from safetensors.torch import load_file
        state = load_file(ckpt)
    patch = build_patch(state)
    save_file(patch, out)
    total = sum(t.numel() for t in patch.values())
    print(f"wrote {out}: {len(patch)} tensors, {total/1e6:.1f} M params")
    for k in list(patch)[:4]:
        print(f"  {k:42s} {tuple(patch[k].shape)}")
    print("  ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
