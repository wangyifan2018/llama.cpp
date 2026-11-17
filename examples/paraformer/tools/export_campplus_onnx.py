#!/usr/bin/env python3
"""Load the ModelScope CAM++ checkpoint and export it to ONNX.

The cabin front end can then run the speaker embedding on the CPU (or any
onnxruntime provider) while the ASR model stays on HTP: a hybrid placement that
costs the ASR tail nothing.

Deps: funasr + torch (funasr 1.4.9 is the last version that works on Python 3.9).

    python3 export_campplus_onnx.py --bin campplus_cn_common.bin --out campplus.onnx
"""

import argparse
import inspect
import os
import sys

import numpy as np
import torch

from funasr.models.campplus.model import CAMPPlus


def build(bin_path):
    model = CAMPPlus(
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
    )
    state = torch.load(bin_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("load_state_dict: %d missing, %d unexpected" % (len(missing), len(unexpected)))
    if missing:
        print("  first missing:", missing[:3])
    if unexpected:
        print("  first unexpected:", unexpected[:3])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=300, help="dummy T for the export graph")
    args = ap.parse_args()

    model = build(args.bin)
    # funasr's CAMPPlus takes (B, T, feat_dim)
    dummy = torch.randn(1, args.frames, 80)
    with torch.no_grad():
        emb = model(dummy)
    print("torch embedding shape:", tuple(emb.shape), "norm %.4f" % float(emb.norm()))

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["feats"],
        output_names=["embedding"],
        dynamic_axes={"feats": {1: "T"}, "embedding": {0: "B"}},
        opset_version=13,
        do_constant_folding=True,
    )
    print("wrote", args.out, "%.1f MB" % (os.path.getsize(args.out) / 1048576.0))

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        ref = emb.numpy()
        got = sess.run(None, {"feats": dummy.numpy()})[0]
        cos = float(np.dot(ref.ravel(), got.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
        print("onnxruntime vs torch cosine: %.6f  max|diff| %.2e" % (cos, float(np.abs(ref - got).max())))
    except ImportError:
        print("onnxruntime not installed, skipped the parity check")


if __name__ == "__main__":
    main()
