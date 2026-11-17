#!/usr/bin/env python3
"""Export the ModelScope CAM++ speaker encoder to a fixed-length ONNX graph.

The cabin gate runs this on the CPU while ASR stays on HTP, so it costs the ASR
tail nothing.

Why a fixed length: the funasr CAMPPlus graph only exports exactly for some input
lengths. Measured worst-case cosine against eager over three fresh inputs:

    T=200 -> 1.000000   T=400 -> 1.000000
    T=100 -> 0.571      T=300 -> 0.940

so a dynamic-axis graph (or an arbitrary length) silently returns a wrong
embedding. This exporter therefore builds a T=200 (2 s) graph, verifies parity at
that length, and callers chunk longer audio and average the chunk embeddings.

    python3 export_campplus_onnx.py --bin campplus_cn_common.bin --out campplus_t200.onnx
"""

import argparse
import os

import numpy as np
import torch

from funasr.models.campplus.model import CAMPPlus

FRAMES = 200  # 2 s at the 10 ms hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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
    state = torch.load(args.bin, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("load_state_dict: %d missing, %d unexpected" % (len(missing), len(unexpected)))
    model.eval()

    dummy = torch.randn(1, FRAMES, 80)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["feats"],
        output_names=["embedding"],
        opset_version=13,
        do_constant_folding=True,
    )
    print("wrote", args.out, "%.1f MB (T=%d)" % (os.path.getsize(args.out) / 1048576.0, FRAMES))

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    worst = 1.0
    for _ in range(5):
        x = torch.randn(1, FRAMES, 80)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"feats": x.numpy()})[0]
        cos = float(np.dot(ref.ravel(), got.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
        worst = min(worst, cos)
    print("parity over 5 inputs at T=%d: worst cosine %.6f" % (FRAMES, worst))
    if worst < 0.999:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()
