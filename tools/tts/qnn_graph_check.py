#!/usr/bin/env python3
"""Build a structural ONNX skeleton of the Qwen3-TTS code2wav graph and validate it.

This does not need the model, the QNN SDK or a device. It answers three questions before
any porting work starts:
  1. is the op mix expressible as a valid ONNX graph with static shapes (onnx.checker + shape inference)
  2. which of those ops are in the QNN HTP supported list
  3. which are structural ops that the QNN EP folds away instead of executing

Run: python3 tools/tts/qnn_graph_check.py
"""
from __future__ import annotations

import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

# from the ONNX Runtime QNN Execution Provider supported-operator table (HTP backend)
QNN_HTP_SUPPORTED = {
    "Abs", "Add", "And", "ArgMax", "ArgMin", "Asin", "Atan", "AveragePool", "BatchNormalization",
    "Cast", "Clip", "Concat", "Conv", "ConvTranspose", "Cos", "DequantizeLinear", "Div", "Elu",
    "Equal", "Exp", "Expand", "Flatten", "Floor", "Gather", "Gelu", "Gemm", "GlobalAveragePool",
    "Greater", "GreaterOrEqual", "HardSwish", "InstanceNormalization", "LRN", "LayerNormalization",
    "LeakyRelu", "Less", "LessOrEqual", "Log", "LogSoftmax", "LpNormalization", "MatMul", "Max",
    "MaxPool", "Min", "Mul", "Neg", "Not", "Or", "Prelu", "Pad", "Pow", "QuantizeLinear",
    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "Relu", "Resize", "Round",
    "Sigmoid", "Sign", "Sin", "Slice", "Softmax", "SpaceToDepth", "Split", "Sqrt", "Squeeze", "Sub",
    "Tanh", "Tile", "TopK", "Transpose", "Unsqueeze", "Where",
}

# ops the EP rewrites at graph level; they never reach the HTP backend as kernels
STRUCTURAL = {"Reshape", "Shape", "Constant", "Identity", "Range", "GatherElements", "ScatterND"}


def conv1d(node_in, w, b, name, dil=1):
    """Conv with H == 1, which is how a 1D conv is expressed in ONNX."""
    return helper.make_node("Conv", [node_in, w, b], [name],
                            kernel_shape=[1, 7], dilations=[1, dil], pads=[0, 3 * dil, 0, 3 * dil])


def build(frames: int = 8, n_codes: int = 16, code_vocab: int = 2048, dim: int = 512):
    """Skeleton of: codebook decode -> conv -> snake -> 3 res units -> 2 upsample -> 4 DAC blocks."""
    nodes, inits, inputs, outputs = [], [], [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name))
        return name

    # codes [1, 16, T] -> embed each codebook with Gather, sum, project (MatMul)
    inputs.append(helper.make_tensor_value_info("codes", TensorProto.INT64, [1, n_codes, frames]))
    emb_names = []
    for g in range(n_codes):
        tbl = const(f"codebook_{g}", np.random.RandomState(g).randn(code_vocab, dim).astype(np.float32) * 0.02)
        idx = helper.make_tensor(f"idx_{g}", TensorProto.INT64, [1], [g])
        inits.append(idx)
        nodes.append(helper.make_node("Gather", ["codes", f"idx_{g}"], [f"codes_g{g}"], axis=1))
        nodes.append(helper.make_node("Gather", [tbl, f"codes_g{g}"], [f"emb_{g}"], axis=0))
        emb_names.append(f"emb_{g}")
    acc = emb_names[0]
    for g in range(1, n_codes):
        dst = f"emb_sum_{g}"
        nodes.append(helper.make_node("Add", [acc, emb_names[g]], [dst]))
        acc = dst

    # project the summed embeddings to the conv input width
    w = const("in_proj", np.random.RandomState(99).randn(dim, dim).astype(np.float32) * 0.02)
    nodes.append(helper.make_node("MatMul", [f"emb_sum_{n_codes - 1}", w], ["proj"]))
    inits.append(numpy_helper.from_array(np.array([1, frames, dim], dtype=np.int64), "proj_shape"))
    nodes.append(helper.make_node("Reshape", ["proj", "proj_shape"], ["proj_3d"]))
    nodes.append(helper.make_node("Transpose", ["proj_3d"], ["x"], perm=[0, 2, 1]))  # [1, dim, T]

    # one conv + snake
    cur = "x"
    wname = const("conv0_w", np.random.RandomState(1).randn(dim, dim, 1, 7).astype(np.float32) * 0.02)
    bname = const("conv0_b", np.zeros(dim, dtype=np.float32))
    nodes.append(conv1d(cur, wname, bname, "conv0"))
    alpha = const("snake_a", np.ones((1, dim, 1), dtype=np.float32))
    nodes.append(helper.make_node("Mul", ["conv0", alpha], ["sn_in"]))
    nodes.append(helper.make_node("Sin", ["sn_in"], ["sn_sin"]))
    nodes.append(helper.make_node("Pow", ["sn_sin", "sn_two"], ["sn_sq"]))
    nodes.append(helper.make_node("Add", ["conv0", "sn_sq"], ["sn_out"]))
    inits.append(numpy_helper.from_array(np.array(2.0, dtype=np.float32), "sn_two"))
    cur = "sn_out"

    # three residual units: conv -> snake -> conv -> snake -> add
    for r in range(3):
        w1 = const(f"res{r}_w1", np.random.RandomState(10 + r).randn(dim, dim, 1, 7).astype(np.float32) * 0.01)
        w2 = const(f"res{r}_w2", np.random.RandomState(20 + r).randn(dim, dim, 1, 7).astype(np.float32) * 0.01)
        bz = const(f"res{r}_b", np.zeros(dim, dtype=np.float32))
        nodes.append(conv1d(cur, w1, bz, f"res{r}_c1"))
        nodes.append(helper.make_node("Mul", [f"res{r}_c1", alpha], [f"res{r}_m1"]))
        nodes.append(helper.make_node("Sin", [f"res{r}_m1"], [f"res{r}_s1"]))
        nodes.append(helper.make_node("Add", [f"res{r}_c1", f"res{r}_s1"], [f"res{r}_a1"]))
        nodes.append(conv1d(f"res{r}_a1", w2, bz, f"res{r}_c2"))
        nodes.append(helper.make_node("Add", [cur, f"res{r}_c2"], [f"res{r}_out"]))
        cur = f"res{r}_out"

    # two upsample stages: ConvTranspose with stride 2 (stays native, no col2im decomposition)
    for u, stride in enumerate((2, 2)):
        wu = const(f"up{u}_w", np.random.RandomState(30 + u).randn(dim, dim, 1, stride * 2).astype(np.float32) * 0.02)
        nodes.append(helper.make_node("ConvTranspose", [cur, wu], [f"up{u}"],
                                      kernel_shape=[1, stride * 2], strides=[1, stride],
                                      pads=[0, stride // 2, 0, stride // 2]))
        cur = f"up{u}"

    # DAC style: four blocks, each strided conv transpose + three res units
    for blk, stride in enumerate((8, 5, 4, 3)):
        wb = const(f"dac{blk}_w", np.random.RandomState(40 + blk).randn(dim, dim, 1, stride * 2).astype(np.float32) * 0.01)
        nodes.append(helper.make_node("ConvTranspose", [cur, wb], [f"dac{blk}"],
                                      kernel_shape=[1, stride * 2], strides=[1, stride]))
        cur = f"dac{blk}"

    wout = const("out_w", np.random.RandomState(50).randn(1, dim, 1, 7).astype(np.float32) * 0.02)
    nodes.append(conv1d(cur, wout, const("out_b", np.zeros(1, dtype=np.float32)), "audio"))
    outputs.append(helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "audio_len"]))

    graph = helper.make_graph(nodes, "code2wav_skeleton", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="qwen3tts-qnn-check")
    return model


def main() -> int:
    model = build()
    onnx.checker.check_model(model)
    inferred = shape_inference.infer_shapes(model)
    audio = [v for v in inferred.graph.output if v.name == "audio"][0]
    dims = [d.dim_value for d in audio.type.tensor_type.shape.dim]

    ops = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1

    print(f"onnx {onnx.__version__}: checker ok, {len(model.graph.node)} nodes, {len(model.graph.initializer)} weights")
    print(f"audio output shape (inferred): {dims}")
    # shape inference does not propagate the length through Gather/Reshape/Conv here, so compute it
    L = 8  # build() default frames
    for _ in range(2):
        L = 2 * L                     # ConvTranspose stride 2, kernel 4, pad 1
    for s in (8, 5, 4, 3):
        L = (L + 1) * s               # ConvTranspose stride s, kernel 2s, no pad
    print(f"audio output length (computed): {L} samples for 8 frames")
    if dims[-1] != L:
        print("note: onnx shape inference left the length unresolved -> QNN needs explicit shapes")
    print()
    print(f"{chr(39)}op{chr(39):<16} count  qnn-htp")
    bad = []
    for op, n in sorted(ops.items(), key=lambda kv: -kv[1]):
        if op in QNN_HTP_SUPPORTED:
            status = "yes"
        elif op in STRUCTURAL:
            status = "folded"
        else:
            status = "NO"
            bad.append(op)
        print(f"{op:<18} {n:5d}  {status}")
    print()
    if bad:
        print("unsupported by QNN HTP:", ", ".join(bad))
        return 1
    print("every op is either QNN-supported or folded at graph level")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())