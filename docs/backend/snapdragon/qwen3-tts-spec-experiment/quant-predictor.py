#!/usr/bin/env python3
"""Build and verify GGUF variants that only touch selected Qwen3-TTS predictor matrices."""
import argparse
from pathlib import Path
import re

import gguf
import numpy as np

MATRIX = re.compile(r"a\.gen\.code\.blk\.[0-4]\.(attn_[qkv]|attn_out|ffn_gate|ffn_up|ffn_down)\.weight")


def metadata(reader):
    return {name: field.contents() for name, field in reader.fields.items() if not name.startswith("GGUF.")}


def convert(source, target, mode):
    reader = gguf.GGUFReader(source)
    params = metadata(reader)
    arch_key = gguf.Keys.General.ARCHITECTURE
    writer = gguf.GGUFWriter(target, params[arch_key], endianess=reader.endianess)
    for name, value in params.items():
        if name == arch_key:
            continue
        types = reader.fields[name].types
        writer.add_key_value(name, value, types[0], sub_type=types[1] if len(types) > 1 else None)
    writer.data_alignment = int(params.get("general.alignment", writer.data_alignment))
    replacements = {}
    for t in reader.tensors:
        match = MATRIX.fullmatch(t.name)
        if not match:
            continue
        down = match[1] == "ffn_down"
        if (mode.startswith("down-") and not down) or (mode == "keep-down-q8" and down):
            continue
        dtype = gguf.GGMLQuantizationType.F16 if mode == "down-f16" else gguf.GGMLQuantizationType.Q8_0
        replacements[t.name] = (gguf.quantize(t.data.astype(np.float32), dtype), dtype)
    expected = {"all-q8": 35, "keep-down-q8": 30, "down-f16": 5, "down-q8": 5}
    if len(replacements) != expected[mode]:
        raise ValueError("unexpected predictor tensor inventory")
    for t in reader.tensors:
        data, dtype = replacements.get(t.name, (t.data, t.tensor_type))
        writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        data, _ = replacements.get(t.name, (t.data, t.tensor_type))
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()

    check = gguf.GGUFReader(target)
    assert metadata(check) == params
    assert len(check.tensors) == len(reader.tensors)
    old_bytes = new_bytes = 0
    for before, after in zip(reader.tensors, check.tensors):
        assert before.name == after.name and np.array_equal(before.shape, after.shape)
        data, dtype = replacements.get(before.name, (before.data, before.tensor_type))
        assert after.tensor_type == dtype and np.array_equal(after.data, data)
        if before.name in replacements:
            old_bytes += before.n_bytes
            new_bytes += after.n_bytes
    print(f"{mode} verified modified={len(replacements)} all_other_tensors_byte_equal=true "
          f"modified_bytes_before={old_bytes} modified_bytes_after={new_bytes} "
          f"file_bytes={target.stat().st_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--modes", nargs="+", choices=("down-f16", "down-q8", "all-q8", "keep-down-q8"),
                        default=("down-f16", "down-q8", "all-q8", "keep-down-q8"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        convert(args.source, args.out / (mode + ".gguf"), mode)
