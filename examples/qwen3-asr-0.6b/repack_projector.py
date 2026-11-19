#!/usr/bin/env python3
"""Convert only the Qwen3-ASR audio projection from F16 to Q8_0."""

import argparse
from pathlib import Path

import gguf


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("use a new output file")
    source = gguf.GGUFReader(args.source)
    projection = [tensor for tensor in source.tensors if tensor.name == "a.conv_out.weight"]
    if len(projection) != 1 or projection[0].tensor_type != gguf.GGMLQuantizationType.F16:
        cli.error("the source must have an F16 a.conv_out.weight")
    writer = gguf.GGUFWriter(args.output, arch=source.fields["general.architecture"].contents())
    for name, field in source.fields.items():
        if not name.startswith("GGUF.") and name != "general.architecture":
            subtype = field.types[1] if len(field.types) > 1 else None
            writer.add_key_value(name, field.contents(), field.types[0], subtype)
    for tensor in source.tensors:
        if tensor.name == "a.conv_out.weight":
            kind = gguf.GGMLQuantizationType.Q8_0
            writer.add_tensor(tensor.name, gguf.quantize(tensor.data, kind), raw_dtype=kind)
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
