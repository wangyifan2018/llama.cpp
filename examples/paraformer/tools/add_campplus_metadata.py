#!/usr/bin/env python3
"""Tag a CAM++ ONNX graph with the metadata sherpa-onnx needs.

sherpa-onnx refuses a speaker model without metadata, and its checks are picky in
two ways that cost time to find:

  * it validates the *framework* field as if it were the model type, so framework
    must be one of the supported families (wespeaker / 3d-speaker / nemo), not
    "torch";
  * it requires output_dim, otherwise Create() aborts with
    "'output_dim' does not exist in the metadata".

With the values below, sherpa's extractor loads our funasr-exported CAM++ graph,
computes the 192-dim embedding on the phone in ~110 ms per 2.3 s clip, and the
gate separates the drivers cleanly (see the design doc).

    python3 add_campplus_metadata.py --in campplus.onnx --out campplus_meta.onnx
"""

import argparse

import onnx

DEFAULTS = {
    "framework": "3d-speaker",
    "model_type": "3d-speaker",
    "sample_rate": "16000",
    "feature_dim": "80",
    "normalize_samples": "1",
    "language": "zh",
    "version": "1",
    "output_dim": "192",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--output-dim", default=DEFAULTS["output_dim"])
    args = ap.parse_args()

    m = onnx.load(args.src)
    for p in list(m.metadata_props):
        m.metadata_props.remove(p)
    for k, v in DEFAULTS.items():
        p = m.metadata_props.add()
        p.key = k
        p.value = args.output_dim if k == "output_dim" else v
    onnx.save(m, args.dst)
    print("wrote", args.dst, "with", DEFAULTS)


if __name__ == "__main__":
    main()
