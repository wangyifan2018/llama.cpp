#!/usr/bin/env python3
"""Score the optional X-ASR INT8 reference on the same 16 kHz manifests."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import wave

import numpy as np
from sentencepiece import SentencePieceProcessor
from sherpa_onnx import OnlineRecognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hotwords", type=Path)
    parser.add_argument("--score", type=float, default=2.0)
    args = parser.parse_args()
    model = args.model
    bpe = SentencePieceProcessor(model_file=str(model / "bpe.model"))
    with NamedTemporaryFile(mode="w", encoding="utf-8") as vocab:
        for i in range(bpe.get_piece_size()):
            print(bpe.id_to_piece(i), bpe.get_score(i), file=vocab)
        vocab.flush()
        recognizer = OnlineRecognizer.from_transducer(
            tokens=str(model / "tokens.txt"), encoder=str(model / "encoder.int8.onnx"),
            decoder=str(model / "decoder.onnx"), joiner=str(model / "joiner.int8.onnx"),
            modeling_unit="bpe", bpe_vocab=vocab.name,
            decoding_method="modified_beam_search", max_active_paths=4, num_threads=2,
            hotwords_file=str(args.hotwords) if args.hotwords else "", hotwords_score=args.score,
        )

    def decode(path):
        with wave.open(str(path), "rb") as wav:
            if wav.getparams()[:3] != (1, 2, 16000):
                raise ValueError(f"expected 16 kHz mono PCM16: {path}")
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        start = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        stream.accept_waveform(16000, np.zeros(9600, dtype=np.float32))
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        return recognizer.get_result(stream), (time.perf_counter() - start) * 1000

    entries = [line.split("\t", 2) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    if not entries:
        raise SystemExit("manifest is empty")
    decode(args.manifest.parent / entries[0][0])
    with args.out.open("w", encoding="utf-8") as out:
        for name, _, reference in entries:
            path = args.manifest.parent / name
            text, elapsed = decode(path)
            row = {"wav": str(path), "reference": reference, "text": text, "decode_ms": elapsed}
            print(json.dumps(row, ensure_ascii=False), file=out, flush=True)


if __name__ == "__main__":
    main()
