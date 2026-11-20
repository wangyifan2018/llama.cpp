"""Corpus -> training shards.

Two products, both keyed by the same 12.5 Hz frames:

  t1  hidden[t, 1024] + codes[t, 16]   the talker's hidden state and the tokenizer codes
  t2  codes[t, 16]    + pcm[t*1920]    the codes and the matching waveform

Needs the reference implementation (pip install qwen-tts, transformers >= 4.57). This script is the
only part of the pipeline that does; everything downstream reads the shard format and runs on plain
torch.

The talker hidden is produced by teacher forcing: the prompt rows, then the codes of the clip, taking
the hidden state of the last position of every frame. Use the same prompt shape as deployment
(speaker row for cloning, voice row for CustomVoice) -- the predictor learns the distribution it sees
here, so a mismatched prompt silently costs agreement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import datafmt


def encode_codes(tokenizer, wav: str) -> np.ndarray:
    """[T, 16] int32 codes for one clip."""
    enc = tokenizer.encode(wav)
    codes = enc[0] if isinstance(enc, (list, tuple)) else enc
    codes = np.asarray(codes, dtype=np.int32)
    if codes.ndim != 2 or codes.shape[1] != datafmt.N_CODES:
        raise ValueError(f"unexpected code shape {codes.shape}")
    return codes


def talker_hidden(model, codes: np.ndarray, speaker_wav: str | None, language: str = "english") -> np.ndarray:
    """[T, 1024] float16, one hidden state per frame, teacher forced over the clip's own codes."""
    import torch

    # the reference exposes the talker through generate(); passing the clip's codes as the
    # continuation is what we need, see qwen_tts/core/models/modeling_qwen3_tts.py: forward()
    # returns the per step hidden states when output_hidden_states=True
    with torch.no_grad():
        out = model.generate(
            input_ids=None,
            voice_clone_prompt=speaker_wav,
            languages=[language],
            codec_input=codes,            # teacher forcing: feed the true codes
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=0,
        )
    h = out.talker_hidden_states
    h = h[0] if isinstance(h, (list, tuple)) else h
    h = h.reshape(-1, datafmt.N_EMBD)
    n = min(len(h), len(codes))
    return h[:n].astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", required=True, help="glob of 24 kHz mono wavs")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--kind", choices=["t1", "t2"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="reference checkpoint")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    ap.add_argument("--language", default="english")
    ap.add_argument("--speaker", default=None, help="reference wav for the speaker row (cloning)")
    ap.add_argument("--frames-per-shard", type=int, default=50_000)
    ap.add_argument("--dry-run", action="store_true", help="fabricate random data, to test the format")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rng = np.random.default_rng(0)
        n = 5_000
        hidden = rng.standard_normal((n, datafmt.N_EMBD)).astype(np.float16)
        codes = rng.integers(0, datafmt.CODE_VOCAB, (n, datafmt.N_CODES), dtype=np.int32)
        if args.kind == "t1":
            datafmt.write_t1(out / "shard_0000.npz", hidden, codes)
        else:
            pcm = rng.standard_normal(n * datafmt.HOP).astype(np.float16) * 0.1
            datafmt.write_t2(out / "shard_0000.npz", codes, pcm)
        print(f"dry run wrote 1 shard to {out}")
        return 0

    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer

    tok = Qwen3TTSTokenizer.from_pretrained(args.tokenizer, device_map=args.device)
    model = None
    if args.kind == "t1":
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device)

    wavs = sorted(Path().glob(args.wavs))
    if not wavs:
        print(f"no wavs match {args.wavs}", file=sys.stderr)
        return 1
    print(f"{len(wavs)} clips -> {out} ({args.kind})")

    buf_h, buf_c, buf_p = [], [], []
    shard = 0
    for i, w in enumerate(wavs):
        codes = encode_codes(tok, str(w))
        if args.kind == "t1":
            h = talker_hidden(model, codes, args.speaker, args.language)
            buf_h.append(h)
            buf_c.append(codes[: len(h)])
        else:
            pcm, sr = sf.read(str(w), dtype="float32", always_2d=False)
            if sr != 24000:
                raise ValueError(f"{w}: {sr} Hz, expected 24000")
            n = len(codes) * datafmt.HOP
            if len(pcm) < n:
                raise ValueError(f"{w}: {len(pcm)} samples < {n} needed for {len(codes)} frames")
            buf_c.append(codes)
            buf_p.append(pcm[:n].astype(np.float16))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(wavs)} clips")

        n_frames = sum(len(x) for x in buf_c)
        if n_frames >= args.frames_per_shard:
            path = out / f"shard_{shard:04d}.npz"
            if args.kind == "t1":
                datafmt.write_t1(path, np.concatenate(buf_h), np.concatenate(buf_c))
            else:
                datafmt.write_t2(path, np.concatenate(buf_c), np.concatenate(buf_p))
            print(f"  wrote {path} ({n_frames} frames)")
            shard += 1
            buf_h, buf_c, buf_p = [], [], []

    if buf_c:
        path = out / f"shard_{shard:04d}.npz"
        if args.kind == "t1":
            datafmt.write_t1(path, np.concatenate(buf_h), np.concatenate(buf_c))
        else:
            datafmt.write_t2(path, np.concatenate(buf_c), np.concatenate(buf_p))
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
