"""Shard format shared by the training stages.

Two kinds of shard, one directory per corpus:

  t1  (predictor)   hidden[N, 1024] f16 + codes[N, 16] i32   one row per audio frame
  t2  (vocoder)     codes[N, 16] i32  + pcm[N*1920] f16      one row per audio frame, 1920 samples each

A shard is an .npz plus a small .json sidecar so tools can check the layout without loading it.
Frames are 80 ms (12.5 Hz), so 200 h of audio is 9,000,000 rows.
"""
from __future__ import annotations

import glob as _glob
import json
from pathlib import Path
import numpy as np

N_EMBD = 1024      # talker hidden size, and the predictor's input
N_CODES = 16       # 1 semantic + 15 acoustic
HOP = 1920         # 24 kHz samples per frame
CODE_VOCAB = 2048  # every codebook


def _sidecar(path: Path, kind: str, n: int, extra: dict) -> None:
    meta = {"kind": kind, "n": int(n), "n_embd": N_EMBD, "n_codes": N_CODES, "hop": HOP}
    meta.update(extra)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def write_t1(path: str | Path, hidden: np.ndarray, codes: np.ndarray) -> None:
    path = Path(path)
    assert hidden.dtype == np.float16 and hidden.shape[1] == N_EMBD, hidden.shape
    assert codes.dtype == np.int32 and codes.shape == (hidden.shape[0], N_CODES), codes.shape
    np.savez(path, hidden=hidden, codes=codes)
    _sidecar(path, "t1", len(hidden), {"source": "prepare_pairs.py"})


def read_t1(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["hidden"], z["codes"]


def write_t2(path: str | Path, codes: np.ndarray, pcm: np.ndarray) -> None:
    path = Path(path)
    assert codes.dtype == np.int32 and codes.shape[1] == N_CODES
    assert pcm.dtype == np.float16 and pcm.size == codes.shape[0] * HOP, (pcm.size, codes.shape[0] * HOP)
    np.savez(path, codes=codes, pcm=pcm)
    _sidecar(path, "t2", len(codes), {"source": "prepare_pairs.py"})


def read_t2(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["codes"], z["pcm"]


def list_shards(pattern: str | Path) -> list[Path]:
    # glob() (not Path.glob) so absolute patterns work on every python version
    return sorted(Path(p) for p in _glob.glob(str(pattern)))


def load_t1_all(pattern: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every t1 shard matching the glob. Only for the S0 gate; T1 proper streams shards."""
    hs, cs = [], []
    for p in list_shards(pattern):
        if p.suffix != ".npz":
            continue
        h, c = read_t1(p)
        hs.append(h)
        cs.append(c)
    if not hs:
        raise FileNotFoundError(f"no t1 shards match {pattern}")
    return np.concatenate(hs), np.concatenate(cs)
