#!/usr/bin/env python3
"""Transcribe with SenseVoiceSmall (ONNX) as an independent bilingual baseline.

Paraformer-zh is documented as bilingual in its training set but was trained on
60000 h that is mostly Mandarin, and the English command path measures 39.50%
WER against 3.39% for Whisper on identical audio. This script exists to answer
one question with a second model rather than with more analysis of the first:
is a single checkpoint that reads both languages well available at all?

SenseVoiceSmall is the candidate. It is one CTC model with a 25055-token
vocabulary covering Mandarin, Cantonese, English, Japanese and Korean, so it
cannot degrade into pinyin when the language flips mid-sentence the way
paraformer-en does.

    python3 sensevoice_baseline.py --onnx model.int8.onnx --tokens tokens.txt \
        --refs cabin_commands.tsv --lang en /tmp/voice-eddy/*_en*.wav

The front end is LFR 7/6 on 80-bin fbank with CMVN, the same shape Paraformer
uses, so the features come from the same code path and a difference cannot be a
feature difference. Text is scored with eval_cer so the rates are comparable
with every other table in this directory.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "x-asr" / "tools"))
from eval_cer import edit_distance, norm_en, norm_zh  # noqa: E402

# Spelled numbers -> digits, so a model that applies ITN is not charged for it.
# Applied to both arms whenever a rate is computed; see PRODUCTION.md.
_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def canon_numbers(text):
    """Rewrite spelled numbers as digits: 'twenty one' -> '21', '40%' -> '40 %'."""
    s = text.lower()

    def tens(m):
        base = _TENS[m.group(1)]
        unit = int(_NUM[m.group(2)])
        return f" {base + unit} "

    s = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
               r" (one|two|three|four|five|six|seven|eight|nine)\b", tens, s)
    for word, digit in _NUM.items():
        s = re.sub(r"\b" + word + r"\b", digit, s)
    return re.sub(r"\s+", " ", s).replace("%", " % ").strip()


def load_tokens(path):
    tok = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            tok[int(parts[1])] = parts[0]
    return tok


def read_wav(path):
    import wave
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz mono PCM16")
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def fbank_lfr(wav, lfr_m=7, lfr_n=6, n_mels=80):
    """80-bin log-mel fbank, then stack LFR frames the way FunASR does."""
    win, hop, nfft = 400, 160, 512
    if len(wav) < win:
        wav = np.pad(wav, (0, win - len(wav)))
    n_frames = 1 + (len(wav) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = wav[idx] * np.hamming(win)[None, :]

    spec = np.abs(np.fft.rfft(frames, n=nfft)) ** 2
    # HTK mel scale, 20 Hz - 8000 Hz
    def hz2mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (np.exp(m / 1127.0) - 1.0)
    mels = np.linspace(hz2mel(20.0), hz2mel(8000.0), n_mels + 2)
    bins = np.floor((nfft + 1) * mel2hz(mels) / 16000).astype(int)
    fb = np.zeros((n_mels, nfft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(1, r - c)
    mel = np.log(spec @ fb.T + 1e-6).astype(np.float32)

    # LFR: stack lfr_n frames with lfr_m context, hop lfr_n
    T = mel.shape[0]
    pad = (lfr_m - 1) // 2
    mel = np.pad(mel, ((pad, pad), (0, 0)), mode="edge")
    T2 = (T + lfr_n - 1) // lfr_n
    out = np.zeros((T2, n_mels * lfr_m), dtype=np.float32)
    for i in range(T2):
        s = i * lfr_n
        out[i] = mel[s:s + lfr_m].reshape(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--cmvn", default=None, help="optional am.mvn (shift/scale)")
    ap.add_argument("--refs", default=None)
    ap.add_argument("--lang", default=None, choices=["zh", "en"])
    ap.add_argument("--text", action="store_true", help="print text too")
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    import onnxruntime as ort
    tokens = load_tokens(args.tokens)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    refs = {}
    if args.refs:
        for line in Path(args.refs).read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if len(f) >= 3:
                refs[Path(f[0]).stem] = (f[1], f[2])

    err = units = 0
    for w in args.wavs:
        pcm = read_wav(w)
        feats = fbank_lfr(pcm)  # (T, 560), already stacked
        # The exported graph takes features, their length, and two scalar
        # controls. language 3 = English, 0 = Mandarin; text_norm 14 leaves
        # numbers as digits, which is what the ITN-aware reference uses.
        lang_id = 3 if args.lang == "en" else 0
        feed = {
            "x": feats[None, ...].astype(np.float32),
            "x_length": np.array([feats.shape[0]], dtype=np.int32),
            "language": np.array([lang_id], dtype=np.int32),
            "text_norm": np.array([14], dtype=np.int32),
        }
        ids = sess.run(None, feed)[0].argmax(-1)[0]

        # greedy CTC collapse
        text, prev = [], -1
        for i in ids:
            if i != prev and i > 3:
                text.append(tokens.get(int(i), ""))
            prev = i
        hyp = "".join(text).replace("▁", " ").strip()
        # The CTC vocabulary carries <|en|><|NEUTRAL|><|Speech|><|withitn|> as
        # separate tokens; they are metadata, not transcript.
        hyp = re.sub(r"<\|[^|]*\|>", "", hyp).strip()
        # text_norm=14 applies ITN, so "21" and "40%" appear where the
        # reference spells them out. Canonicalise both sides to digits or the
        # comparison charges this model for formatting the other one omits.
        hyp = canon_numbers(hyp)

        stem = Path(w).stem
        line = f"{stem}\t{hyp}"
        if stem in refs and args.lang:
            lang, ref = refs[stem]
            if lang == args.lang:
                if args.lang == "zh":
                    r, h = norm_zh(ref), norm_zh(hyp)
                else:
                    r = canon_numbers(norm_en(ref)).split()
                    h = canon_numbers(norm_en(hyp)).split()
                e = edit_distance(r, h)
                err += e
                units += len(r)
                line += f"\t{e}/{len(r)}"
        print(line)
    if units:
        print(f"\nTOTAL {err}/{units} = {100.0 * err / units:.2f}%")


if __name__ == "__main__":
    main()