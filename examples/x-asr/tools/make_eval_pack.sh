#!/bin/sh
# Render the evaluation utterances with the host speech synthesiser and resample
# them to 16 kHz mono. The noise option adds a cabin recording with equal weight;
# it is not level calibrated, and in the first attempt it made the speech
# unintelligible, so do not read the result as a fixed SNR.
#
#   tools/make_eval_pack.sh OUT_DIR [SNR_DB]
#
# The transcript for each file is the synthesiser input in tools/eval_texts.tsv,
# so this is a labelled set. It is clean synthetic speech, not cabin audio: use
# it to rank variants, not to estimate field accuracy.
set -eu

OUT=${1:?usage: make_eval_pack.sh OUT_DIR [SNR_DB]}
SNR=${2:-}
HERE=$(dirname "$0")
ZH_VOICE=${ZH_VOICE:-Flo (中文（中国大陆）)}
EN_VOICE=${EN_VOICE:-Samantha}
NOISE=${NOISE:-}
mkdir -p "$OUT"
: > "$OUT/manifest.tsv"

while IFS="$(printf '	')" read -r name lang text; do
    [ -n "$name" ] || continue
    if [ "$lang" = "zh" ]; then voice=$ZH_VOICE; else voice=$EN_VOICE; fi
    aiff="$OUT/$name.aiff"
    say -v "$voice" -o "$aiff" "$text"
    wav="$OUT/$name.wav"
    ffmpeg -nostdin -v error -y -i "$aiff" -ar 16000 -ac 1 -c:a pcm_s16le "$wav"
    rm -f "$aiff"
    if [ -n "$SNR" ] && [ -n "$NOISE" ]; then
        noisy="$OUT/${name}-snr${SNR}.wav"
        ffmpeg -nostdin -v error -y -stream_loop -1 -i "$NOISE" -i "$wav" \
            -filter_complex "[0:a]aformat=channel_layouts=mono,volume=1[n];[1:a]aformat=channel_layouts=mono[s];[s][n]amix=inputs=2:duration=first:weights='1 1'[m]" \
            -map "[m]" -ar 16000 -ac 1 -c:a pcm_s16le "$noisy" || cp "$wav" "$noisy"
        printf '%s\t%s\t%s\n' "${name}-snr${SNR}.wav" "$lang" "$text" >> "$OUT/manifest.tsv"
    else
        printf '%s\t%s\t%s\n' "${name}.wav" "$lang" "$text" >> "$OUT/manifest.tsv"
    fi
done < "$HERE/eval_texts.tsv"

echo "wrote $(wc -l < "$OUT/manifest.tsv") entries to $OUT/manifest.tsv"
