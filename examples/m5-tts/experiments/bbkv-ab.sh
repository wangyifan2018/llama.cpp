#!/system/bin/sh
# Experiment 1 (doc 21.6): does the backbone stage benefit from a smaller KV pad?
# Only the backbone stage gets M5_KV_PAD; depth stays at the compiled-in 32.
# Alternating rounds cancel thermal drift; codes/WAV hashes check equivalence.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
unset M5_KV_PAD M5_HVX_DIRECT_V M5_PREFIX_STATS
T='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
OUT=$D/bbkv-results.txt
R=$D/bbkv-out
mkdir -p "$R"
: > "$OUT"

run () {
  tag=$1 pad=$2
  BB="backbone-resident backbone-Q8_0.gguf HTP0 --pipe"
  [ "$pad" != "default" ] && BB="M5_KV_PAD=$pad $BB"
  timeout 240 ./m5-engine model --cwd "$D" --voice serena --seed 1234 \
    --codec codec-snake.onnx --chunk 4 --threads 3 \
    --backbone "$BB" \
    --text "$T" --out "$R/$tag.wav" --dump-codes "$R/$tag.codes" > "$R/$tag.log" 2>&1
  line=$(grep -a '"audio_s"' "$R/$tag.log" | tail -1)
  [ -z "$line" ] && line="FAILED $(tail -3 "$R/$tag.log" | tr '\n' ' ')"
  h=$(sha256sum "$R/$tag.wav" "$R/$tag.codes" 2>/dev/null | cut -c1-16 | tr '\n' ',')
  echo "$tag pad=$pad $line H=$h" >> "$OUT"
  echo "$tag pad=$pad $line H=$h"
  sleep 1
}

ROUNDS=${ROUNDS:-4}
r=1
while [ "$r" -le "$ROUNDS" ]; do
  if [ $((r % 2)) -eq 1 ]; then order="A B C D"; else order="D C B A"; fi
  for k in $order; do
    case "$k" in
      A) run "r$r:$k:default" default ;;
      B) run "r$r:$k:pad64" 64 ;;
      C) run "r$r:$k:pad128" 128 ;;
      D) run "r$r:$k:pad32" 32 ;;
    esac
  done
  r=$((r + 1))
done
echo DONE >> "$OUT"
echo DONE
