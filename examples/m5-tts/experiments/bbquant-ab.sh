#!/system/bin/sh
# Doc 21.6 item 4: backbone intermediate quantisation (Q6_K / Q5_K_M / mixed).
# Only the backbone stage's gguf changes; depth stays Q4_0 core, codec unchanged.
# Alternating rounds cancel thermal drift. Quantisation changes outputs, so the
# gate is audio length + ASR + human listening, not bit equality.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
unset M5_KV_PAD M5_HVX_DIRECT_V M5_PREFIX_STATS M5_PRIME_CACHE
T='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
OUT=$D/bbquant-results.txt
R=$D/bbquant-out
rm -rf "$R"; mkdir -p "$R"
: > "$OUT"

run () {
  tag=$1 gguf=$2
  BB="backbone-resident $gguf HTP0 --pipe"
  timeout 240 ./m5-engine model --cwd "$D" --voice serena --seed 1234 \
    --codec codec-snake.onnx --chunk 4 --threads 3 \
    --backbone "$BB" \
    --text "$T" --out "$R/$tag.wav" --dump-codes "$R/$tag.codes" > "$R/$tag.log" 2>&1
  line=$(grep -a '"audio_s"' "$R/$tag.log" | tail -1)
  [ -z "$line" ] && line="FAILED $(tail -3 "$R/$tag.log" | tr '\n' ' ')"
  h=$(sha256sum "$R/$tag.wav" "$R/$tag.codes" 2>/dev/null | cut -c1-12 | tr '\n' ',')
  echo "$tag $gguf $line H=$h" >> "$OUT"
  echo "$tag $gguf $line H=$h"
  sleep 1
}

ROUNDS=${ROUNDS:-3}
r=1
while [ "$r" -le "$ROUNDS" ]; do
  if [ $((r % 2)) -eq 1 ]; then order="A B C D E"; else order="E D C B A"; fi
  for k in $order; do
    case "$k" in
      A) run "r$r:$k:q8"    backbone-Q8_0.gguf ;;
      B) run "r$r:$k:q6k"   backbone-Q6_K.gguf ;;
      C) run "r$r:$k:q5km"  backbone-Q5_K_M.gguf ;;
      D) run "r$r:$k:mix4"  backbone-mix4.gguf ;;
      E) run "r$r:$k:q4_0"  backbone-Q4_0.gguf ;;
    esac
  done
  r=$((r + 1))
done
echo DONE >> "$OUT"
echo DONE
