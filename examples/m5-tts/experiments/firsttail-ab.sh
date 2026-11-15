#!/system/bin/sh
# Experiment 2 (doc 21.6 #2): first-chunk tail alignment.
#   A  --first 2, tail keep 2400 (default)  - current best (0.398 s)
#   B  --first 1, M5_FIRST_TAIL=0          - one frame earlier, should emit
#   C  --first 2, M5_FIRST_TAIL=0           - boundary-only change: same WAV,
#                                             first_pcm must be unchanged
#   D  --first 1, tail keep 2400            - current behaviour: no PCM from
#                                             the first chunk (waits 5 frames)
# Fresh output dir per round: --dump-codes appends, reused tags would stack.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
unset M5_KV_PAD M5_HVX_DIRECT_V M5_FIRST_TAIL
T='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
OUT=$D/firsttail-results.txt
R=$D/firsttail-out
mkdir -p "$R"
: > "$OUT"

run () {
  tag=$1 first=$2 ft=$3
  if [ "$ft" = "none" ]; then unset M5_FIRST_TAIL; else export M5_FIRST_TAIL=$ft; fi
  timeout 240 ./m5-engine model --cwd "$D" --voice serena --seed 1234 \
    --codec codec-snake.onnx --first "$first" --chunk 4 --threads 3 \
    --text "$T" --out "$R/$tag.wav" --dump-codes "$R/$tag.codes" > "$R/$tag.log" 2>&1
  unset M5_FIRST_TAIL
  line=$(grep -a '"audio_s"' "$R/$tag.log" | tail -1)
  [ -z "$line" ] && line="FAILED $(tail -3 "$R/$tag.log" | tr '\n' ' ')"
  h=$(sha256sum "$R/$tag.wav" "$R/$tag.codes" 2>/dev/null | cut -c1-16 | tr '\n' ',')
  echo "$tag first=$first tail=$ft $line H=$h" >> "$OUT"
  echo "$tag first=$first tail=$ft $line H=$h"
  sleep 1
}

r=1
while [ "$r" -le 3 ]; do
  rm -rf "$R"; mkdir -p "$R"
  if [ $((r % 2)) -eq 1 ]; then order="A B C D"; else order="D C B A"; fi
  for k in $order; do
    case "$k" in
      A) run "r$r:$k" 2 none ;;
      B) run "r$r:$k" 1 0 ;;
      C) run "r$r:$k" 2 0 ;;
      D) run "r$r:$k" 1 none ;;
    esac
  done
  r=$((r + 1))
done
echo DONE >> "$OUT"
echo DONE
