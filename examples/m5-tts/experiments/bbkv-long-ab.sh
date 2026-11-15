#!/system/bin/sh
# Long-sentence A/B for the backbone KV pad: sequences past 256 rows force
# graph rebuilds at different rates, check that pad128 is not eaten alive.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
unset M5_KV_PAD M5_HVX_DIRECT_V
T='前方三百米右转进入高架桥，随后靠左行驶进入匝道，别着急，我们还有足够的时间，沿着这条路一直向前，过了红绿灯路口后右转就到目的地了。'
OUT=$D/bbkv-long-results.txt
R=$D/bbkv-long-out
mkdir -p "$R"
: > "$OUT"

run () {
  tag=$1 pad=$2
  BB="backbone-resident backbone-Q8_0.gguf HTP0 --pipe"
  [ "$pad" != "default" ] && BB="M5_KV_PAD=$pad $BB"
  timeout 300 ./m5-engine model --cwd "$D" --voice serena --seed 1234 \
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

r=1
while [ "$r" -le 3 ]; do
  if [ $((r % 2)) -eq 1 ]; then order="A C"; else order="C A"; fi
  for k in $order; do
    case "$k" in
      A) run "r$r:$k:default" default ;;
      C) run "r$r:$k:pad128" 128 ;;
    esac
  done
  r=$((r + 1))
done
echo DONE >> "$OUT"
echo DONE
