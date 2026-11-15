#!/system/bin/sh
# Validate the compiled-in KV32 default against the upstream 256-wide pad.
# Alternating rounds cancel thermal drift; codes/WAV hashes check equivalence.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
T='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
OUT=$D/kv32-results.txt
R=$D/kv32-out
mkdir -p "$R"
: > "$OUT"

run () {
  tag=$1
  timeout 240 ./m5-engine model --cwd "$D" --voice serena --seed 1234 \
    --codec codec-snake.onnx --chunk 4 --threads 3 \
    --text "$T" --out "$R/$tag.wav" --dump-codes "$R/$tag.codes" > "$R/$tag.log" 2>&1
  line=$(grep -a '"audio_s"' "$R/$tag.log" | tail -1)
  [ -z "$line" ] && line="FAILED"
  h=$(sha256sum "$R/$tag.wav" "$R/$tag.codes" 2>/dev/null | cut -c1-16 | tr '\n' ',')
  echo "$tag $line H=$h" >> "$OUT"
  echo "$tag $line H=$h"
  sleep 1
}

case_ () {
  case "$1" in
    A) unset M5_KV_PAD M5_HVX_DIRECT_V; run "$2:A_default_kv32" ;;
    B) export M5_KV_PAD=256; unset M5_HVX_DIRECT_V; run "$2:B_pad256" ;;
    C) unset M5_KV_PAD; export M5_HVX_DIRECT_V=0; run "$2:C_nodirectv" ;;
  esac
}

for r in 1 2 3 4; do
  if [ $((r % 2)) -eq 1 ]; then order="A B C"
  else order="C B A"; fi
  for k in $order; do case_ "$k" "r$r"; done
done
echo "DONE" >> "$OUT"
echo "DONE"
