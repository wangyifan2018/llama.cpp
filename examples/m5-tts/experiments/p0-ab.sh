#!/system/bin/sh
# P0: quantify the real depth/codec overlap contention, then re-sweep the
# existing knobs by END-TO-END wall time (not by codec_s or first packet).
# 4 alternating rounds (forward / reverse) to cancel thermal drift.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
T="今天的天气很好，我们一起去公园散步，看看路边的花，听听树上的鸟叫。"
OUT=$D/p0-results.txt
: > "$OUT"

run () {
  tag=$1; shift
  timeout 240 ./m5-engine model --text "$T" --out "$D/p0.wav" --cwd "$D" "$@" > "$D/p0.log" 2>&1
  line=$(grep -a '"audio_s"' "$D/p0.log" | tail -1)
  if [ -z "$line" ]; then line="FAILED"; fi
  echo "$tag $line" >> "$OUT"
  echo "$tag $line"
  sleep 1
}

case_ () {
  case "$1" in
    A) run "$2:A_overlap_t3_c4"    --threads 3 --first 2 --chunk 4 ;;
    B) run "$2:B_nooverlap_t3_c4"  --threads 3 --first 2 --chunk 4 --no-overlap ;;
    C) run "$2:C_overlap_t2_c4"    --threads 2 --first 2 --chunk 4 ;;
    D) run "$2:D_overlap_t4_c4"    --threads 4 --first 2 --chunk 4 ;;
    E) run "$2:E_overlap_t3_c8"    --threads 3 --first 2 --chunk 8 ;;
    F) run "$2:F_overlap_t3_c16"   --threads 3 --first 2 --chunk 16 ;;
    G) run "$2:G_overlap_t2_c8"    --threads 2 --first 2 --chunk 8 ;;
    H) run "$2:H_overlap_t1_c4"    --threads 1 --first 2 --chunk 4 ;;
  esac
}

for r in 1 2 3 4; do
  if [ $((r % 2)) -eq 1 ]; then order="A B C D E F G H"
  else order="H G F E D C B A"; fi
  for k in $order; do case_ "$k" "r$r"; done
done
echo "DONE" >> "$OUT"
echo "DONE"
