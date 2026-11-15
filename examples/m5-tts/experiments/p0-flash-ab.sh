#!/system/bin/sh
# P0 attribution: isolate the fused attention kernel from KV precision.
# A = FP32 KV + ordinary attention (compiled-in KV32 + direct-V default)
# B = FP32 KV + FLASH_ATTN_EXT (M5_DEPTH_ATTENTION=flash-f32)
# Alternating rounds cancel thermal drift. Codes/WAV hashes check equivalence.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1 M5_HEAD_STATS=1
OUT=$D/p0flash-results.txt
R=$D/p0flash-out
mkdir -p "$R"
: > "$OUT"

T1='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
T2='沿中山北路行驶三点五公里，在第二个路口左转，限速六十公里每小时。'
T3='各位旅客请注意，开往北京南站的列车即将检票进站。'

text_for () {
  case "$1" in
    1) echo "$T1" ;;
    2) echo "$T2" ;;
    3) echo "$T3" ;;
  esac
}

run () {
  tag=$1; tid=$2; seed=$3; mode=$4
  text=$(text_for "$tid")
  if [ "$mode" = B ]; then export M5_DEPTH_ATTENTION=flash-f32
  else unset M5_DEPTH_ATTENTION; fi
  timeout 240 ./m5-engine model --cwd "$D" --voice serena --seed "$seed" \
    --codec codec-snake.onnx --chunk 4 --threads 3 \
    --text "$text" --out "$R/$tag.wav" --dump-codes "$R/$tag.codes" > "$R/$tag.log" 2>&1 || {
      echo "$tag FAILED" >> "$OUT"; echo "$tag FAILED"; tail -6 "$R/$tag.log"; return; }
  line=$(grep -a '"audio_s"' "$R/$tag.log" | tail -1)
  [ -z "$line" ] && line="FAILED"
  h=$(sha256sum "$R/$tag.wav" "$R/$tag.codes" 2>/dev/null | cut -c1-16 | tr '\n' ',')
  core=$(grep -a 'core_ms' "$R/$tag.log" | tail -20 | awk '{split($3,a,"="); s+=a[2]; n++} END {if(n) printf "%.2f", s/n; else print "0"}')
  st=$(grep -a 'stage_call_s' "$R/$tag.log" | tail -1 | sed 's/.*stage_call_s=//')
  echo "$tag mode=$mode tid=$tid seed=$seed core_ms=$core stage_call_s=$st H=$h $line" >> "$OUT"
  echo "$tag core=$core stage=$st $line H=$h"
  sleep 1
}

for r in 1 2 3 4; do
  if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for k in $order; do run "perf_r${r}_$k" 1 1234 "$k"; done
done
for t in 1 2 3; do
  for k in A B; do run "text${t}_$k" "$t" 1234 "$k"; done
done
for k in A B; do run "seed42_$k" 1 42 "$k"; done
echo DONE >> "$OUT"
echo DONE
