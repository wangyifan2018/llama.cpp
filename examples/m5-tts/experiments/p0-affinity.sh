#!/system/bin/sh
# P0 part 2: does CPU placement change the depth/codec contention balance?
# taskset binds the whole process tree (engine + stage children inherit affinity).
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
T="今天的天气很好，我们一起去公园散步，看看路边的花，听听树上的鸟叫。"
OUT=$D/p0-aff-results.txt
: > "$OUT"

run () {
  tag=$1; mask=$2
  if [ "$mask" = "none" ]; then
    timeout 240 ./m5-engine model --text "$T" --out "$D/p0.wav" --cwd "$D" \
      --threads 3 --first 2 --chunk 4 > "$D/p0.log" 2>&1
  else
    timeout 240 taskset "$mask" ./m5-engine model --text "$T" --out "$D/p0.wav" --cwd "$D" \
      --threads 3 --first 2 --chunk 4 > "$D/p0.log" 2>&1
  fi
  line=$(grep -a '"audio_s"' "$D/p0.log" | tail -1)
  if [ -z "$line" ]; then line="FAILED"; fi
  echo "$tag $line" >> "$OUT"
  echo "$tag $line"
  sleep 1
}

case_ () {
  case "$1" in
    X) run "$2:X_default" none ;;
    Y) run "$2:Y_little_0f" 0f ;;
    Z) run "$2:Z_big_f0" f0 ;;
    W) run "$2:W_cpu7_80" 80 ;;
  esac
}

for r in 1 2 3 4; do
  if [ $((r % 2)) -eq 1 ]; then order="X Y Z W"
  else order="W Z Y X"; fi
  for k in $order; do case_ "$k" "r$r"; done
done
echo "DONE" >> "$OUT"
echo "DONE"
