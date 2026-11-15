#!/system/bin/sh
# 双/四实例并发探针：N 个 m5-engine 同时跑同一句，比较墙钟与单实例。
# 判据：wall(N) / wall(1) ≈ 1 → 可并行；≈ N → 完全串行（单 HTP0 分时）。
V=/data/local/tmp/llama-m5-htp-v73
OUT=$V/multi-out
TEXT='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
RES=$V/multi-results.txt

export LD_LIBRARY_PATH=$V
export ADSP_LIBRARY_PATH=$V
export GGML_HEXAGON_NHMX=0
export GGML_HEXAGON_OPPOLL=1

rm -rf $OUT; mkdir -p $OUT
: > $RES
cd $V || exit 1

now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
npu_temp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1; }

run_group() {
  n=$1
  i=1
  t0=$(now_ms)
  while [ $i -le $n ]; do
    ./m5-engine model --text "$TEXT" --voice serena --out "$OUT/n${n}_$i.wav" \
      > "$OUT/n${n}_$i.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  t1=$(now_ms)
  wall=$((t1 - t0))
  echo "MULTI conc=$n wall_ms=$wall" >> $RES
  i=1
  while [ $i -le $n ]; do
    json=$(grep -oE '\{"audio_s".*\}' "$OUT/n${n}_$i.log" | tail -1)
    echo "  REQ conc=$n idx=$i $json" >> $RES
    i=$((i + 1))
  done
  sleep 3
}

R=${ROUNDS:-2}
r=1
while [ "$r" -le "$R" ]; do
  if [ $((r % 2)) -eq 1 ]; then order="1 2 4"; else order="4 2 1"; fi
  for n in $order; do
    echo "== round $r conc=$n =="
    run_group $n
  done
  r=$((r + 1))
done

echo "--- hashes ---" >> $RES
i=1
while [ $i -le 4 ]; do
  [ -f "$OUT/n4_$i.wav" ] && sha256sum "$OUT/n4_$i.wav" >> $RES
  i=$((i + 1))
done
cat $RES
