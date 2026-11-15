#!/system/bin/sh
# OPPOLL 对照：忙轮询（=1）与阻塞等待（=0）的 CPU 成本 vs 墙钟。
# 目的：把「CPU 饱和」拆成「真实算力」与「轮询空转」，判断 codec 移出 CPU 是否必要。
V=/data/local/tmp/llama-m5-htp-v73
OUT=$V/oppoll-out
TEXT='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
RES=$V/oppoll-results.txt

export LD_LIBRARY_PATH=$V
export ADSP_LIBRARY_PATH=$V
export GGML_HEXAGON_NHMX=0

rm -rf $OUT; mkdir -p $OUT
: > $RES
cd $V || exit 1

up_s() { awk '{printf "%d", $1*1000}' /proc/uptime; }

run() {
  tag=$1
  poll=$2
  sleep 3
  a=$(head -1 /proc/stat)
  t0=$(up_s)
  GGML_HEXAGON_OPPOLL=$poll ./m5-engine model --text "$TEXT" --voice serena \
    --out "$OUT/$tag.wav" > "$OUT/$tag.log" 2>&1
  t1=$(up_s)
  b=$(head -1 /proc/stat)
  freqs=""
  for c in 0 1 2 3 4 5 6 7; do
    f=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null)
    freqs="$freqs ${f:-0}"
  done
  echo "STAT tag=$tag oppoll=$poll wall_ms=$((t1 - t0)) freqs=$freqs" >> $RES
  echo "  A $a" >> $RES
  echo "  B $b" >> $RES
  json=$(grep -oE '\{"audio_s".*\}' "$OUT/$tag.log" | tail -1)
  echo "  REQ $json" >> $RES
}

R=${ROUNDS:-2}
r=1
while [ "$r" -le "$R" ]; do
  if [ $((r % 2)) -eq 1 ]; then
    run "r${r}p1" 1; run "r${r}p0" 0
  else
    run "r${r}p0" 0; run "r${r}p1" 1
  fi
  r=$((r + 1))
done

echo "--- hashes ---" >> $RES
sha256sum $OUT/*.wav >> $RES
cat $RES
