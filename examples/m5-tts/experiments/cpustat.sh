#!/system/bin/sh
# 整机 CPU 记账：请求窗口内从 /proc/stat 取全核聚合，算出平均占用核数。
# 用途：判定瓶颈是 HTP 串行还是 CPU 饱和；若平均核数接近 8，则 codec 移出
# CPU 才有价值，否则多实例串行是 HTP 造成的、卸载无用。
V=/data/local/tmp/llama-m5-htp-v73
OUT=$V/cpustat-out
TEXT='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
RES=$V/cpustat-results.txt

export LD_LIBRARY_PATH=$V
export ADSP_LIBRARY_PATH=$V
export GGML_HEXAGON_NHMX=0
export GGML_HEXAGON_OPPOLL=1

rm -rf $OUT; mkdir -p $OUT
: > $RES
cd $V || exit 1

up_s() { awk '{printf "%d", $1*1000}' /proc/uptime; }

run() {
  n=$1
  tag=$2
  sleep 3
  a=$(head -1 /proc/stat)
  t0=$(up_s)
  i=1
  while [ $i -le $n ]; do
    ./m5-engine model --text "$TEXT" --voice serena --out "$OUT/$tag-$i.wav" \
      > "$OUT/$tag-$i.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  t1=$(up_s)
  b=$(head -1 /proc/stat)
  echo "STAT tag=$tag conc=$n wall_ms=$((t1 - t0))" >> $RES
  echo "  A $a" >> $RES
  echo "  B $b" >> $RES
  i=1
  while [ $i -le $n ]; do
    json=$(grep -oE '\{"audio_s".*\}' "$OUT/$tag-$i.log" | tail -1)
    echo "  REQ conc=$n idx=$i $json" >> $RES
    i=$((i + 1))
  done
  sleep 2
}

R=${ROUNDS:-2}
r=1
while [ "$r" -le "$R" ]; do
  if [ $((r % 2)) -eq 1 ]; then order="1 2 4"; else order="4 2 1"; fi
  for n in $order; do
    run $n "r${r}c${n}"
  done
  r=$((r + 1))
done

echo "--- hashes ---" >> $RES
for f in $OUT/*.wav; do sha256sum "$f" >> $RES; done
cat $RES
