#!/system/bin/sh
# CPU 占用采样：在 N 个 m5-engine 并发跑同一句期间，累计引擎自身与两个 stage
# 子进程的 CPU jiffies（utime+stime），换算成 CPU-秒与平均占用核数。
# 目的：判定瓶颈是 HTP 串行还是 CPU 饱和——这决定"codec 移出 CPU"有没有价值。
V=/data/local/tmp/llama-m5-htp-v73
OUT=$V/cpuprof-out
TEXT='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
RES=$V/cpuprof-results.txt

export LD_LIBRARY_PATH=$V
export ADSP_LIBRARY_PATH=$V
export GGML_HEXAGON_NHMX=0
export GGML_HEXAGON_OPPOLL=1

rm -rf $OUT; mkdir -p $OUT
: > $RES
cd $V || exit 1

jif() { [ -f /proc/$1/stat ] && awk '{print $14+$15}' /proc/$1/stat 2>/dev/null || echo 0; }

# 累计一个进程族的 CPU jiffies，直到 $2 个采样 tick 结束
# 用法: sample_family <ticks> <n_engines>
sample_family() {
  ticks=$1; n=$2
  tot=0; t=0; last=0
  while [ $t -lt $ticks ]; do
    sleep 0.1
    # 引擎 + 两个 stage（多实例时按所有 pid 求和）
    sum=0
    for p in $(pidof m5-engine 2>/dev/null); do sum=$((sum + $(jif $p))); done
    for p in $(pidof backbone-resident 2>/dev/null); do sum=$((sum + $(jif $p))); done
    for p in $(pidof depth-resident 2>/dev/null); do sum=$((sum + $(jif $p))); done
    if [ "$sum" -gt "$last" ]; then
      tot=$((tot + sum - last))
    fi
    last=$sum
    t=$((t + 1))
  done
  echo $tot
}

run() {
  n=$1
  tag=$2
  i=1
  while [ $i -le $n ]; do
    ./m5-engine model --text "$TEXT" --voice serena --out "$OUT/$tag-$i.wav" \
      > "$OUT/$tag-$i.log" 2>&1 &
    i=$((i + 1))
  done
  # 采样 300 tick 上限，引擎通常 4-9 s 完成
  jif_sum=$(sample_family 120 $n)
  wait
  cpu_s=$(awk "BEGIN{printf \"%.2f\", $jif_sum/100}")
  echo "PROF tag=$tag conc=$n cpu_jiffies=$jif_sum cpu_s=$cpu_s" >> $RES
  i=1
  while [ $i -le $n ]; do
    json=$(grep -oE '\{"audio_s".*\}' "$OUT/$tag-$i.log" | tail -1)
    echo "  REQ conc=$n idx=$i $json" >> $RES
    i=$((i + 1))
  done
  sleep 3
}

run 1 solo
run 2 duo
run 4 quad
cat $RES
