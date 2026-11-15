#!/system/bin/sh
# 常驻 server 的 CPU 记账：排除 CLI 每次启动的 ORT session 初始化，量出一次
# 真实合成吃掉多少 CPU-秒，以及 2 路并发时 CPU 是否饱和。
V=/data/local/tmp/llama-m5-htp-v73
OUT=$V/srvcpu-out
BODY='{"input":"前方三百米右转，随后靠左行驶。别着急，我们还有时间。","voice":"serena","response_format":"wav"}'
RES=$V/srvcpu-results.txt

export LD_LIBRARY_PATH=$V
export ADSP_LIBRARY_PATH=$V
export GGML_HEXAGON_NHMX=0
export GGML_HEXAGON_OPPOLL=1

rm -rf $OUT; mkdir -p $OUT
: > $RES
cd $V || exit 1

req() {
  tag=$1
  curl -s -m 300 -o /dev/null -D "$OUT/$tag.hdr" \
    -H 'Content-Type: application/json' -d "$BODY" \
    http://127.0.0.1:8080/v1/audio/speech
}

emit() {
  tag=$1
  A=$2
  B=$3
  echo "STAT tag=$tag" >> $RES
  echo "  A $A" >> $RES
  echo "  B $B" >> $RES
  for h in $OUT/$tag*.hdr; do
    [ -f "$h" ] || continue
    el=$(grep -i '^X-M5-Elapsed' "$h" | tr -d '\r' | awk '{print $2}')
    bb=$(grep -i '^X-M5-Backbone' "$h" | tr -d '\r' | awk '{print $2}')
    dp=$(grep -i '^X-M5-Depth' "$h" | tr -d '\r' | awk '{print $2}')
    cd=$(grep -i '^X-M5-Codec' "$h" | tr -d '\r' | awk '{print $2}')
    fr=$(grep -i '^X-M5-Frames' "$h" | tr -d '\r' | awk '{print $2}')
    fp=$(grep -i '^X-M5-First' "$h" | tr -d '\r' | awk '{print $2}')
    echo "  REQ $(basename $h .hdr) elapsed=$el bb=$bb depth=$dp codec=$cd frames=$fr first=$fp" >> $RES
  done
}

# --- start server ---
nohup ./m5-server model --port 8080 --cwd $V --threads 3 --voice serena \
  --voice-encoder $V/codec_encoder.onnx > $V/srv.log 2>&1 &
ok=0
i=1
while [ $i -le 90 ]; do
  sleep 1
  if curl -s -m 1 http://127.0.0.1:8080/health >/dev/null 2>&1; then ok=1; break; fi
  i=$((i + 1))
done
echo "server ready=$ok after ${i}s" >> $RES
[ "$ok" = "1" ] || { cat $RES; exit 1; }

# warmup：一次请求把 prime 缓存与图都热起来，然后静置
req "warm"; sleep 6

# --- solo: 4 sequential requests ---
A=$(head -1 /proc/stat)
i=1
while [ $i -le 4 ]; do req "solo$i"; i=$((i + 1)); done
B=$(head -1 /proc/stat)
emit "solo" "$A" "$B"
sleep 4

# --- duo: 2 parallel, two rounds ---
r=1
while [ $r -le 2 ]; do
  A=$(head -1 /proc/stat)
  req "duo${r}a" & req "duo${r}b" &
  wait
  B=$(head -1 /proc/stat)
  emit "duo$r" "$A" "$B"
  sleep 4
  r=$((r + 1))
done

echo "--- freqs after ---" >> $RES
for c in 0 1 2 3 4 5 6 7; do
  echo -n "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null) " >> $RES
done
echo "" >> $RES

echo "--- metrics ---" >> $RES
curl -s -m 5 http://127.0.0.1:8080/metrics >> $RES
cat $RES
