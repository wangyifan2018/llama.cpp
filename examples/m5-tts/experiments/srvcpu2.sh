#!/system/bin/sh
# 常驻 server 的 CPU 记账（第二版）。
# 修正：wait 只等 curl 的 pid，不要等常驻 server 自己。
# 新增：每次请求窗口记录 8 核频率与 governor，用于解释抖动。
V=/data/local/tmp/llama-m5-htp-v73
OUT=$V/srvcpu2-out
BODY='{"input":"前方三百米右转，随后靠左行驶。别着急，我们还有时间。","voice":"serena","response_format":"wav"}'
RES=$V/srvcpu2-results.txt
TH=${THREADS:-3}

export LD_LIBRARY_PATH=$V
export ADSP_LIBRARY_PATH=$V
export GGML_HEXAGON_NHMX=0
export GGML_HEXAGON_OPPOLL=1

rm -rf $OUT; mkdir -p $OUT
: > $RES
cd $V || exit 1

freqs() {
  f=""
  for c in 0 1 2 3 4 5 6 7; do
    f="$f $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null)"
  done
  echo "$f"
}

req() {
  tag=$1
  curl -s -m 300 -o /dev/null -D "$OUT/$tag.hdr" \
    -H 'Content-Type: application/json' -d "$BODY" \
    http://127.0.0.1:8080/v1/audio/speech
}

emit() {
  tag=$1; A=$2; B=$3; FA=$4; FB=$5
  echo "STAT tag=$tag" >> $RES
  echo "  A $A" >> $RES
  echo "  B $B" >> $RES
  echo "  FA$FA" >> $RES
  echo "  FB$FB" >> $RES
  for h in $OUT/$tag*.hdr; do
    [ -f "$h" ] || continue
    el=$(grep -i '^X-M5-Elapsed' "$h" | tr -d '\r' | awk '{print $2}')
    bb=$(grep -i '^X-M5-Backbone' "$h" | tr -d '\r' | awk '{print $2}')
    dp=$(grep -i '^X-M5-Depth' "$h" | tr -d '\r' | awk '{print $2}')
    cd=$(grep -i '^X-M5-Codec' "$h" | tr -d '\r' | awk '{print $2}')
    fp=$(grep -i '^X-M5-First' "$h" | tr -d '\r' | awk '{print $2}')
    echo "  REQ $(basename $h .hdr) elapsed=$el bb=$bb depth=$dp codec=$cd first=$fp" >> $RES
  done
}

nohup ./m5-server model --port 8080 --cwd $V --threads $TH --voice serena \
  --voice-encoder $V/codec_encoder.onnx > $V/srv2.log 2>&1 &
SRV=$!
ok=0
i=1
while [ $i -le 90 ]; do
  sleep 1
  if curl -s -m 1 http://127.0.0.1:8080/health >/dev/null 2>&1; then ok=1; break; fi
  i=$((i + 1))
done
echo "server ready=$ok threads=$TH after ${i}s" >> $RES
[ "$ok" = "1" ] || { cat $RES; exit 1; }

req "warm"; sleep 8

# --- solo: 4 sequential ---
A=$(head -1 /proc/stat); FA=$(freqs)
i=1
while [ $i -le 4 ]; do req "solo$i"; i=$((i + 1)); done
B=$(head -1 /proc/stat); FB=$(freqs)
emit "solo" "$A" "$B" "$FA" "$FB"
sleep 5

# --- duo: 2 parallel x 2 rounds ---
r=1
while [ $r -le 2 ]; do
  A=$(head -1 /proc/stat); FA=$(freqs)
  req "duo${r}a" & p1=$!
  req "duo${r}b" & p2=$!
  wait $p1 $p2
  B=$(head -1 /proc/stat); FB=$(freqs)
  emit "duo$r" "$A" "$B" "$FA" "$FB"
  sleep 5
  r=$((r + 1))
done

echo "--- metrics ---" >> $RES
curl -s -m 5 http://127.0.0.1:8080/metrics >> $RES
cat $RES
