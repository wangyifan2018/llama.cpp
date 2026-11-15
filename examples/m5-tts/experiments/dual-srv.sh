#!/system/bin/sh
# 真并行判定：两个独立 server 实例（8080/8081）各带自己的 stage 进程，
# 同时接收请求。若两者 elapsed 都是 3.6 s，则为真并行（吞吐翻倍）；
# 若都变 7 s 左右，则单 HTP0 + CPU 上限使并行退化为分时。
# 同时按进程族拆分 CPU，定位 stage 的 host 侧开销。
V=/data/local/tmp/llama-m5-htp-v73
OUT=$V/dual-out
BODY='{"input":"前方三百米右转，随后靠左行驶。别着急，我们还有时间。","voice":"serena","response_format":"wav"}'
RES=$V/dual-results.txt

export LD_LIBRARY_PATH=$V
export ADSP_LIBRARY_PATH=$V
export GGML_HEXAGON_NHMX=0
export GGML_HEXAGON_OPPOLL=1

rm -rf $OUT; mkdir -p $OUT
: > $RES
cd $V || exit 1

fam_jif() {
  # 求和指定进程名的全部实例
  name=$1; sum=0
  for p in $(pidof $name 2>/dev/null); do
    v=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null)
    sum=$((sum + ${v:-0}))
  done
  echo $sum
}

snap() {
  s=$(fam_jif m5-server)
  b=$(fam_jif backbone-resident)
  d=$(fam_jif depth-resident)
  echo "srv=$s bb=$b dp=$d"
}

req() {
  port=$1; tag=$2
  curl -s -m 300 -o /dev/null -D "$OUT/$tag.hdr" \
    -H 'Content-Type: application/json' -d "$BODY" \
    http://127.0.0.1:$port/v1/audio/speech
}

start() {
  port=$1; log=$2
  nohup ./m5-server model --port $port --cwd $V --threads 3 --voice serena \
    --voice-encoder $V/codec_encoder.onnx > "$log" 2>&1 &
  i=1
  while [ $i -le 90 ]; do
    sleep 1
    if curl -s -m 1 http://127.0.0.1:$port/health >/dev/null 2>&1; then return 0; fi
    i=$((i + 1))
  done
  return 1
}

hdr_line() {
  h=$1
  el=$(grep -i '^X-M5-Elapsed' "$h" | tr -d '\r' | awk '{print $2}')
  bb=$(grep -i '^X-M5-Backbone' "$h" | tr -d '\r' | awk '{print $2}')
  dp=$(grep -i '^X-M5-Depth' "$h" | tr -d '\r' | awk '{print $2}')
  cd=$(grep -i '^X-M5-Codec' "$h" | tr -d '\r' | awk '{print $2}')
  fp=$(grep -i '^X-M5-First' "$h" | tr -d '\r' | awk '{print $2}')
  echo "elapsed=$el bb=$bb depth=$dp codec=$cd first=$fp"
}

echo "== starting two servers ==" >> $RES
start 8080 $V/dualA.log; echo "  A ready=$?" >> $RES
start 8081 $V/dualB.log; echo "  B ready=$?" >> $RES

# warm both, then settle
req 8080 warmA; req 8081 warmB; sleep 8

# --- solo on A, with per-family CPU ---
S0=$(snap)
req 8080 soloA
S1=$(snap)
echo "PHASE soloA  $S0 | $S1" >> $RES
echo "  REQ soloA $(hdr_line $OUT/soloA.hdr)" >> $RES
sleep 5

# --- true parallel: one request to each server ---
S0=$(snap)
req 8080 parA & p1=$!
req 8081 parB & p2=$!
wait $p1 $p2
S1=$(snap)
echo "PHASE parAB  $S0 | $S1" >> $RES
echo "  REQ parA $(hdr_line $OUT/parA.hdr)" >> $RES
echo "  REQ parB $(hdr_line $OUT/parB.hdr)" >> $RES
sleep 5

# --- two on the same server (serialised by render_mutex) for reference ---
S0=$(snap)
req 8080 seq1 & p1=$!
req 8080 seq2 & p2=$!
wait $p1 $p2
S1=$(snap)
echo "PHASE seqAA  $S0 | $S1" >> $RES
echo "  REQ seq1 $(hdr_line $OUT/seq1.hdr)" >> $RES
echo "  REQ seq2 $(hdr_line $OUT/seq2.hdr)" >> $RES

echo "--- metrics A ---" >> $RES
curl -s -m 5 http://127.0.0.1:8080/metrics >> $RES
cat $RES
