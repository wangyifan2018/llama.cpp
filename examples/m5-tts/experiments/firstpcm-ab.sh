#!/system/bin/sh
# Break down first-packet latency with existing switches only (no rebuild).
# Alternating rounds cancel thermal drift. --verbose gives step-0 timestamps;
# M5_PREFIX_STATS shows how many prefix rows were cached.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
export M5_PREFIX_STATS=1 M5_HEAD_STATS=1
OUT=$D/firstpcm-results.txt
R=$D/firstpcm-out
mkdir -p "$R"
: > "$OUT"

T='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
TS='你好，世界。'
TL='今天的天气很好，我们一起去公园散步，看看路边的花，听听树上的鸟叫。等太阳落山，我们再一起慢慢走回家，路上还能买点水果。'

run () {
  tag=$1; text=$2; shift 2
  timeout 240 ./m5-engine model --cwd "$D" --voice serena --seed 1234 \
    --codec codec-snake.onnx --threads 3 --verbose "$@" \
    --text "$text" --out "$R/$tag.wav" > "$R/$tag.log" 2>&1 || {
      echo "$tag FAILED" >> "$OUT"; echo "$tag FAILED"; tail -6 "$R/$tag.log"; return; }
  line=$(grep -a '"audio_s"' "$R/$tag.log" | tail -1)
  [ -z "$line" ] && line="FAILED"
  v0=$(grep -a 'step 0 t=' "$R/$tag.log" | head -1)
  v10=$(grep -a 'step 10 t=' "$R/$tag.log" | head -1)
  pre=$(grep -a 'm5-backbone prefill_ms' "$R/$tag.log" | head -1)
  pfx=$(grep -a 'prefix shared=' "$R/$tag.log" | tail -1)
  hs=$(grep -a 'heads_s=' "$R/$tag.log" | tail -1)
  echo "$tag | $line | V0=$v0 | V10=$v10 | $pre | $pfx | $hs" >> "$OUT"
  echo "$tag | $line | $v0"
  sleep 1
}

for r in 1 2; do
  if [ $((r % 2)) -eq 1 ]; then order="1 2 3 4 6 8"; else order="8 6 4 3 2 1"; fi
  for f in $order; do run "r${r}_f${f}" "$T" --first "$f" --chunk 4; done
done

run "nooverlap_f2" "$T" --first 2 --chunk 4 --no-overlap
run "short_f2" "$TS" --first 2 --chunk 4
run "long_f2" "$TL" --first 2 --chunk 4
run "f2_c2" "$T" --first 2 --chunk 2
run "f2_c8" "$T" --first 2 --chunk 8

echo DONE >> "$OUT"
echo DONE
