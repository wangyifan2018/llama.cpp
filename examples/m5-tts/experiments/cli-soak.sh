#!/system/bin/sh
# Companion to thermal-baseline.sh: is the historical 0.75 -> 0.91 drift thermal,
# or an artefact of the CLI path (fresh process + fresh HTP session per run)?
# 12 back-to-back CLI runs, same text/seed as the A/B scripts, with NPU temps.
cd /data/local/tmp/llama-m5-htp-v73 || exit 1
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
unset M5_KV_PAD M5_HVX_DIRECT_V M5_PREFIX_STATS
T='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
OUT=$D/clisoak-results.txt
R=$D/clisoak-out
rm -rf "$R"; mkdir -p "$R"
: > "$OUT"

temps () {
  cat /sys/class/thermal/thermal_zone25/temp /sys/class/thermal/thermal_zone24/temp \
      /sys/class/thermal/thermal_zone31/temp 2>/dev/null | tr '\n' ' '
}

N=${N:-12}
i=1
while [ "$i" -le "$N" ]; do
  b=$(temps)
  timeout 240 ./m5-engine model --cwd "$D" --voice serena --seed 1234 \
    --codec codec-snake.onnx --chunk 4 --threads 3 \
    --text "$T" --out "$R/run$i.wav" > "$R/run$i.log" 2>&1
  a=$(temps)
  line=$(grep -a '"elapsed_s"' "$R/run$i.log" | tail -1)
  if [ -z "$line" ]; then
    echo "run$i FAILED" >> "$OUT"
  else
    el=$(echo "$line" | sed 's/.*"elapsed_s":\([0-9.]*\).*/\1/')
    fr=$(echo "$line" | sed 's/.*"frames":\([0-9]*\).*/\1/')
    bb=$(echo "$line" | sed 's/.*"backbone_s":\([0-9.]*\).*/\1/')
    dp=$(echo "$line" | sed 's/.*"depth_s":\([0-9.]*\).*/\1/')
    cd=$(echo "$line" | sed 's/.*"codec_s":\([0-9.]*\).*/\1/')
    pf=$(awk -v e="$el" -v f="$fr" 'BEGIN{printf "%.2f", e*1000/f}')
    echo "run$i $line ms_per_frame=$pf temp_before=\"$b\" temp_after=\"$a\"" >> "$OUT"
    echo "run$i $line ms_per_frame=$pf temp_before=\"$b\" temp_after=\"$a\""
  fi
  i=$((i + 1))
done
echo DONE >> "$OUT"
echo DONE
