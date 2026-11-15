#!/bin/bash
# Doc 21.6 item 5: thermal / drift baseline.
# Drives the resident server (the real product path) back to back, sampling the
# NPU / DSP / CPU / DDR thermal zones and CPU clocks around every request.
# Server has no seed knob, so output length varies; we normalise with
# per-frame cost = elapsed_s / frames instead of comparing raw RTF.
set -u
ADB=${ADB:-adb}
BASE=http://127.0.0.1:18080
TEXT='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
OUT=${M5_ASSETS:-$HOME/m5-assets}/thermal-results.txt
TMP=/tmp/thermal-hdr.txt
: > "$OUT"

Z_NPU=/sys/class/thermal/thermal_zone25/temp   # nsphvx-0
Z_AOSS=/sys/class/thermal/thermal_zone24/temp  # aoss-1 (DSP)
Z_CPU=/sys/class/thermal/thermal_zone1/temp    # cpuss-0
Z_DDR=/sys/class/thermal/thermal_zone31/temp   # ddr
Z_SHELL=/sys/class/thermal/thermal_zone40/temp # shell_front
Z_BATT=/sys/class/thermal/thermal_zone44/temp  # Battery

sample () {
  $ADB shell "echo \$(cat $Z_NPU) \$(cat $Z_AOSS) \$(cat $Z_CPU) \$(cat $Z_DDR) \$(cat $Z_SHELL) \$(cat $Z_BATT) \$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) \$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq) \$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq)" </dev/null | tr -d '\r'
}

req () {
  local tag=$1
  local before after wall
  before=$(sample)
  wall=$(curl -s -m 120 --noproxy '*' -D "$TMP" -o /dev/null -w '%{time_total}' \
    -H 'Content-Type: application/json' \
    -d "{\"input\":\"$TEXT\",\"voice\":\"serena\",\"response_format\":\"wav\"}" \
    "$BASE/v1/audio/speech")
  after=$(sample)
  local el fr rtf fp bb
  el=$(grep -i '^X-M5-Elapsed-S:' "$TMP" | tr -d '\r' | awk '{print $2}')
  fr=$(grep -i '^X-M5-Frames:'    "$TMP" | tr -d '\r' | awk '{print $2}')
  rtf=$(grep -i '^X-M5-RTF:'      "$TMP" | tr -d '\r' | awk '{print $2}')
  fp=$(grep -i '^X-M5-First-Pcm-S:' "$TMP" | tr -d '\r' | awk '{print $2}')
  bb=$(grep -i '^X-M5-Backbone-S:' "$TMP" | tr -d '\r' | awk '{print $2}')
  local pf
  pf=$(awk -v e="${el:-0}" -v f="${fr:-0}" 'BEGIN{ if (f>0) printf "%.2f", e*1000/f; else print "NA" }')
  echo "$tag wall=$wall elapsed=${el:-NA} frames=${fr:-NA} rtf=${rtf:-NA} first=${fp:-NA} bb=${bb:-NA} ms_per_frame=$pf temp_before=\"$before\" temp_after=\"$after\"" | tee -a "$OUT"
}

echo "# phase=cold  (idle 120 s, then 6 requests)" | tee -a "$OUT"
echo "idle_start $(sample)" | tee -a "$OUT"
sleep 120
echo "idle_end   $(sample)" | tee -a "$OUT"
for i in $(seq 1 6); do req "cold:$i"; done

echo "# phase=soak  (24 back-to-back requests)" | tee -a "$OUT"
for i in $(seq 1 24); do req "soak:$i"; done

echo "# phase=recover (idle 180 s, then 3 requests)" | tee -a "$OUT"
echo "idle2_start $(sample)" | tee -a "$OUT"
sleep 180
echo "idle2_end   $(sample)" | tee -a "$OUT"
for i in $(seq 1 3); do req "post:$i"; done

echo DONE | tee -a "$OUT"
