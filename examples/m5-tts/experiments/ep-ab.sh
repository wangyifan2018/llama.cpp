#!/bin/bash
# codec execution-provider A/B: ORT default CPU EP vs XNNPACK, CLI path.
# Gate: codec_s / elapsed_s / WAV hash.
set -u
ADB=${ADB:-adb}
V=/data/local/tmp/llama-m5-htp-v73
TEXT=${TEXT:-"前方三百米右转，随后靠左行驶。别着急，我们还有时间。"}
OUT=${OUT:-$V/ep-ab-results.txt}
ROUNDS=${ROUNDS:-4}

run() { # $1=tag $2=ep
  local tag=$1 ep=$2
  $ADB shell "cd $V && LD_LIBRARY_PATH=$V ADSP_LIBRARY_PATH=$V \
    GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1 M5_CODEC_EP=$ep \
    ./m5-engine model --text '$TEXT' --voice serena --out $V/ep-out-$tag.wav 2>$V/ep-log-$tag.txt \
    | grep audio_s" </dev/null | tr -d '\r' | sed "s/^/$tag $ep /"
  $ADB shell "sha256sum $V/ep-out-$tag.wav | cut -c1-16; grep -c 'codec: EP XNNPACK' $V/ep-log-$tag.txt" </dev/null | tr -d '\r' | sed "s/^/$tag /"
}

: > /tmp/ep-ab-local.txt
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="cpu xnn"; else order="xnn cpu"; fi
  for k in $order; do
    if [ "$k" = cpu ]; then ep=CPU; else ep=XNNPACK; fi
    run "r${r}$k" "$ep" | tee -a /tmp/ep-ab-local.txt
  done
done
echo "---"
cat /tmp/ep-ab-local.txt
