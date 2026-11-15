#!/bin/bash
# Concurrency probe against the resident m5-server.
# Measures, for concurrency N in 1/2/4 (wav) and 1/2/4/8 (pcm):
#   - per-request time-to-first-byte and total time
#   - batch wall clock (all N finished)
#   - whether throughput holds (audio seconds / second) or collapses
set -u
HOST=${HOST:-127.0.0.1:18080}
TEXT=${TEXT:-"前方三百米右转，随后靠左行驶。别着急，我们还有时间。"}
OUT=${OUT:-${M5_ASSETS:-$HOME/m5-assets}/conc-results.txt}
now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }

req() { # $1=format $2=out prefix
  local fmt=$1 pre=$2
  curl -s --noproxy '*' -o "${pre}.${fmt}" \
       -w '%{time_starttransfer} %{time_total} %{size_download}\n' \
       -H 'Content-Type: application/json' \
       -d "{\"input\":\"$TEXT\",\"voice\":\"serena\",\"response_format\":\"$fmt\"}" \
       -D "${pre}.hdr" "http://$HOST/v1/audio/speech" > "${pre}.tm"
}

batch() { # $1=conc $2=fmt $3=tag
  local n=$1 fmt=$2 tag=$3
  local t0=$(now_ms) i
  for i in $(seq 1 "$n"); do req "$fmt" "/tmp/cc.${tag}.${i}" & done
  wait
  local t1=$(now_ms)
  echo "BATCH tag=$tag conc=$n fmt=$fmt wall_ms=$((t1-t0))" >> "$OUT"
  for i in $(seq 1 "$n"); do
    local tm=$(cat "/tmp/cc.${tag}.${i}.tm")
    local fp=$(grep -i 'X-M5-First-Pcm' "/tmp/cc.${tag}.${i}.hdr" 2>/dev/null | tr -d '\r' | awk '{print $2}')
    local fr=$(grep -i 'X-M5-Frames' "/tmp/cc.${tag}.${i}.hdr" 2>/dev/null | tr -d '\r' | awk '{print $2}')
    local el=$(grep -i 'X-M5-Elapsed' "/tmp/cc.${tag}.${i}.hdr" 2>/dev/null | tr -d '\r' | awk '{print $2}')
    echo "REQ tag=$tag conc=$n idx=$i ttfb_total_bytes=$tm first_pcm=$fp frames=$fr elapsed=$el" >> "$OUT"
  done
}

shasum_of() { shasum -a 256 "$1" | cut -c1-12; }

: > "$OUT"
echo "# $(date) host=$HOST text_len=${#TEXT}" >> "$OUT"

for round in 1 2 3; do
  for n in 1 2 4; do batch "$n" wav "wav${n}r${round}"; done
  for n in 1 2 4 8; do batch "$n" pcm "pcm${n}r${round}"; done
done

# hash check: every wav in a concurrency group must be identical to the solo one
{
  echo "--- wav hashes ---"
  for round in 1 2 3; do
    for n in 1 2 4; do
      for i in $(seq 1 "$n"); do
        printf 'wav%d r%d #%d %s\n' "$n" "$round" "$i" "$(shasum_of "/tmp/cc.wav${n}r${round}.${i}.wav")"
      done
    done
  done
} >> "$OUT"

echo done
