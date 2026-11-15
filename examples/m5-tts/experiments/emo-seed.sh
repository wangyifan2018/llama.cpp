#!/system/bin/sh
cd /data/local/tmp/llama-m5-htp-v73
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
T="今天天气很好，我们一起去公园散步吧。"
for s in 42 777 2026; do
  timeout 200 ./m5-engine model --text "$T" --seed $s --out $D/emo-seed$s.wav --cwd $D --threads 3 > /dev/null 2>&1
  echo "done seed $s"
done
