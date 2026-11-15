#!/system/bin/sh
cd /data/local/tmp/llama-m5-htp-v73
D=$PWD
export LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
T="今天天气很好，我们一起去公园散步吧。"
for cap in NONE happy sad angry gentle 开心 悲伤; do
  if [ "$cap" = "NONE" ]; then
    timeout 200 ./m5-engine model --text "$T" --out $D/emo-base.wav --cwd $D --threads 3 > $D/emo-$cap.log 2>&1
  else
    timeout 200 ./m5-engine model --text "$T" --caption "$cap" --out $D/emo-$cap.wav --cwd $D --threads 3 > $D/emo-$cap.log 2>&1
  fi
  echo "done $cap rc=$?"
done
