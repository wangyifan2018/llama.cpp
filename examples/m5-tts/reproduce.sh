#!/usr/bin/env bash
# Idempotent rebuild of the M5-TTS model package and device stack.
#
#   ./reproduce.sh check      # 体检：路径、已有产物哈希（不写盘）
#   ./reproduce.sh fetch      # 从 Hugging Face 下模型包到 $ONNX_DIR
#   ./reproduce.sh models     # tokenizer.bin / voices.json / 两个 gguf / depth-weights.f32
#   ./reproduce.sh codec      # codec 三步链 -> codec_decoder_cached.onnx
#   ./reproduce.sh binaries   # build_android.sh + docker_build_stage.sh
#   ./reproduce.sh package    # 把上面产物摆成 device 侧目录结构 $OUT/model
#   ./reproduce.sh release    # 把派生产物打成 Release 附件（可选 --upload）
#   ./reproduce.sh deploy     # push + deploy.sh --no-models + 门禁
#   ./reproduce.sh all        # fetch models codec binaries package deploy
#
# Flags:
#   --force       无视已有哈希，全部重建
#   --no-device   跳过 deploy 里的设备动作（只生成产物）
#   --upload      配合 release：用 gh 上传到 GitHub Release
#   --tag <name>  配合 release：release 标签，默认 m5-v1
#
# 每个产物都有已知哈希：哈希对就跳过，不对就重建。二进制没有哈希（工具链
# 不可逐字节复现），存在即跳过，验它们靠 deploy 阶段的行为门禁。
# 详见 MODEL_PACKAGE.md。
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

# ---------------------------------------------------------------- 路径与旋钮
M5_SRC=${M5_SRC:-$HOME/Desktop/M5-TTS}                 # ASLP-lab/M5-TTS checkout
ONNX_DIR=${ONNX_DIR:-$HOME/m5-assets/android_onnx}     # HF 模型包（M5-TTS-50M-ONNX）
OUT=${OUT:-$HOME/m5-assets/model-package}              # 本脚本的产物目录
VENV=${VENV:-$HOME/m5-assets/pyenv}                    # 转换用 Python 环境
M5_BUILD=${M5_BUILD:-$HOME/m5-assets/build}            # m5-server/m5-engine/libm5engine.so
M5_STAGE=${M5_STAGE:-$HOME/m5-assets/stage}            # backbone-resident + 5 个 .so
M5_ORT_ROOT=${M5_ORT_ROOT:-$HOME/m5-assets/ort-aar}    # 解包的 onnxruntime-android.aar
VDIR=${VDIR:-/data/local/tmp/llama-m5-htp-v73}         # 设备目录
FORWARD_PORT=${FORWARD_PORT:-18080}
HF_MODEL=${HF_MODEL:-ASLP-lab/M5-TTS-50M-ONNX}
REF_WAV=${REF_WAV:-$M5_SRC/assets/aslp_default_voice.wav}
TRANSCRIPT=${TRANSCRIPT:-猪携带该疾病，然后通过蚊子传染给人类。}
PY=${PY:-$VENV/bin/python}
PYTHON=${PYTHON:-}                # 建 venv 用的解释器（见 pick_python）
ADB=${ADB:-$(command -v adb || echo "$HOME/Library/Android/sdk/platform-tools/adb")}

# 转换环境钉版本：量化脚本针对 ORT 1.18 的 calibration API 做了放宽，
# 1.23 上未验证；设备运行时用的是 1.23.2，两者不要混。
# onnxruntime 1.18 / onnx 1.16 只发布到 cp312 的 wheel，Python 3.13 会去源码
# 编译 onnx（几分钟起步、还经常失败），所以下面会主动挑 3.11/3.12。
PIP_PKGS=${PIP_PKGS:-"numpy==1.26.4 onnx==1.16.2 onnxruntime==1.18.1 huggingface_hub"}

# ---------------------------------------------------------------- 已知哈希
H_BB_ONNX=103fc4f5ff86cd19d11d1e45126cf001996824e6db40ca3577c9e8c9a330b7df
H_DEPTH_ONNX=44e8be816445e4574007628cfbf5af6c088f5f5c0b37218d112c6b333d739b82
H_ENCODER=ae0364957dd52df283e88baea8ea09d1f1f34b579cd53d2806b38290ebc03b57
H_TEXT_EMB=39c3849c8bbad6c94bb8e33784453d0af069335ac85da353419d7b41144afdb5
H_AUDIO_EMB=5c0e175c1feae525d66931dd81d727cb4f244f70121504c5d014f250afc51551
H_MANIFEST=17818a7d9e695c3953cdb5b05302df103981c94815b937a4db5c553067901dcd
H_BB_GGUF=379a89db4c03245720ad1edd834a09c468e7c861821f940cb79c5458cf2808c2
H_DEPTH_GGUF=45ae171494281f5466a11da9cd29b9e4508322125929b1a9117984630a993fc5
H_DEPTH_F32=9e38455b6180c1db3f56e73d35f78955a5fc2aa30a71297e427def131f2e4bc2
H_TOKENIZER=e3edd1c94d1cbcfa2af2dfe2bec21a60279e6c6640cd6d7160ed1b2e40458c42
H_CODEC=ac3045a87ef9d485701977f8187eb0e27adc0d029488bc717c5b8da4a016f252

GATE_TEXT=${GATE_TEXT:-今天的天气很好，我们一起去公园散步，看看路边的花，听听树上的鸟叫。}

FORCE=0
NO_DEVICE=0
CMD=""

# ---------------------------------------------------------------- 小工具
log()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m  x %s\033[0m\n' "$*" >&2; exit 1; }

sha() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else sha256sum "$1" | cut -d' ' -f1; fi
}

# matches <file> <expected-sha> -> 0 when present and identical
matches() { [ -f "$1" ] && [ "$(sha "$1")" = "$2" ]; }

# need <file> <expected-sha> <label>; prints why it rebuilds, returns 0 to build
need() {
    local f=$1 want=$2 label=$3 got
    if [ "$FORCE" = 1 ]; then note "$label: forced"; return 0; fi
    if [ ! -f "$f" ]; then note "$label: missing"; return 0; fi
    got=$(sha "$f")
    if [ "$got" = "$want" ]; then note "$label: ok ($want)"; return 1; fi
    warn "$label: hash mismatch"
    note "  want $want"
    note "  got  $got"
    return 0
}

need_file() {
    local f=$1 label=$2
    if [ "$FORCE" = 1 ]; then note "$label: forced"; return 0; fi
    if [ -f "$f" ]; then note "$label: present (no hash check)"; return 1; fi
    note "$label: missing"; return 0
}

pick_python() {
    [ -n "$PYTHON" ] && return 0
    local c
    for c in python3.12 python3.11 python3; do
        command -v "$c" >/dev/null 2>&1 || continue
        if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
            PYTHON=$c
            return 0
        fi
    done
    die "need Python 3.11/3.12 (onnx 1.16 / onnxruntime 1.18 ship no cp313 wheel); set PYTHON=python3.12"
}

require_py() {
    if [ ! -x "$PY" ]; then
        pick_python
        log "creating python env at $VENV ($("$PYTHON" -c 'import sys;print(sys.version.split()[0])'))"
        "$PYTHON" -m venv "$VENV" || die "python -m venv failed for $PYTHON"
        # shellcheck disable=SC2086
        "$VENV/bin/pip" install -q $PIP_PKGS || {
            warn "pip install failed; check PIP_INDEX_URL / network, then re-run"
            note "packages: $PIP_PKGS"
            note "mirror blocked? try: PIP_INDEX_URL=https://pypi.org/simple $0 $CMD"
            exit 1
        }
    fi
    "$PY" -c 'import numpy, onnx, onnxruntime' 2>/dev/null || {
        warn "python env is missing numpy/onnx/onnxruntime; installing"
        # shellcheck disable=SC2086
        "$VENV/bin/pip" install -q $PIP_PKGS || die "pip install failed; packages: $PIP_PKGS"
    }
    "$PY" -c 'import numpy, onnx, onnxruntime' || die "python env at $VENV is unusable"
}

require_onnx_dir() {
    [ -f "$ONNX_DIR/backbone.onnx" ] && [ -f "$ONNX_DIR/depth_frame.onnx" ] || {
        warn "model bundle not found in $ONNX_DIR"
        note "run: $0 fetch     (hf download $HF_MODEL)"
        exit 1
    }
}

# ---------------------------------------------------------------- check
cmd_check() {
    log "paths"
    note "upstream code : $M5_SRC"
    if [ -d "$M5_SRC/.git" ]; then
        note "  commit      : $(git -C "$M5_SRC" rev-parse --short HEAD 2>/dev/null)"
        note "  dirty files : $(git -C "$M5_SRC" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    else
        warn "  not a git checkout"
    fi
    note "model bundle  : $ONNX_DIR"
    note "artifacts     : $OUT"
    note "python env    : $VENV"
    note "native build  : $M5_BUILD"
    note "stage build   : $M5_STAGE"
    note "ort aar       : $M5_ORT_ROOT"
    note "device dir    : $VDIR"
    note "adb           : $ADB"

    log "bundle"
    local pair
    for pair in "backbone.onnx:$H_BB_ONNX" "depth_frame.onnx:$H_DEPTH_ONNX" \
                "codec_encoder.onnx:$H_ENCODER" "text_embeddings.npy:$H_TEXT_EMB" \
                "audio_embeddings.npy:$H_AUDIO_EMB" "manifest.json:$H_MANIFEST"; do
        local name=${pair%%:*} want=${pair##*:}
        if [ ! -f "$ONNX_DIR/$name" ]; then note "$name: MISSING"
        elif [ "$(sha "$ONNX_DIR/$name")" = "$want" ]; then note "$name: ok"
        else warn "$name: hash mismatch (bundle is a different revision?)"; fi
    done

    log "derived"
    for pair in "backbone-Q8_0.gguf:$H_BB_GGUF" "depth-Q4_0.gguf:$H_DEPTH_GGUF" \
                "depth-weights.f32:$H_DEPTH_F32" "tokenizer.bin:$H_TOKENIZER"; do
        local name=${pair%%:*} want=${pair##*:}
        if matches "$OUT/$name" "$want"; then note "$name: ok"
        elif [ -f "$OUT/$name" ]; then warn "$name: hash mismatch"
        else note "$name: missing"; fi
    done
    if matches "$OUT/codec_decoder_cached.onnx" "$H_CODEC"; then note "codec_decoder_cached.onnx: ok (four-step chain applied)"
    elif [ -f "$OUT/codec_decoder_cached.onnx" ]; then warn "codec_decoder_cached.onnx: hash mismatch (missing snake_fuse?)"
    else note "codec_decoder_cached.onnx: missing"; fi

    log "binaries"
    for f in "$M5_BUILD/m5-server" "$M5_BUILD/m5-engine" "$M5_BUILD/libm5engine.so" \
             "$M5_STAGE/backbone-resident" "$M5_STAGE/libggml-htp-v73.so"; do
        if [ -f "$f" ]; then note "$(basename "$f"): present"; else note "$(basename "$f"): missing"; fi
    done
}

# ---------------------------------------------------------------- fetch
cmd_fetch() {
    require_py
    if [ -f "$ONNX_DIR/depth_frame.onnx" ] && [ ! "$FORCE" = 1 ]; then
        note "bundle already in $ONNX_DIR (use --force to re-download)"
        return 0
    fi
    log "downloading $HF_MODEL -> $ONNX_DIR"
    "$PY" - "$HF_MODEL" "$ONNX_DIR" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
rev = os.environ.get("HF_REVISION") or None
os.makedirs(dest, exist_ok=True)
snapshot_download(repo_id=repo, local_dir=dest, revision=rev)
print("downloaded to", dest, "revision", rev or "default")
PY
    log "verifying bundle"
    local pair
    for pair in "backbone.onnx:$H_BB_ONNX" "depth_frame.onnx:$H_DEPTH_ONNX" \
                "codec_encoder.onnx:$H_ENCODER" "manifest.json:$H_MANIFEST"; do
        local name=${pair%%:*} want=${pair##*:}
        [ "$(sha "$ONNX_DIR/$name")" = "$want" ] || die "$name does not match the reference revision"
        note "$name: ok"
    done
}

# ---------------------------------------------------------------- models
cmd_models() {
    require_onnx_dir
    require_py
    mkdir -p "$OUT"

    if need "$OUT/tokenizer.bin" "$H_TOKENIZER" "tokenizer.bin + voices.json"; then
        log "prepare_android_voice.py"
        "$PY" "$HERE/prepare_android_voice.py" "$M5_SRC" "$ONNX_DIR" "$REF_WAV" "$TRANSCRIPT" "$OUT"
    fi

    if need "$OUT/backbone-Q8_0.gguf" "$H_BB_GGUF" "backbone-Q8_0.gguf"; then
        log "convert_backbone.py --quant Q8_0"
        PYTHONPATH="$REPO/gguf-py" "$PY" "$HERE/convert_backbone.py" --quant Q8_0 \
            "$ONNX_DIR/backbone.onnx" "$OUT/backbone-Q8_0.gguf"
    fi

    if need "$OUT/depth-Q4_0.gguf" "$H_DEPTH_GGUF" "depth-Q4_0.gguf"; then
        log "convert_backbone.py --depth --quant Q4_0"
        PYTHONPATH="$REPO/gguf-py" "$PY" "$HERE/convert_backbone.py" --depth --quant Q4_0 \
            "$ONNX_DIR/depth_frame.onnx" "$OUT/depth-Q4_0.gguf"
    fi

    if need "$OUT/depth-weights.f32" "$H_DEPTH_F32" "depth-weights.f32"; then
        log "pipe_depth.py --android"
        "$PY" "$HERE/pipe_depth.py" --android "$ONNX_DIR/depth_frame.onnx" "$OUT/depth-weights.f32"
    fi
}

# ---------------------------------------------------------------- codec
cmd_codec() {
    require_onnx_dir
    require_py
    mkdir -p "$OUT"
    [ -f "$ONNX_DIR/codec_decoder_cached.onnx" ] || die "bundle has no codec_decoder_cached.onnx in $ONNX_DIR"
    [ -f "$OUT/voices.json" ] || die "voices.json missing: run '$0 models' first (calibration needs codes)"

    if ! need "$OUT/codec_decoder_cached.onnx" "$H_CODEC" "codec_decoder_cached.onnx"; then
        return 0
    fi

    # 四步，顺序不能换：先量化 Conv，再把 ConvTranspose 折成 MatMul，
    # 然后单独补一遍 MatMul 量化，最后融合 Snake。中间产物的体积是指纹：
    # 456,464,460 -> 362,056,576 -> 281,959,623 -> 140,865,832 -> 140,935,716
    log "1/4 quantize_codec_static.py (Conv/ConvTranspose QDQ, last 4 Conv in FP32)"
    "$PY" "$HERE/quantize_codec_static.py" "$ONNX_DIR/codec_decoder_cached.onnx" \
        "$OUT/codec-qdq-ex4.onnx" --meta "$ONNX_DIR/codec_cached.json" \
        --voices "$OUT/voices.json" --exclude-last 4

    log "2/4 decompose_convtranspose.py (ConvTranspose -> MatMul)"
    "$PY" "$HERE/decompose_convtranspose.py" \
        "$OUT/codec-qdq-ex4.onnx" "$OUT/codec-ct.onnx"

    log "3/4 quantize_codec_static.py --ops MatMul (QDQ per-channel INT8)"
    "$PY" "$HERE/quantize_codec_static.py" "$OUT/codec-ct.onnx" "$OUT/codec-ctmm-q.onnx" \
        --meta "$ONNX_DIR/codec_cached.json" --voices "$OUT/voices.json" --ops MatMul

    log "4/4 snake_fuse.py (29 Snake chains -> m5::Snake)"
    "$PY" "$HERE/snake_fuse.py" "$OUT/codec-ctmm-q.onnx" "$OUT/codec_decoder_cached.onnx"

    if ! matches "$OUT/codec_decoder_cached.onnx" "$H_CODEC"; then
        warn "codec_decoder_cached.onnx does not match the reference hash"
        note "  this is not fatal (different onnx/ort versions renumber nodes),"
        note "  but check all four steps ran and the size is 140,935,716 bytes"
    fi
}

# ---------------------------------------------------------------- binaries
cmd_binaries() {
    local rebuilt=0
    if need_file "$M5_BUILD/m5-server" "m5-server"; then
        log "build_android.sh -> $M5_BUILD"
        M5_ORT_ROOT="$M5_ORT_ROOT" bash "$HERE/build_android.sh" "$M5_BUILD"
        rebuilt=1
    fi
    if need_file "$M5_STAGE/backbone-resident" "backbone-resident"; then
        log "docker_build_stage.sh -> $M5_STAGE (first run is slow: x86 emulation)"
        bash "$HERE/docker_build_stage.sh" "$M5_STAGE"
        rebuilt=1
    fi
    [ "$rebuilt" = 1 ] || note "binaries: nothing to do"
}

# ---------------------------------------------------------------- package
cmd_package() {
    require_onnx_dir
    log "assembling $OUT/model"
    mkdir -p "$OUT/model"
    local f
    for f in manifest.json text_embeddings.npy audio_embeddings.npy \
             tokenizer.bin voices.json depth-weights.f32 codec_cached.json; do
        [ -f "$OUT/$f" ] || [ -f "$ONNX_DIR/$f" ] || die "missing $f: run '$0 models' first"
    done
    [ -f "$OUT/codec_decoder_cached.onnx" ] || die "missing codec: run '$0 codec' first"

    # runtime files that come straight from the bundle
    cp -f "$ONNX_DIR/manifest.json" "$ONNX_DIR/text_embeddings.npy" \
          "$ONNX_DIR/audio_embeddings.npy" "$ONNX_DIR/codec_cached.json" "$OUT/model/"
    # files we generate
    cp -f "$OUT/tokenizer.bin" "$OUT/voices.json" "$OUT/depth-weights.f32" \
          "$OUT/codec_decoder_cached.onnx" "$OUT/model/"
    # top-level files the stages and the clone endpoint expect
    cp -f "$ONNX_DIR/codec_encoder.onnx" "$OUT/"
    for f in backbone-Q8_0.gguf depth-Q4_0.gguf; do
        [ -f "$OUT/$f" ] || die "missing $f: run '$0 models' first"
    done
    note "$(ls -l "$OUT/model" | tail -n +2 | wc -l | tr -d ' ') files in model/"
}

# ---------------------------------------------------------------- gate
# 设备上 voices.json 的第一个音色名；deploy.sh 用它做 --voice（不然默认 serena
# 在新包上根本不存在，server 会 open 失败）。读不到就返回空。
first_voice() {
    command -v python3 >/dev/null 2>&1 || return 0
    "$ADB" shell "cat $VDIR/model/voices.json" 2>/dev/null \
        | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["voices"][0]["name"])
except Exception:
    pass' 2>/dev/null || true
}

gate() {
    [ "$NO_DEVICE" = 1 ] && { note "gate skipped (--no-device)"; return 0; }
    log "gate: /health + one fixed-sentence wav"
    "$ADB" forward "tcp:$FORWARD_PORT" tcp:8080 >/dev/null 2>&1 || true
    local base="http://127.0.0.1:$FORWARD_PORT"
    local health=""
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        health=$(curl -s -m 3 "$base/health" || true)
        case "$health" in *ok*) break ;; esac
        sleep 1
    done
    case "$health" in
        *ok*) note "health: $health" ;;
        *) die "server did not answer on $base/health (check adb forward and m5-server)" ;;
    esac

    # 第一个音色的名字决定期望值（prepare_android_voice 用参考 wav 的文件名）
    local voice="" exp_frames="" exp_sha=""
    voice=$(first_voice)
    case "$voice" in
        aslp_default_voice|官方样例) exp_frames=94; exp_sha=f6a5f5f48b73e8d8f0e510405bac98d8d6313289faf513ab8b15c46b55a089d5 ;;
        serena) exp_frames=87; exp_sha=10b0bdc1e6e9d8de82d396983fc9292cf94edd16201cdae1c2bbaef33c2108f3 ;;
        "") warn "cannot read the first voice name; hash check disabled" ;;
        *) warn "voice '$voice' has no reference hash; hash check disabled" ;;
    esac
    note "voice: ${voice:-<server default>}"

    local hdr=/tmp/m5-gate-headers.txt wav=/tmp/m5-gate.wav
    local body
    if [ -n "$voice" ]; then
        body=$(printf '{"input":"%s","voice":"%s","response_format":"wav"}' "$GATE_TEXT" "$voice")
    else
        body=$(printf '{"input":"%s","response_format":"wav"}' "$GATE_TEXT")
    fi
    curl -s -m 120 -D "$hdr" -o "$wav" -H 'Content-Type: application/json' -d "$body" "$base/v1/audio/speech"

    local frames elapsed rtf first
    frames=$(sed -n 's/.*X-M5-Frames: *\([0-9]*\).*/\1/p' "$hdr" | tr -d '\r')
    elapsed=$(sed -n 's/.*X-M5-Elapsed-S: *\([0-9.]*\).*/\1/p' "$hdr" | tr -d '\r')
    rtf=$(sed -n 's/.*X-M5-RTF: *\([0-9.]*\).*/\1/p' "$hdr" | tr -d '\r')
    first=$(sed -n 's/.*X-M5-First-Pcm-S: *\([0-9.]*\).*/\1/p' "$hdr" | tr -d '\r')
    [ -f "$wav" ] || die "no wav returned (see $hdr)"

    note "frames=$frames elapsed=${elapsed}s rtf=$rtf first_pcm=${first}s"
    note "wav sha256: $(sha "$wav")"

    if [ -n "$exp_frames" ] && [ "$frames" != "$exp_frames" ]; then
        warn "expected $exp_frames frames for voice '$voice', got '$frames'"
        note "  frames differ => codes differ => weights, quantisation or codec model changed"
    fi
    if [ -n "$exp_sha" ]; then
        if [ "$(sha "$wav")" = "$exp_sha" ]; then
            note "wav is byte-identical to the reference build"
        else
            warn "wav differs from the reference build (listening may still be fine)"
            note "  check: codes (gguf/voices), codec chain (3 steps), threads, seed"
        fi
    fi
    case "$rtf" in
        "") note "no RTF header (streaming response?)" ;;
        *) awk -v v="$rtf" 'BEGIN { if (v + 0 > 1.0) { printf "  ! RTF %s > 1: check the codec chain and --threads\n", v } }' ;;
    esac
}

# ---------------------------------------------------------------- deploy
cmd_deploy() {
    [ "$NO_DEVICE" = 1 ] && { note "deploy skipped (--no-device)"; return 0; }
    [ -x "$ADB" ] || die "adb not found; set ADB="
    "$ADB" get-state >/dev/null 2>&1 || die "no device attached (adb devices)"
    [ -f "$OUT/model/manifest.json" ] || die "package missing: run '$0 package' first"

    log "pushing model package to $VDIR"
    "$ADB" shell "mkdir -p $VDIR/model" >/dev/null
    "$ADB" push "$OUT/model/." "$VDIR/model/" >/dev/null
    "$ADB" push "$OUT/backbone-Q8_0.gguf" "$OUT/depth-Q4_0.gguf" "$OUT/codec_encoder.onnx" "$VDIR/" >/dev/null

    log "deploying binaries (deploy.sh --no-models)"
    # deploy.sh 的默认音色是 serena；新包里没有它，server 会 open 失败。
    # 用包里实际存在的第一个音色（读不到就交给 server 的默认 voice 0）。
    local voice
    voice=$(first_voice)
    note "default voice: ${voice:-<server default>}"
    M5_BUILD="$M5_BUILD" M5_STAGE="$M5_STAGE" M5_DEFAULT_VOICE="$voice" \
        bash "$HERE/deploy.sh" "$VDIR" --no-models

    gate
}

# ---------------------------------------------------------------- release
# 只打包「HF 上没有、必须由本脚本生成」的东西：codec 四步链产物，和三个从
# HF 权重转出来的运行时文件。仓库本身不放二进制——git 单文件 >100 MiB 直接
# 拒收，LFS 每 clone 一次都算 1.3 GB 流量，而 Release 附件不限总量与带宽
# （单文件上限 2 GB）。理由与取舍见 MODEL_PACKAGE.md §8。
RELEASE_DIR=${RELEASE_DIR:-$OUT/release}
RELEASE_TAG=${RELEASE_TAG:-m5-v1}
UPLOAD=0

fsize() { wc -c < "$1" | tr -d ' '; }

# archive <base> <tar-args...>；中途可以带 -C，两个目录的文件能进同一个包
archive() {
    local base=$1 out; shift
    mkdir -p "$RELEASE_DIR"
    if command -v zstd >/dev/null 2>&1; then
        out=$RELEASE_DIR/$base.tar.zst
        tar -cf - "$@" | zstd -q -19 -T0 -o "$out" -f
    else
        out=$RELEASE_DIR/$base.tar.gz
        tar -czf "$out" "$@"
    fi
    printf '%s\n' "$out"
}

# 一行哈希：<sha256>  <name>  <bytes>  <reference|drift>
sum_line() {
    local f=$1 want=${2:-} name
    name=$(basename "$f")
    if [ -n "$want" ] && [ "$(sha "$f")" = "$want" ]; then
        printf '%s  %s  %s  reference\n' "$(sha "$f")" "$name" "$(fsize "$f")"
    elif [ -n "$want" ]; then
        printf '%s  %s  %s  drift\n' "$(sha "$f")" "$name" "$(fsize "$f")"
    else
        printf '%s  %s  %s  -\n' "$(sha "$f")" "$name" "$(fsize "$f")"
    fi
}

cmd_release() {
    local model=$OUT/model
    [ -f "$model/codec_decoder_cached.onnx" ] || die "missing codec: run '$0 codec' first"
    local f
    for f in "$model/depth-weights.f32" "$model/tokenizer.bin" "$model/voices.json" \
             "$OUT/backbone-Q8_0.gguf" "$OUT/depth-Q4_0.gguf"; do
        [ -f "$f" ] || die "missing $(basename "$f"): run '$0 models' first"
    done

    log "bundling $RELEASE_TAG -> $RELEASE_DIR"
    local codec_arc models_arc
    codec_arc=$(archive "m5-codec-int8-$RELEASE_TAG" -C "$model" codec_decoder_cached.onnx)
    models_arc=$(archive "m5-models-q4-$RELEASE_TAG" \
        -C "$OUT" backbone-Q8_0.gguf depth-Q4_0.gguf \
        -C "$model" depth-weights.f32 tokenizer.bin voices.json)
    note "$(basename "$codec_arc"): $(fsize "$codec_arc") bytes"
    note "$(basename "$models_arc"): $(fsize "$models_arc") bytes"

    local commit upstream hf_rev
    commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
    upstream=$(git -C "$M5_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
    hf_rev=${HF_REVISION:-main}

    {
        sum_line "$model/codec_decoder_cached.onnx" "$H_CODEC"
        sum_line "$OUT/backbone-Q8_0.gguf" "$H_BB_GGUF"
        sum_line "$OUT/depth-Q4_0.gguf" "$H_DEPTH_GGUF"
        sum_line "$model/depth-weights.f32" "$H_DEPTH_F32"
        sum_line "$model/tokenizer.bin" "$H_TOKENIZER"
        sum_line "$model/voices.json" ""
    } > "$RELEASE_DIR/SHA256SUMS"

    local drift=0
    awk '$4 == "drift" { n++ } END { exit (n > 0) }' "$RELEASE_DIR/SHA256SUMS" || drift=1

    {
        cat <<EOF
# M5-TTS derived artifacts — $RELEASE_TAG

Only the files that cannot be downloaded are here. Everything else comes from
[Hugging Face](https://huggingface.co/$HF_MODEL); \`reproduce.sh fetch\` verifies
those against the hashes in \`MODEL_PACKAGE.md\`.

Built from:

| | |
|---|---|
| this repo | \`$commit\` |
| upstream code | \`ASLP-lab/M5-TTS\` @ \`$upstream\` |
| weights | \`$HF_MODEL\` @ \`$hf_rev\` |
| toolchain | python 3.12, onnx 1.16.2, onnxruntime 1.18.1 |

| archive | contents | bytes |
|---|---|---|
| \`$(basename "$codec_arc")\` | \`codec_decoder_cached.onnx\` (four-step chain applied) | $(fsize "$codec_arc") |
| \`$(basename "$models_arc")\` | \`backbone-Q8_0.gguf\`, \`depth-Q4_0.gguf\`, \`depth-weights.f32\`, \`tokenizer.bin\`, \`voices.json\` | $(fsize "$models_arc") |

## Install

\`\`\`sh
V=/data/local/tmp/llama-m5-htp-v73
adb shell "mkdir -p \$V/model"
mkdir -p /tmp/m5
tar -xaf $(basename "$codec_arc")  -C /tmp/m5
tar -xaf $(basename "$models_arc") -C /tmp/m5
adb push /tmp/m5/codec_decoder_cached.onnx \$V/model/
adb push /tmp/m5/depth-weights.f32 /tmp/m5/tokenizer.bin /tmp/m5/voices.json \$V/model/
adb push /tmp/m5/backbone-Q8_0.gguf /tmp/m5/depth-Q4_0.gguf \$V/
\`\`\`

\`manifest.json\`, \`text_embeddings.npy\`, \`audio_embeddings.npy\`, \`codec_cached.json\`
and \`codec_encoder.onnx\` still come from the HF bundle — copy them out of
\`\$ONNX_DIR\` by hand, or just let \`reproduce.sh package deploy\` do the whole
layout. Then run the gate: \`reproduce.sh gate\` must report the reference WAV
hash for the bundled voice.

## Hashes

\`\`\`
$(cat "$RELEASE_DIR/SHA256SUMS")
\`\`\`

\`reference\` means byte-identical to the build recorded in \`MODEL_PACKAGE.md\`;
\`drift\` means the pipeline ran but onnx/onnxruntime renumbered or re-rounded
something. Drift is not automatically bad — check the size and run the gate —
but do not mix drifted codec weights with existing \`voices.json\` codes without
re-encoding them.

## Licence

These are derivatives of \`$HF_MODEL\`. That repository ships no LICENSE file;
its README defers to "the terms attached to each release". Confirm you may
redistribute the weights before publishing this release publicly.
EOF
    } > "$RELEASE_DIR/RELEASE_NOTES.md"

    note "hashes:"
    sed 's/^/      /' "$RELEASE_DIR/SHA256SUMS"
    [ "$drift" = 0 ] || warn "at least one artifact drifted from the reference hash (see above)"

    if [ "$UPLOAD" = 1 ]; then
        command -v gh >/dev/null 2>&1 || die "gh not found: install it, or upload manually"
        log "uploading to release '$RELEASE_TAG'"
        if gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
            gh release upload "$RELEASE_TAG" "$codec_arc" "$models_arc" \
                "$RELEASE_DIR/SHA256SUMS" "$RELEASE_DIR/RELEASE_NOTES.md" --clobber
        else
            gh release create "$RELEASE_TAG" \
                --title "M5-TTS derived artifacts $RELEASE_TAG" \
                --notes-file "$RELEASE_DIR/RELEASE_NOTES.md" \
                "$codec_arc" "$models_arc" "$RELEASE_DIR/SHA256SUMS"
        fi
        note "done: $(gh release view "$RELEASE_TAG" --json url -q .url 2>/dev/null || echo "$RELEASE_TAG")"
    else
        note "upload skipped; re-run with --upload, or:"
        note "  gh release create $RELEASE_TAG --notes-file $RELEASE_DIR/RELEASE_NOTES.md \\"
        note "      $codec_arc $models_arc"
    fi
}

# ---------------------------------------------------------------- main
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1 ;;
        --no-device) NO_DEVICE=1 ;;
        --upload) UPLOAD=1 ;;
        --tag) shift; RELEASE_TAG=${1:-}; [ -n "$RELEASE_TAG" ] || die "--tag needs a value" ;;
        -h|--help)
            cat <<'EOF'
usage: reproduce.sh [check|fetch|models|codec|binaries|package|release|deploy|gate|all]
                    [--force] [--no-device] [--upload] [--tag NAME]

  check     体检：路径、bundle 与已有产物的哈希（不写盘）
  fetch     从 Hugging Face 下 ASLP-lab/M5-TTS-50M-ONNX 到 $ONNX_DIR
  models    tokenizer.bin / voices.json / backbone-Q8_0.gguf / depth-Q4_0.gguf / depth-weights.f32
  codec     codec 四步链：Conv QDQ -> CT->MatMul -> MatMul QDQ -> snake_fuse
  binaries  build_android.sh + docker_build_stage.sh
  package   摆成设备侧目录结构 $OUT/model
  release   打 Release 附件（codec + 派生的 gguf/f32），--upload 直接上传
  deploy    push 模型包 + deploy.sh --no-models + 门禁
  gate      只跑门禁（/health + 固定句子的 wav 哈希）
  all       fetch -> models -> codec -> binaries -> package -> deploy

env overrides: M5_SRC ONNX_DIR OUT VENV M5_BUILD M5_STAGE M5_ORT_ROOT VDIR
               REF_WAV TRANSCRIPT PY ADB FORWARD_PORT HF_MODEL
               RELEASE_DIR RELEASE_TAG HF_REVISION
see MODEL_PACKAGE.md
EOF
            exit 0 ;;
        -* ) die "unknown flag: $1" ;;
        *) [ -z "$CMD" ] || die "only one command at a time"; CMD=$1 ;;
    esac
    shift
done
: "${CMD:=check}"

case "$CMD" in
    check)    cmd_check ;;
    fetch)    cmd_fetch ;;
    models)   cmd_models ;;
    codec)    cmd_codec ;;
    binaries) cmd_binaries ;;
    package)  cmd_package ;;
    release)  cmd_release ;;
    deploy)   cmd_deploy ;;
    gate)     gate ;;
    all)
        cmd_fetch
        cmd_models
        cmd_codec
        cmd_binaries
        cmd_package
        cmd_deploy
        ;;
    *) die "unknown command: $CMD (try --help)" ;;
esac
