#!/usr/bin/env bash
# Build an installable package for the OpenAI-compatible ASR service.
#
#   examples/paraformer/tools/package_device.sh OUTDIR [--check]
#
# Produces a tree that mirrors the on-device layout, a manifest of every file
# with its SHA-256, and an install script. Nothing here needs the phone: it
# assembles files that must already have been built and pushed nowhere.
#
# --check  validate the inputs and the resulting tree, then stop before writing.
#          This is the mode the gate runs, so a broken layout fails before
#          anyone copies it to a car.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
EX=$(cd "$HERE/.." && pwd)
REPO=$(cd "$EX/../.." && pwd)

BUILD=${BUILD:-$REPO/build-android-xasr}
PF_MODELS=${PF_MODELS:-$HOME/models/paraformer}
PF_VAD_FILE=${PF_VAD_FILE:-$PF_MODELS/silero_vad.onnx}
PF_WARMUP_WAV=${PF_WARMUP_WAV:-$PF_MODELS/tingting/hvac_zh01.wav}
CHECK_ONLY=0

OUT=
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        -*) echo "unknown flag $arg" >&2; exit 1 ;;
        *) OUT=$arg ;;
    esac
done

say()  { printf '%s\n' "$*"; }
fail() { printf 'package: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- inputs
BIN=$BUILD/bin/llama-paraformer-api
[ -x "$BIN" ] || fail "missing $BIN (build it first: see PRODUCTION.md)"

# The model is the one artifact that is not built here. Accept the documented
# name and say exactly where it was looked for when it is absent, because a
# missing weight is the most common failure and "no such file" alone does not
# tell you which path was tried.
MODEL=$PF_MODELS/paraformer-zh-q8_0.gguf
[ -r "$MODEL" ] || fail "missing model $MODEL (set PF_MODELS to the directory that holds paraformer-zh-q8_0.gguf)"
python3 "$HERE/fetch_models.py" --out "$PF_MODELS" --only paraformer-zh-q8_0.gguf --verify-only
[ -r "$PF_VAD_FILE" ] || fail "missing Silero VAD file $PF_VAD_FILE (set PF_VAD_FILE)"
[ -r "$PF_WARMUP_WAV" ] || fail "missing warmup WAV $PF_WARMUP_WAV (set PF_WARMUP_WAV)"
VAD_SHA=$(sha256sum "$PF_VAD_FILE" | awk '{print $1}')
[ "$VAD_SHA" = 2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f ] || fail "unexpected Silero VAD SHA-256"

# Libraries: the ggml set plus the DSP skeleton. The skeleton is named per arch
# and lives next to the ARM64 library of the same name, so both go in one
# directory and the launcher points ADSP_LIBRARY_PATH at it.
shopt -s nullglob
GGML_LIBS=("$BUILD"/bin/libggml*.so)
shopt -u nullglob
[ ${#GGML_LIBS[@]} -gt 0 ] || fail "no libggml*.so in $BUILD/bin"

# The DSP skeleton is not a link-time dependency of the binary: the adsp daemon
# loads it by name at session open, so it is absent from NEEDED and a package
# without it starts fine and then fails on the DSP with 0x80000406. It is also
# not in bin/ -- the build puts it next to the hexagon backend source. Look for
# the v73 target first, because that is what SM6650 reports.
SKEL=$BUILD/ggml/src/ggml-hexagon/libggml-htp-v73.so
CXX=${PF_CXX_SHARED:-$BUILD/ggml/src/ggml-hexagon/ship/libc++_shared.so}
[ -r "$SKEL" ] || fail "missing SM6650 DSP skeleton $SKEL"
[ -r "$CXX" ] || fail "missing C++ runtime $CXX (set PF_CXX_SHARED)"

# Sherpa and ONNX Runtime come from the prebuilt SDK, not from this build tree.
# Look in the places they actually are, in order: an explicit PF_ONNX_DIR, the
# build tree, then a checkout that has been pulled from a device. The last one
# matters because the SDK is only staged on the phone in this repo's workflow.
ONNX_LIBS=()
for cand in "${PF_ONNX_DIR:-}" "$BUILD/onnx/bin" "$REPO/onnx-sdk/bin" /tmp/sherpa/lib; do
    [ -n "$cand" ] || continue
    if [ -r "$cand/libonnxruntime.so" ] && [ -r "$cand/libsherpa-onnx-c-api.so" ]; then
        ONNX_LIBS=("$cand/libonnxruntime.so" "$cand/libsherpa-onnx-c-api.so")
        break
    fi
done
if [ ${#ONNX_LIBS[@]} -eq 0 ]; then
    fail "cannot find libonnxruntime.so and libsherpa-onnx-c-api.so. Set PF_ONNX_DIR to their directory, or pull them off the device:
  mkdir -p $REPO/onnx-sdk/bin && adb pull /data/local/tmp/llama.cpp/onnx/bin/. $REPO/onnx-sdk/bin/"
fi

HOTWORDS=$EX/hotwords_vehicle.txt
CONTEXT=$EX/context_vehicle.txt
RULES=$HERE/cabin_text_rules.tsv
[ -r "$HOTWORDS" ] && [ -r "$CONTEXT" ] && [ -r "$RULES" ] || fail "missing the cabin accuracy profile"

# ---------------------------------------------------------------- layout
# Keep the pieces in variables so --check can validate the same names the
# install script will use. A path that only exists in one of the two is the bug
# this indirection prevents.
RC_SRC=$EX/deploy/paraformer-asr.rc
LAUNCHER_SRC=$EX/deploy/paraformer-asr.sh
CONF_SRC=$EX/deploy/paraformer-asr.conf
for f in "$RC_SRC" "$LAUNCHER_SRC" "$CONF_SRC"; do
    [ -r "$f" ] || fail "missing $f"
done

# ---------------------------------------------------------------- check
say "binary   $BIN"
say "model    $MODEL"
say "libs     ${#GGML_LIBS[@]} ggml + v73 skeleton + C++ runtime + ${#ONNX_LIBS[@]} onnx"
say "rc       $RC_SRC"
say "launcher $LAUNCHER_SRC"
say "conf     $CONF_SRC"

# The config names paths that must match this layout. Check the two that a
# mismatch would break silently rather than at startup: the binary and libdir.
grep -q '^PF_BIN=/vendor/bin/llama-paraformer-api$' "$CONF_SRC" \
    || fail "paraformer-asr.conf PF_BIN does not match where this package installs the binary"
grep -q '^PF_LIBDIR=/vendor/lib64/paraformer$' "$CONF_SRC" \
    || fail "paraformer-asr.conf PF_LIBDIR does not match where this package installs the libraries"

if [ "$CHECK_ONLY" = 1 ]; then
    say "check: layout and inputs are consistent"
    exit 0
fi

[ -n "$OUT" ] || fail "no output directory given"

# ---------------------------------------------------------------- write
if [ -e "$OUT" ] && { [ ! -d "$OUT" ] || [ -n "$(ls -A "$OUT")" ]; }; then
    fail "output must be a new or empty directory: $OUT"
fi
mkdir -p "$OUT/system/vendor/bin" \
         "$OUT/system/vendor/lib64/paraformer" \
         "$OUT/system/vendor/etc/init" \
         "$OUT/system/vendor/etc/paraformer" \
         "$OUT/data/vendor/paraformer"

cp "$LAUNCHER_SRC" "$OUT/system/vendor/bin/paraformer-asr"
chmod 0755 "$OUT/system/vendor/bin/paraformer-asr"

cp "$BIN" "$OUT/system/vendor/bin/llama-paraformer-api"
chmod 0755 "$OUT/system/vendor/bin/llama-paraformer-api"

cp "${GGML_LIBS[@]}" "$SKEL" "$CXX" "${ONNX_LIBS[@]}" "$OUT/system/vendor/lib64/paraformer/"
chmod 0644 "$OUT/system/vendor/lib64/paraformer/"*.so

cp "$RC_SRC" "$OUT/system/vendor/etc/init/paraformer-asr.rc"
chmod 0644 "$OUT/system/vendor/etc/init/paraformer-asr.rc"
cp "$CONF_SRC" "$OUT/system/vendor/etc/paraformer/paraformer-asr.conf"
chmod 0644 "$OUT/system/vendor/etc/paraformer/paraformer-asr.conf"

cp "$MODEL" "$OUT/system/vendor/etc/paraformer/"
chmod 0644 "$OUT/system/vendor/etc/paraformer/paraformer-zh-q8_0.gguf"

assets=$OUT/system/vendor/etc/paraformer
cp "$HOTWORDS" "$CONTEXT" "$RULES" "$assets/"
cp "$PF_VAD_FILE" "$assets/silero_vad.onnx"
cp "$PF_WARMUP_WAV" "$assets/warmup.wav"
chmod 0644 "$assets/$(basename "$HOTWORDS")" "$assets/$(basename "$CONTEXT")" "$assets/$(basename "$RULES")" \
    "$assets/silero_vad.onnx" "$assets/warmup.wav"
cp -R "$EX/web" "$assets/web"
git -C "$REPO" rev-parse HEAD >"$OUT/SOURCE_REVISION"
git -C "$REPO" status --porcelain -- examples/paraformer docs/backend/snapdragon/api-reference.md >"$OUT/SOURCE_STATUS"

# Assert the two things a package can silently get wrong. Both have happened:
# a skeleton that was never copied (runtime 0x80000406 on the DSP, no startup
# error) and a config whose paths do not match the tree. Checking the tree that
# was actually written, not the source paths, is what makes this worth running.
[ -r "$OUT/system/vendor/lib64/paraformer/$(basename "$SKEL")" ] \
    || fail "the skeleton did not land in the package"
[ -r "$OUT/system/vendor/bin/llama-paraformer-api" ] \
    || fail "the server binary did not land in the package"
[ -r "$OUT/system/vendor/etc/init/paraformer-asr.rc" ] \
    || fail "the init rc did not land in the package"
say "verified skeleton, binary and rc are in the tree"

cat >"$OUT/install.sh" <<'INSTALL'
#!/system/bin/sh
# Install this package on the device. Run as root from the directory that holds
# system/ and data/.
#
#   adb push <OUT> /data/local/tmp/pkg && adb shell su -c 'sh /data/local/tmp/pkg/install.sh'
#
# Then reboot, or `setprop vendor.paraformer.restart 1` if the rc is already
# installed. A first install of the .rc needs a reboot: init only reads
# /vendor/etc/init at boot.
set -eu
cd "$(dirname "$0")"

# Verify before copying. This runs on a car image, so a truncated push has to
# fail here rather than as a mysterious DSP error later.
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c SHA256SUMS >/dev/null || { echo "SHA-256 mismatch, refusing to install" >&2; exit 1; }
fi

# Files under system/ shadow the vendor partition. remount may be needed on a
# user build; on a production image this runs from a factory or OTA path.
mount -o rw,remount /vendor 2>/dev/null || true

cp -R system/. /
chmod 0755 /vendor/bin/paraformer-asr /vendor/bin/llama-paraformer-api
chmod 0644 /vendor/etc/init/paraformer-asr.rc /vendor/etc/paraformer/paraformer-asr.conf
chmod 0644 /vendor/lib64/paraformer/*.so
# The service runs as system and must read the model, which is the largest file
# here; keep it group-readable rather than world-readable.
chown root:system /vendor/etc/paraformer/paraformer-zh-q8_0.gguf
chmod 0640 /vendor/etc/paraformer/paraformer-zh-q8_0.gguf

# The launcher creates this, but doing it here with the right ownership avoids
# one boot where the first write fails.
mkdir -p /data/vendor/paraformer
chown system:system /data/vendor/paraformer
chmod 0770 /data/vendor/paraformer

if [ -f /vendor/etc/paraformer/api-key ]; then
    chown root:system /vendor/etc/paraformer/api-key
    chmod 0640 /vendor/etc/paraformer/api-key
fi

echo "installed; reboot to let init pick up the new .rc"
INSTALL
chmod 0755 "$OUT/install.sh"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS )
say "wrote    $OUT/SHA256SUMS"

say "wrote    $OUT/install.sh"
say ""
say "next:"
say "  adb push $OUT /data/local/tmp/pkg"
say "  adb shell su -c 'sh /data/local/tmp/pkg/install.sh'"
say "  adb reboot"
say "  curl -sS localhost:8080/health   # after adb forward tcp:8080 tcp:8080"
