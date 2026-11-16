#!/bin/sh
# Build the sherpa-onnx control harness (onnx-bench) for Android arm64.
#
# SHERPA_ANDROID must contain:
#   include/c-api.h            from the sherpa-onnx source tree at the release tag
#   lib/libsherpa-onnx-c-api.so
#   lib/libonnxruntime.so      both from the release android tarball, arm64-v8a
#
# Example for sherpa-onnx v1.13.8:
#   curl -fLO https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-android.tar.bz2
#   tar xf sherpa-onnx-v1.13.8-android.tar.bz2
#   mkdir -p "$SHERPA_ANDROID/lib" && cp jniLibs/arm64-v8a/*.so "$SHERPA_ANDROID/lib/"
#   curl -fL -o "$SHERPA_ANDROID/include/c-api.h" \
#       https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.8/sherpa-onnx/c-api/c-api.h
set -eu

: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to the Android NDK directory}"
SHERPA_ANDROID=${SHERPA_ANDROID:-/tmp/sherpa-android}
OUT=${OUT:-$SHERPA_ANDROID/onnx-bench}

CXX=$(ls -d "$ANDROID_NDK_HOME"/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android34-clang++ | head -1)
[ -x "$CXX" ] || { echo "no aarch64 clang++ under $ANDROID_NDK_HOME" >&2; exit 1; }

"$CXX" -O2 -std=c++17 -static-libstdc++ \
    -I"$SHERPA_ANDROID/include" \
    "$(dirname "$0")/onnx-bench.cpp" \
    -L"$SHERPA_ANDROID/lib" -lsherpa-onnx-c-api -lonnxruntime \
    -o "$OUT"

echo "built $OUT"
