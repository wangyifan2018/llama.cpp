#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo "usage: $0 PATCHED_LLAMA_SOURCE CMAKE_BUILD_DIR" >&2
    exit 2
fi
src=$1
build=$2
here=$(cd "$(dirname "$0")" && pwd)
"${CXX:-c++}" -std=c++17 -O3 -DNDEBUG \
    -I"$src/tools/mtmd" -I"$src/ggml/include" -I"$src/include" \
    "$here/replay.cpp" -L"$build/bin" -lmtmd -lggml-cpu -lggml-base -lggml -lllama \
    -Wl,-rpath,"$build/bin" -o "$build/bin/qwen3tts-spec-replay"
