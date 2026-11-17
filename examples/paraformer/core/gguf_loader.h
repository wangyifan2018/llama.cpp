// Minimal GGUF loader contract used by the Paraformer example.
// The upstream CrispASR loader drags in a large amount of unrelated runtime
// machinery; this shim keeps only the API paraformer.cpp actually calls.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace core_gguf {

using tensor_map = std::map<std::string, ggml_tensor *>;

struct WeightLoad {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    tensor_map tensors;
};

gguf_context * open_metadata(const char * path);
void free_metadata(gguf_context * gctx);
uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t default_val);

// Read a GGUF file into memory once (caller keeps the bytes).
bool read_file(const char * path, std::vector<char> & bytes);

// Build a weight set from in-memory GGUF bytes. Each call creates an
// independent ggml_context + backend buffer and copies tensor data from the
// bytes into `backend`. This is the shared-load primitive: one file read, many
// materialized contexts (host for CPU, tiled device buffer for HTP).
bool load_weights_bytes(const std::vector<char> & bytes, ggml_backend_t backend, const char * model_tag,
                        WeightLoad & out);

// Convenience: read the file then load into `backend` (CLI / single-context path).
bool load_weights(const char * path, ggml_backend_t backend, const char * model_tag, WeightLoad & out);

void free_weights(WeightLoad & wl);

} // namespace core_gguf
