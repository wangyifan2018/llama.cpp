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

bool load_weights(const char * path, ggml_backend_t backend, const char * model_tag, WeightLoad & out);
void free_weights(WeightLoad & wl);

} // namespace core_gguf
