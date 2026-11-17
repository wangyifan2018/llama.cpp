#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace core_gguf {

gguf_context * open_metadata(const char * path) {
    ggml_context * ctx = nullptr;
    gguf_context * g = gguf_init_from_file(path, {true, &ctx});
    if (!g) {
        fprintf(stderr, "paraformer: cannot open GGUF '%s'\n", path);
        if (ctx) {
            ggml_free(ctx);
        }
        return nullptr;
    }
    return g;
}

void free_metadata(gguf_context * gctx) {
    if (gctx) {
        gguf_free(gctx);
    }
}

uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t default_val) {
    if (!gctx) {
        return default_val;
    }
    const int id = gguf_find_key(gctx, key);
    if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_UINT32) {
        return default_val;
    }
    return gguf_get_val_u32(gctx, id);
}

bool load_weights(const char * path, ggml_backend_t backend, const char * model_tag, WeightLoad & out) {
    ggml_context * ctx = nullptr;
    gguf_context * g = gguf_init_from_file(path, {true, &ctx});
    if (!g || !ctx) {
        fprintf(stderr, "%s: cannot load GGUF '%s'\n", model_tag, path);
        if (g) {
            gguf_free(g);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        return false;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff file_size = file.tellg();
    const size_t data_offset = (size_t) gguf_get_data_offset(g);
    if (file_size < 0 || data_offset > (size_t) file_size) {
        fprintf(stderr, "%s: invalid GGUF data offset\n", model_tag);
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    const size_t data_size = (size_t) file_size - data_offset;

    for (int i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) {
            fprintf(stderr, "%s: missing tensor '%s'\n", model_tag, name);
            gguf_free(g);
            ggml_free(ctx);
            return false;
        }
        const size_t offset = (size_t) gguf_get_tensor_offset(g, i);
        const size_t bytes = ggml_nbytes(t);
        if (offset > data_size || bytes > data_size - offset) {
            fprintf(stderr, "%s: tensor '%s' is outside the GGUF data section\n", model_tag, name);
            gguf_free(g);
            ggml_free(ctx);
            return false;
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!out.buf) {
        fprintf(stderr, "%s: cannot allocate weight buffer\n", model_tag);
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    out.ctx = ctx;

    std::vector<char> scratch;
    file.seekg(0, std::ios::beg);
    for (int i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        const size_t offset = (size_t) gguf_get_tensor_offset(g, i);
        const size_t bytes = ggml_nbytes(t);
        scratch.resize(bytes);
        file.seekg((std::streamoff) (data_offset + offset), std::ios::beg);
        if (!file.read(scratch.data(), (std::streamsize) bytes)) {
            fprintf(stderr, "%s: cannot read tensor '%s'\n", model_tag, name);
            gguf_free(g);
            free_weights(out);
            return false;
        }
        ggml_backend_tensor_set(t, scratch.data(), 0, bytes);
        out.tensors[name] = t;
    }

    gguf_free(g);
    return true;
}

void free_weights(WeightLoad & wl) {
    if (wl.buf) {
        ggml_backend_buffer_free(wl.buf);
        wl.buf = nullptr;
    }
    if (wl.ctx) {
        ggml_free(wl.ctx);
        wl.ctx = nullptr;
    }
    wl.tensors.clear();
}

} // namespace core_gguf
