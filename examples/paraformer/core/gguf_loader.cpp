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

bool read_file(const char * path, std::vector<char> & bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "paraformer: cannot open GGUF '%s'\n", path);
        return false;
    }
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        fprintf(stderr, "paraformer: empty GGUF '%s'\n", path);
        return false;
    }
    bytes.resize((size_t) size);
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), size)) {
        fprintf(stderr, "paraformer: short read on '%s'\n", path);
        bytes.clear();
        return false;
    }
    return true;
}

bool load_weights_bytes(const std::vector<char> & bytes, ggml_backend_t backend, const char * model_tag,
                        WeightLoad & out) {
    ggml_context * ctx = nullptr;
    gguf_context * g = gguf_init_from_buffer(bytes.data(), bytes.size(), {true, &ctx});
    if (!g || !ctx) {
        fprintf(stderr, "%s: cannot parse GGUF bytes\n", model_tag);
        if (g)
            gguf_free(g);
        if (ctx)
            ggml_free(ctx);
        return false;
    }

    const size_t file_size = bytes.size();
    const size_t data_offset = (size_t) gguf_get_data_offset(g);
    if (data_offset > file_size) {
        fprintf(stderr, "%s: invalid GGUF data offset\n", model_tag);
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    const size_t data_size = file_size - data_offset;

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
        const size_t nbytes = ggml_nbytes(t);
        if (offset > data_size || nbytes > data_size - offset) {
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

    for (int i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        const size_t offset = (size_t) gguf_get_tensor_offset(g, i);
        const size_t nbytes = ggml_nbytes(t);
        const char * src = bytes.data() + data_offset + offset;
        ggml_backend_tensor_set(t, src, 0, nbytes);
        out.tensors[name] = t;
    }

    gguf_free(g);
    return true;
}

bool load_weights(const char * path, ggml_backend_t backend, const char * model_tag, WeightLoad & out) {
    std::vector<char> bytes;
    if (!read_file(path, bytes))
        return false;
    return load_weights_bytes(bytes, backend, model_tag, out);
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
