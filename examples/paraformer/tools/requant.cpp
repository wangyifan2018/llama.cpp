// Re-quantize a Paraformer GGUF with ggml's own quantizer.
//
// The published Q8_0 file is the only weight we have offline, so a Q4_K A/B has
// to be made from it: read every tensor, dequantize, and write a new GGUF with
// the 2D weights in the requested type. Rows whose first dimension is not a
// multiple of the type's block size stay F16 (the entry projection is 560 wide).
//
//   llama-paraformer-requant in.gguf out.gguf Q4_K

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool copy_kv(const gguf_context * src, gguf_context * dst) {
    const int64_t n_kv = gguf_get_n_kv(src);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(src, i);
        const gguf_type type = gguf_get_kv_type(src, i);
        switch (type) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (dst, key, gguf_get_val_u8  (src, i)); break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (dst, key, gguf_get_val_i8  (src, i)); break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (dst, key, gguf_get_val_u16 (src, i)); break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (dst, key, gguf_get_val_i16 (src, i)); break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (dst, key, gguf_get_val_u32 (src, i)); break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (dst, key, gguf_get_val_i32 (src, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (dst, key, gguf_get_val_f32 (src, i)); break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (dst, key, gguf_get_val_u64 (src, i)); break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (dst, key, gguf_get_val_i64 (src, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (dst, key, gguf_get_val_f64 (src, i)); break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(dst, key, gguf_get_val_bool(src, i)); break;
            case GGUF_TYPE_STRING:  gguf_set_val_str (dst, key, gguf_get_val_str (src, i)); break;
            case GGUF_TYPE_ARRAY: {
                const gguf_type atype = gguf_get_arr_type(src, i);
                const size_t n = gguf_get_arr_n(src, i);
                if (atype == GGUF_TYPE_STRING) {
                    std::vector<const char *> strs(n);
                    for (size_t j = 0; j < n; ++j)
                        strs[j] = gguf_get_arr_str(src, i, j);
                    gguf_set_arr_str(dst, key, strs.data(), n);
                } else {
                    gguf_set_arr_data(dst, key, atype, gguf_get_arr_data(src, i), n);
                }
                break;
            }
            default:
                std::fprintf(stderr, "requant: unsupported kv type %d for %s\n", (int) type, key);
                return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s in.gguf out.gguf Q4_K|Q5_K|Q8_0|F16\n", argv[0]);
        return 1;
    }
    // Small name table: the only types this tool needs to produce.
    struct { const char * name; enum ggml_type type; } types[] = {
        { "F16", GGML_TYPE_F16 }, { "Q8_0", GGML_TYPE_Q8_0 }, { "Q5_K", GGML_TYPE_Q5_K },
        { "Q4_K", GGML_TYPE_Q4_K }, { "Q6_K", GGML_TYPE_Q6_K },
    };
    enum ggml_type target = GGML_TYPE_COUNT;
    for (const auto & t : types) {
        if (std::strcmp(argv[3], t.name) == 0) {
            target = t.type;
        }
    }
    if (target == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "requant: unknown type %s\n", argv[3]);
        return 2;
    }
    const size_t blck = (size_t) ggml_blck_size(target);

    ggml_context * src_ctx = nullptr;
    gguf_init_params params = { /*no_alloc=*/false, &src_ctx };
    gguf_context * src = gguf_init_from_file(argv[1], params);
    if (!src) {
        std::fprintf(stderr, "requant: cannot read %s\n", argv[1]);
        return 3;
    }

    setvbuf(stdout, nullptr, _IONBF, 0);
    const int64_t n_tensors = gguf_get_n_tensors(src);
    // Generous arena: the quantized copy is written here before the GGUF writer
    // reads it back, so size it from the source plus slack.
    // Worst case is F16 (two bytes per element), so size the arena for that.
    size_t need = 64u << 20;
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_get_tensor(src_ctx, gguf_get_tensor_name(src, i));
        if (t) {
            need += (size_t) ggml_nelements(t) * 2 + 4096;
        }
    }
    ggml_init_params wp = { need, nullptr, /*no_alloc=*/false };
    ggml_context * dst_ctx = ggml_init(wp);
    gguf_context * dst = gguf_init_empty();
    if (!copy_kv(src, dst)) {
        return 4;
    }

    size_t total_in = 0, total_out = 0;
    int quantized = 0, kept = 0;
    std::vector<float> f32;
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_get_tensor(src_ctx, gguf_get_tensor_name(src, i));
        if (!t) {
            std::fprintf(stderr, "requant: missing tensor %s\n", gguf_get_tensor_name(src, i));
            return 5;
        }
        const int64_t ne0 = t->ne[0];
        const int64_t nrows = t->ne[1] > 0 ? t->ne[1] : 1;
        const bool can_quant = ggml_n_dims(t) >= 2 && (ne0 % (int64_t) blck) == 0 && target != GGML_TYPE_F16;
        // Everything that is not quantized keeps its original type, so the
        // runtime sees the exact types it was built against.
        const enum ggml_type type = can_quant ? target : t->type;

        f32.resize((size_t) ggml_nelements(t));
        const ggml_type_traits * tr = ggml_get_type_traits(t->type);
        if (tr && tr->to_float) {
            tr->to_float(t->data, f32.data(), (int64_t) f32.size());
        } else if (t->type == GGML_TYPE_I32) {
            const int32_t * p = (const int32_t *) t->data;
            for (size_t k = 0; k < f32.size(); ++k)
                f32[k] = (float) p[k];
        } else if (t->type == GGML_TYPE_F32) {
            std::memcpy(f32.data(), t->data, f32.size() * sizeof(float));
        } else {
            std::fprintf(stderr, "requant: %s has type %d with no to_float\n", t->name, (int) t->type);
            return 8;
        }

        ggml_tensor * out = ggml_new_tensor(dst_ctx, type, ggml_n_dims(t), t->ne);
        ggml_set_name(out, t->name);
        if (!can_quant) {
            std::memcpy(out->data, t->data, (size_t) ggml_nbytes(t));
            ++kept;
        } else if (type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t *) out->data, (int64_t) f32.size());
            ++kept;
        } else {
            const size_t written = ggml_quantize_chunk(type, f32.data(), out->data, 0, nrows, ne0, nullptr);
            if (written != ggml_nbytes(out)) {
                std::fprintf(stderr, "requant: %s wrote %zu of %zu bytes\n", t->name, written, ggml_nbytes(out));
                return 6;
            }
            ++quantized;
        }
        total_in += (size_t) ggml_nbytes(t);
        total_out += (size_t) ggml_nbytes(out);
        gguf_add_tensor(dst, out);
    }

    if (!gguf_write_to_file(dst, argv[2], false)) {
        std::fprintf(stderr, "requant: cannot write %s\n", argv[2]);
        return 7;
    }
    std::printf("requant: %lld tensors, %d quantized %s, %d kept F16, %.1f MB -> %.1f MB\n", (long long) n_tensors,
                quantized, ggml_type_name(target), kept, total_in / 1048576.0, total_out / 1048576.0);

    gguf_free(dst);
    ggml_free(dst_ctx);
    gguf_free(src);
    ggml_free(src_ctx);
    return 0;
}
