// Adapted from CrispASR. See LICENSE.crispasr and README.md.

#include "xasr.h"

#include "kaldi-fbank.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int graph_nodes = 32768;

using xasr_clock = std::chrono::steady_clock;

double span_ms(xasr_clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(xasr_clock::now() - begin).count();
}

bool timing_requested() {
    static const bool on = std::getenv("XASR_TIMING") != nullptr;
    return on;
}

struct xasr_hparams {
    std::vector<int> n_layers, downsample, n_heads, dims, qd, vd, pd, kernel;
    int pos_dim = 48, decoder_dim = 512, joiner_dim = 512, context_size = 2, vocab = 0, blank_id = 0, feature_dim = 80;
    int unk_id = -1;
    std::vector<int> chunk_ms_table, left_ctx_table;
};

struct conv_mod {
    ggml_tensor *in_w, *in_b, *causal_w, *causal_b, *chunk_w, *chunk_b, *chunk_scale, *out_w, *out_b;
};

struct zlayer {
    ggml_tensor *aw_in_w, *aw_in_b, *aw_pos_w;
    ggml_tensor *sa1_in_w, *sa1_in_b, *sa1_out_w, *sa1_out_b, *sa2_in_w, *sa2_in_b, *sa2_out_w, *sa2_out_b;
    ggml_tensor *ff_in_w[3], *ff_in_b[3], *ff_out_w[3], *ff_out_b[3];
    ggml_tensor *na_in_w, *na_in_b, *na_out_w, *na_out_b;
    conv_mod cv[2];
    ggml_tensor *norm_bias, *bypass, *bypass_mid;
    float norm_scale = 1.0f; // exp(norm.log_scale)
};

struct zstack {
    std::vector<zlayer> layers;
    std::vector<float> ds_w; // softmax(ds_bias); empty when downsample == 1
    ggml_tensor * combiner = nullptr;
};

struct xasr_hotword {
    float bonus;
    std::vector<int32_t> tokens;
};

struct xasr_model {
    xasr_hparams hp;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<std::string> vocab;
    // encoder_embed
    ggml_tensor *conv_w[3], *conv_b[3], *cnx_dw_w, *cnx_dw_b, *cnx_pw1_w, *cnx_pw1_b, *cnx_pw2_w, *cnx_pw2_b;
    ggml_tensor *cnx_pw1_q8 = nullptr, *cnx_pw2_q8 = nullptr; // optional Q8_0 copies of the pointwise projections
    ggml_context * aux_ctx = nullptr;
    ggml_backend_buffer_t aux_buf = nullptr;
    ggml_tensor *emb_out_w, *emb_out_b, *emb_norm_bias;
    float emb_norm_scale = 1.0f;
    std::vector<zstack> stacks;
    std::vector<float> out_ds_w;
    ggml_tensor *enc_proj_w, *enc_proj_b;
    std::vector<xasr_hotword> hotwords;
    // host copies: decoder + joiner
    std::vector<float> dec_emb, dec_conv, dec_proj_w, dec_proj_b, join_out_w, join_out_b;
};

// Joiner projection graph on the selected backend: tanh(enc + pred) is uploaded,
// one quantised matmul produces the logits, and argmax returns the token ids.
struct xasr_joiner {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_tensor * act = nullptr;
    ggml_tensor * pred = nullptr;
    ggml_tensor * idx = nullptr;
    ggml_tensor * raw_idx = nullptr;
    ggml_tensor * token_bonus = nullptr;
    int max_frames = 0;
    bool ready = false;
};

// Chunk geometry, fixed per context.
struct geom {
    int decode_chunk_len = 48, T = 61, chunk = 24, left = 256, tail_pad_ms = 1200;
};

} // namespace

// One stream slot: its own encoder graph, scheduler and state. The model
// weights are shared by every slot through xasr_context::model.
struct xasr_slot {
    int batch_size = 1;
    ggml_cgraph * graph = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
    std::vector<ggml_tensor *> caches;
    // Streaming-cache writes whose value is a strided view of a tensor already in
    // the graph. The tail copy reads it directly, so the value is never
    // materialised into its own contiguous buffer first. See cache_fold().
    std::vector<std::pair<ggml_tensor *, std::string>> folded_caches;
    std::vector<ggml_tensor *> masks;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> constants;
    ggml_tensor * feats = nullptr;
    ggml_tensor * output = nullptr;
    bool busy = false;

    ~xasr_slot() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
    }
};

struct xasr_context {
    xasr_context_params params;
    xasr_model model;
    geom g;
    ggml_backend_t backend = nullptr, backend_cpu = nullptr;
    std::vector<std::unique_ptr<xasr_slot>> slots;
    std::unique_ptr<xasr_slot> batch_two, batch_four;
    std::vector<float> cache_copy;
    xasr_slot * building = nullptr; // slot being built, used by fixed_input
    xasr_joiner joiner;
    void (*dsp_stats)(ggml_backend_t, double *, double *, uint64_t *) = nullptr;
};

namespace {

ggml_tensor * req(xasr_model & m, const std::string & name) {
    auto * tensor = ggml_get_tensor(m.ctx, name.c_str());
    if (!tensor) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return tensor;
}

ggml_tensor * req_shape(xasr_model & m, const std::string & name, std::initializer_list<int> shape, bool f32) {
    auto * tensor = req(m, name);
    size_t axis = 0;
    for (int dim : shape) {
        if (tensor->ne[axis++] != dim) {
            throw std::runtime_error("unexpected shape: " + name);
        }
    }
    while (axis < GGML_MAX_DIMS) {
        if (tensor->ne[axis++] != 1) {
            throw std::runtime_error("unexpected shape: " + name);
        }
    }
    if (f32 && tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("expected F32 tensor: " + name);
    }
    return tensor;
}

uint32_t kv_u32(const gguf_context * g, const char * key, uint32_t fallback) {
    const int64_t id = gguf_find_key(g, key);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32 ? gguf_get_val_u32(g, id) : fallback;
}

std::vector<float> to_f32(ggml_tensor * t) {
    std::vector<float> out((size_t)ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        const auto * traits = ggml_get_type_traits(t->type);
        if (!traits->to_float) {
            throw std::runtime_error(std::string("cannot read tensor: ") + ggml_get_name(t));
        }
        traits->to_float(raw.data(), out.data(), (int64_t)out.size());
    }
    return out;
}

// Quantise a loaded F32 weight into a Q8_0 tensor that lives in the model buffer.
void quantize_into_q8(ggml_tensor * src, ggml_tensor * dst) {
    if (!src || !dst) {
        throw std::runtime_error("missing tensor for the Q8_0 conversion");
    }
    const std::vector<float> data = to_f32(src);
    std::vector<char> quant(ggml_nbytes(dst));
    const int64_t n_per_row = dst->ne[0];
    const int64_t nrows = ggml_nrows(dst);
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, data.data(), quant.data(), 0, nrows, n_per_row, nullptr);
    if (written != quant.size()) {
        throw std::runtime_error("Q8_0 conversion size mismatch for " + std::string(src->name));
    }
    ggml_backend_tensor_set(dst, quant.data(), 0, quant.size());
}

std::vector<int> kv_int_array(gguf_context* g, const char * key) {
    std::vector<int> out;
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY)
        return out;
    const size_t n = gguf_get_arr_n(g, id);
    if (n > 8) {
        return {};
    }
    const enum gguf_type t = gguf_get_arr_type(g, id);
    const void * d = gguf_get_arr_data(g, id);
    for (size_t i = 0; i < n; i++) {
        int64_t value;
        switch (t) {
            case GGUF_TYPE_INT32:  value = static_cast<const int32_t *>(d)[i];  break;
            case GGUF_TYPE_UINT32: value = static_cast<const uint32_t *>(d)[i]; break;
            case GGUF_TYPE_INT64:  value = static_cast<const int64_t *>(d)[i];  break;
            default: return {};
        }
        if (value < 1 || value > 8192) {
            return {};
        }
        out.push_back(static_cast<int>(value));
    }
    return out;
}

std::vector<float> softmax_host(const std::vector<float> & b) {
    std::vector<float> w(b.size());
    const float mx = *std::max_element(b.begin(), b.end());
    float s = 0.0f;
    for (size_t i = 0; i < b.size(); i++)
        s += (w[i] = std::exp(b[i] - mx));
    for (auto & v : w)
        v /= s;
    return w;
}

bool load(xasr_model & m, const char * path, ggml_backend_t backend) {
    gguf_context_ptr metadata(gguf_init_from_file(path, {true, &m.ctx}));
    auto * g = metadata.get();
    if (!g) {
        return false;
    }
    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch < 0 || gguf_get_kv_type(g, arch) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(g, arch), "xasr") != 0) {
        throw std::runtime_error("expected an X-ASR GGUF");
    }
    auto & hp = m.hp;
    hp.n_layers = kv_int_array(g, "xasr.n_layers");
    hp.downsample = kv_int_array(g, "xasr.downsample");
    hp.n_heads = kv_int_array(g, "xasr.n_heads");
    hp.dims = kv_int_array(g, "xasr.dims");
    hp.qd = kv_int_array(g, "xasr.query_head_dim");
    hp.vd = kv_int_array(g, "xasr.value_head_dim");
    hp.pd = kv_int_array(g, "xasr.pos_head_dim");
    hp.kernel = kv_int_array(g, "xasr.conv_kernel");
    hp.chunk_ms_table = kv_int_array(g, "xasr.chunk_ms");
    hp.left_ctx_table = kv_int_array(g, "xasr.left_context_frames");
    hp.pos_dim = (int)kv_u32(g, "xasr.pos_dim", 48);
    hp.decoder_dim = (int)kv_u32(g, "xasr.decoder_dim", 512);
    hp.joiner_dim = (int)kv_u32(g, "xasr.joiner_dim", 512);
    hp.context_size = (int)kv_u32(g, "xasr.context_size", 2);
    hp.vocab = (int)kv_u32(g, "xasr.vocab_size", 0);
    hp.blank_id = (int)kv_u32(g, "xasr.blank_id", 0);
    hp.feature_dim = (int)kv_u32(g, "xasr.feature_dim", 80);
    const uint32_t unk = kv_u32(g, "xasr.unk_id", 0xFFFFFFFFu);
    hp.unk_id = unk == 0xFFFFFFFFu ? -1 : (int)unk;
    // This example implements the released X-ASR-zh-en topology.
    const size_t S = hp.dims.size();
    const std::vector<int> ffn_dims{512, 768, 1536, 2048, 1536, 768};
    if (hp.n_layers != std::vector<int>{2, 2, 4, 5, 4, 2} ||
        kv_int_array(g, "xasr.ffn_dim") != ffn_dims ||
        hp.downsample != std::vector<int>{1, 2, 4, 8, 4, 2} ||
        hp.dims != std::vector<int>{192, 256, 512, 768, 512, 256} ||
        hp.n_heads != std::vector<int>{4, 4, 4, 8, 4, 4} ||
        hp.qd != std::vector<int>(S, 32) || hp.vd != std::vector<int>(S, 12) ||
        hp.pd != std::vector<int>(S, 4) || hp.kernel != std::vector<int>{31, 31, 15, 15, 15, 31} ||
        hp.pos_dim != 48 || hp.decoder_dim != 512 || hp.joiner_dim != 512 ||
        hp.feature_dim != 80 || hp.context_size != 2 || hp.vocab != 5000 ||
        hp.blank_id < 0 || hp.blank_id >= hp.vocab || hp.unk_id < -1 || hp.unk_id >= hp.vocab ||
        hp.chunk_ms_table != std::vector<int>{160, 480, 960, 1920} ||
        hp.left_ctx_table != std::vector<int>{96, 256, 256, 256}) {
        throw std::runtime_error("unsupported X-ASR topology");
    }
    const int64_t tokens = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (tokens < 0 || gguf_get_kv_type(g, tokens) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, tokens) != GGUF_TYPE_STRING || gguf_get_arr_n(g, tokens) != size_t(hp.vocab)) {
        throw std::runtime_error("invalid X-ASR vocabulary");
    }
    for (int i = 0; i < hp.vocab; ++i) {
        m.vocab.emplace_back(gguf_get_arr_str(g, tokens, i));
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto end = file.tellg();
    const size_t data_offset = gguf_get_data_offset(g);
    if (end < 0 || data_offset > static_cast<uint64_t>(end) || gguf_get_n_tensors(g) > 1024) {
        throw std::runtime_error("invalid X-ASR weights file");
    }
    const size_t data_size = static_cast<size_t>(end) - data_offset;
    if (data_size > 768 * size_t(1024 * 1024)) {
        throw std::runtime_error("X-ASR weights exceed the example size limit");
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        auto * tensor = req(m, gguf_get_tensor_name(g, i));
        const size_t offset = gguf_get_tensor_offset(g, i);
        const size_t size = ggml_nbytes(tensor);
        if (offset > data_size || size > data_size - offset ||
            (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_Q8_0)) {
            throw std::runtime_error("invalid X-ASR tensor: " + std::string(tensor->name));
        }
    }
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, backend);
    if (!m.buf) {
        throw std::runtime_error("cannot allocate X-ASR weights");
    }
    ggml_backend_buffer_set_usage(m.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<char> bytes;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        auto * tensor = req(m, gguf_get_tensor_name(g, i));
        const size_t offset = gguf_get_tensor_offset(g, i);
        const size_t size = ggml_nbytes(tensor);
        bytes.resize(size);
        file.seekg(static_cast<std::streamoff>(data_offset + offset));
        if (!file.read(bytes.data(), static_cast<std::streamsize>(size))) {
            throw std::runtime_error("cannot read X-ASR tensor: " + std::string(tensor->name));
        }
        ggml_backend_tensor_set(tensor, bytes.data(), 0, size);
    }
    const int conv_channels[] = {1, 8, 32, 128};
    for (int i = 0; i < 3; i++) {
        m.conv_w[i] = req_shape(m, "emb.conv" + std::to_string(i) + ".weight", {3, 3, conv_channels[i], conv_channels[i + 1]}, true);
        m.conv_b[i] = req_shape(m, "emb.conv" + std::to_string(i) + ".bias", {conv_channels[i + 1]}, true);
    }
    m.cnx_dw_w = req_shape(m, "emb.cnx.dw.weight", {7, 7, 1, 128}, true);
    m.cnx_dw_b = req_shape(m, "emb.cnx.dw.bias", {128}, true);
    m.cnx_pw1_w = req_shape(m, "emb.cnx.pw1.weight", {1, 1, 128, 384}, true);
    m.cnx_pw1_b = req_shape(m, "emb.cnx.pw1.bias", {384}, true);
    m.cnx_pw2_w = req_shape(m, "emb.cnx.pw2.weight", {1, 1, 384, 128}, true);
    m.cnx_pw2_b = req_shape(m, "emb.cnx.pw2.bias", {128}, true);
    // Optional Q8_0 copies of the two ConvNeXt pointwise projections. They are
    // F32 in the GGUF and run at 6 to 10 GMAC/s on the DSP, against 40 GMAC/s
    // for the quantised matmuls. They live in their own context and buffer
    // because the GGUF context is sized exactly for the file.
    const char * cnx_q8_env = getenv("XASR_CNX_Q8");
    if (!cnx_q8_env || atoi(cnx_q8_env) != 0) {
        ggml_init_params aux = { ggml_tensor_overhead() * 8, nullptr, true };
        m.aux_ctx = ggml_init(aux);
        if (!m.aux_ctx) {
            throw std::runtime_error("cannot allocate the quantised ConvNeXt context");
        }
        m.cnx_pw1_q8 = ggml_new_tensor_2d(m.aux_ctx, GGML_TYPE_Q8_0, 128, 384);
        ggml_set_name(m.cnx_pw1_q8, "emb.cnx.pw1.q8");
        m.cnx_pw2_q8 = ggml_new_tensor_2d(m.aux_ctx, GGML_TYPE_Q8_0, 384, 128);
        ggml_set_name(m.cnx_pw2_q8, "emb.cnx.pw2.q8");
        m.aux_buf = ggml_backend_alloc_ctx_tensors(m.aux_ctx, backend);
        if (!m.aux_buf) {
            throw std::runtime_error("cannot allocate the quantised ConvNeXt weights");
        }
        quantize_into_q8(m.cnx_pw1_w, m.cnx_pw1_q8);
        quantize_into_q8(m.cnx_pw2_w, m.cnx_pw2_q8);
        fprintf(stderr, "xasr: ConvNeXt pointwise projections converted to Q8_0, %zu bytes\n",
                (size_t) ggml_nbytes(m.cnx_pw1_q8) + (size_t) ggml_nbytes(m.cnx_pw2_q8));
    }

    m.emb_out_w = req_shape(m, "emb.out.weight", {2432, 192}, false);
    m.emb_out_b = req_shape(m, "emb.out.bias", {192}, true);
    m.emb_norm_bias = req_shape(m, "emb.norm.bias", {192}, true);
    m.emb_norm_scale = std::exp(to_f32(req_shape(m, "emb.norm.log_scale", {1}, true))[0]);

    m.stacks.resize(S);
    for (size_t s = 0; s < S; s++) {
        auto & st = m.stacks[s];
        const int dim = hp.dims[s];
        const int heads = hp.n_heads[s];
        const int value = heads * hp.vd[s];
        const int query = heads * (2 * hp.qd[s] + hp.pd[s]);
        const int hidden = 3 * dim / 4;
        const int kernel = hp.kernel[s];
        const std::string sp = "z." + std::to_string(s) + ".";
        if (hp.downsample[s] > 1) {
            st.ds_w = softmax_host(to_f32(req_shape(m, sp + "ds_bias", {hp.downsample[s]}, true)));
            st.combiner = req_shape(m, sp + "combiner", {dim}, true);
        }
        st.layers.resize((size_t)hp.n_layers[s]);
        for (int l = 0; l < hp.n_layers[s]; l++) {
            auto & L = st.layers[(size_t)l];
            const std::string p = sp + std::to_string(l) + ".";
            L.aw_in_w = req_shape(m, p + "aw.in.weight", {dim, query}, false);
            L.aw_in_b = req_shape(m, p + "aw.in.bias", {query}, true);
            L.aw_pos_w = req_shape(m, p + "aw.pos.weight", {48, heads * hp.pd[s]}, false);
            L.sa1_in_w = req_shape(m, p + "sa1.in.weight", {dim, value}, false);
            L.sa1_in_b = req_shape(m, p + "sa1.in.bias", {value}, true);
            L.sa1_out_w = req_shape(m, p + "sa1.out.weight", {value, dim}, false);
            L.sa1_out_b = req_shape(m, p + "sa1.out.bias", {dim}, true);
            L.sa2_in_w = req_shape(m, p + "sa2.in.weight", {dim, value}, false);
            L.sa2_in_b = req_shape(m, p + "sa2.in.bias", {value}, true);
            L.sa2_out_w = req_shape(m, p + "sa2.out.weight", {value, dim}, false);
            L.sa2_out_b = req_shape(m, p + "sa2.out.bias", {dim}, true);
            for (int f = 0; f < 3; f++) {
                const int ff = ffn_dims[s] * (3 + f) / 4;
                const std::string fp = p + "ff" + std::to_string(f + 1) + ".";
                L.ff_in_w[f] = req_shape(m, fp + "in.weight", {dim, ff}, false);
                L.ff_in_b[f] = req_shape(m, fp + "in.bias", {ff}, true);
                L.ff_out_w[f] = req_shape(m, fp + "out.weight", {ff, dim}, false);
                L.ff_out_b[f] = req_shape(m, fp + "out.bias", {dim}, true);
            }
            L.na_in_w = req_shape(m, p + "na.in.weight", {dim, 3 * hidden}, false);
            L.na_in_b = req_shape(m, p + "na.in.bias", {3 * hidden}, true);
            L.na_out_w = req_shape(m, p + "na.out.weight", {hidden, dim}, false);
            L.na_out_b = req_shape(m, p + "na.out.bias", {dim}, true);
            for (int c = 0; c < 2; c++) {
                const std::string cp = p + "cv" + std::to_string(c + 1) + ".";
                auto & C = L.cv[c];
                C.in_w = req_shape(m, cp + "in.weight", {dim, 2 * dim}, false);
                C.in_b = req_shape(m, cp + "in.bias", {2 * dim}, true);
                C.causal_w = req_shape(m, cp + "causal.weight", {kernel / 2 + 1, 1, dim}, true);
                C.causal_b = req_shape(m, cp + "causal.bias", {dim}, true);
                C.chunk_w = req_shape(m, cp + "chunk.weight", {kernel, 1, dim}, true);
                C.chunk_b = req_shape(m, cp + "chunk.bias", {dim}, true);
                C.chunk_scale = req_shape(m, cp + "chunk_scale", {kernel, dim, 2}, true);
                C.out_w = req_shape(m, cp + "out.weight", {dim, dim}, false);
                C.out_b = req_shape(m, cp + "out.bias", {dim}, true);
            }
            L.norm_bias = req_shape(m, p + "norm.bias", {dim}, true);
            L.norm_scale = std::exp(to_f32(req_shape(m, p + "norm.log_scale", {1}, true))[0]);
            L.bypass = req_shape(m, p + "bypass", {dim}, true);
            L.bypass_mid = req_shape(m, p + "bypass_mid", {dim}, true);
        }
    }
    m.out_ds_w = softmax_host(to_f32(req_shape(m, "z.out_ds_bias", {2}, true)));
    m.enc_proj_w = req_shape(m, "join.encoder_proj.weight", {768, 512}, false);
    m.enc_proj_b = req_shape(m, "join.encoder_proj.bias", {512}, true);
    m.dec_emb = to_f32(req_shape(m, "dec.emb.weight", {512, 5000}, false));
    m.dec_conv = to_f32(req_shape(m, "dec.conv.weight", {2, 4, 512}, true));
    m.dec_proj_w = to_f32(req_shape(m, "join.decoder_proj.weight", {512, 512}, false));
    m.dec_proj_b = to_f32(req_shape(m, "join.decoder_proj.bias", {512}, true));
    m.join_out_w = to_f32(req_shape(m, "join.output_linear.weight", {512, 5000}, false));
    m.join_out_b = to_f32(req_shape(m, "join.output_linear.bias", {5000}, true));
    return true;
}

// ---- graph helpers ---------------------------------------------------------

ggml_tensor * lin(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    const int64_t time = x->ne[1];
    const int64_t batch = x->ne[2];
    if (batch > 1) {
        x = ggml_reshape_2d(g, x, x->ne[0], time * batch);
    }
    ggml_tensor * y = ggml_mul_mat(g, w, x);
    if (b) {
        y = ggml_add(g, y, b);
    }
    return batch > 1 ? ggml_reshape_3d(g, y, y->ne[0], time, batch) : y;
}

// Swoosh-L(x) = log(1 + e^(x-4)) - 0.08x - 0.035; Swoosh-R: offset 1, constant 0.313261687.
ggml_tensor * swoosh(ggml_context * g, ggml_tensor * x, float off, float c) {
    ggml_tensor * sp = ggml_softplus(g, ggml_scale_bias(g, x, 1.0f, -off));
    return ggml_sub(g, sp, ggml_scale_bias(g, x, 0.08f, c));
}
ggml_tensor * swoosh_l(ggml_context * g, ggml_tensor * x) {
    return swoosh(g, x, 4.0f, 0.035f);
}
ggml_tensor * swoosh_r(ggml_context * g, ggml_tensor * x) {
    return swoosh(g, x, 1.0f, 0.313261687f);
}

// BiasNorm: x * mean((x - bias)^2)^-1/2 * exp(log_scale), over channels (ne0).
ggml_tensor * bias_norm(ggml_context * g, ggml_tensor * x, ggml_tensor * bias, float scale) {
    ggml_tensor * centered = ggml_sub(g, x, bias);
    ggml_tensor * sum = ggml_sum_rows(g, ggml_sqr(g, centered));
    // Silence drives the mean square to zero and the division to NaN, which
    // turns the logits into noise and makes the argmax depend on rounding. Floor
    // the sum; for speech the term is below float resolution of the sum.
    sum = ggml_clamp(g, sum, 1e-6f, INFINITY);
    ggml_tensor * denom = ggml_reshape_1d(g, sum, ggml_nelements(sum));
    denom = ggml_reshape_3d(g, ggml_sqrt(g, denom), 1, x->ne[1], x->ne[2]);
    return ggml_scale(g, ggml_div(g, x, denom), scale * std::sqrt(float(x->ne[0])));
}

// orig + (x - orig) * scale
ggml_tensor * bypass(ggml_context * g, ggml_tensor * orig, ggml_tensor * x, ggml_tensor * scale) {
    return ggml_add(g, orig, ggml_mul(g, ggml_sub(g, x, orig), scale));
}

ggml_tensor * ffn(ggml_context * g, ggml_tensor * x, ggml_tensor * w1, ggml_tensor * b1, ggml_tensor * w2, ggml_tensor * b2) {
    return lin(g, swoosh_l(g, lin(g, x, w1, b1)), w2, b2);
}

// Depthwise correlation over time of (C, Tin) with a (K, C) kernel: out (C, Tin-K+1).
ggml_tensor * dwconv_valid(ggml_context * g, ggml_tensor * x, ggml_tensor * w) {
    const int C = (int)x->ne[0], Tin = (int)x->ne[1];
    ggml_tensor * xt = ggml_cont(g, ggml_transpose(g, x)); // (Tin, C)
    ggml_tensor * y = ggml_ssm_conv(g, ggml_reshape_3d(g, xt, Tin, C, x->ne[2]), ggml_reshape_2d(g, w, w->ne[0], C));
    return y;
}

// Last n frames of each lane in a (channels, time, batch) tensor.
ggml_tensor * last_cols(ggml_context * g, ggml_tensor * x, int n) {
    return ggml_cont(g, ggml_view_3d(g, x, x->ne[0], n, x->ne[2], x->nb[1], x->nb[2], (size_t)(x->ne[1] - n) * x->nb[1]));
}

// Same window, left strided. Only for the cache write path: the destination is a
// ggml_cpy, whose kernels handle a strided source, so materialising the window
// first would copy the same bytes twice.
ggml_tensor * last_cols_view(ggml_context * g, ggml_tensor * x, int n) {
    return ggml_view_3d(g, x, x->ne[0], n, x->ne[2], x->nb[1], x->nb[2], (size_t)(x->ne[1] - n) * x->nb[1]);
}

// Streaming caches are written by copying the last rows of a wider buffer into a
// smaller one. Building that window with last_cols() materialises it (one CONT),
// then ggml_cpy moves it again; folding the two into one ggml_cpy whose source is
// the strided view removes one copy and one graph node per cache.
// XASR_CACHE_FOLD=0 restores the two-step path.
bool cache_fold() {
    static const bool on = [] {
        const char * s = std::getenv("XASR_CACHE_FOLD");
        return !s || std::atoi(s) != 0;
    }();
    return on;
}

// Record a cache write for the tail of build_chunk_graph. The name matches the
// input tensor the copy lands in, so add_cache() can pair them in build order.
void queue_cache(xasr_context * c, ggml_tensor * src, const std::string & name) {
    c->building->folded_caches.emplace_back(src, name);
}

// Copy the last n frames of src into streaming cache `name`.
void out_tensor(ggml_cgraph * gf, ggml_tensor * t, const std::string & name);

void cache_write(xasr_context * c, ggml_context * g, ggml_cgraph * gf, ggml_tensor * src, int n,
                 const std::string & name) {
    if (cache_fold()) {
        queue_cache(c, last_cols_view(g, src, n), name);
    } else {
        out_tensor(gf, last_cols(g, src, n), "o_" + name);
    }
}

ggml_tensor * rows(ggml_context * g, ggml_tensor * x, int r0, int n) {
    return ggml_cont(g, ggml_view_3d(g, x, n, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)r0 * ggml_element_size(x)));
}

// CompactRelPositionalEncoding rows for relative positions -(seq+left-1) .. seq-1.
std::vector<float> compact_pe(int pos_dim, int seq, int left) {
    const int L2 = left + 2 * seq - 1;
    std::vector<float> pe((size_t)L2 * pos_dim);
    const float c = std::sqrt((float)pos_dim);
    const float length_scale = (float)pos_dim / (2.0f * (float)3.14159265358979323846);
    for (int n = 0; n < L2; n++) {
        const float x = (float)(n - (seq + left - 1));
        const float sgn = x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f);
        const float xc = c * sgn * (std::log(std::fabs(x) + c) - std::log(c));
        const float xa = std::atan(xc / length_scale);
        float * r = pe.data() + (size_t)n * pos_dim;
        for (int f = 0; f < pos_dim / 2; f++) {
            r[2 * f] = std::cos(xa * (float)(f + 1));
            r[2 * f + 1] = std::sin(xa * (float)(f + 1));
        }
        r[pos_dim - 1] = 1.0f;
    }
    return pe;
}

std::string lname(const char * what, int li) {
    return std::string(what) + "_" + std::to_string(li);
}

void in_tensor(ggml_tensor * t, const std::string & name) {
    ggml_set_name(t, name.c_str());
    ggml_set_input(t);
}

void out_tensor(ggml_cgraph * gf, ggml_tensor * t, const std::string & name) {
    ggml_set_name(t, name.c_str());
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
}

ggml_tensor * fixed_input(xasr_context * c, ggml_tensor * tensor, std::vector<float> values) {
    ggml_set_input(tensor);
    ggml_set_output(tensor);
    c->building->constants.emplace_back(tensor, std::move(values));
    return tensor;
}

ggml_tensor * project_positions(xasr_context * c, ggml_context * g, const std::vector<float> & positions,
                               ggml_tensor * weight, int head_dim, int heads) {
    const int width = c->model.hp.pos_dim;
    const int length = positions.size() / width;
    const auto weights = to_f32(weight);
    std::vector<float> projected(size_t(head_dim) * length * heads);
    for (int h = 0; h < heads; ++h) {
        for (int t = 0; t < length; ++t) {
            for (int k = 0; k < head_dim; ++k) {
                const float * row = weights.data() + (h * head_dim + k) * width;
                float sum = 0.0f;
                for (int i = 0; i < width; ++i) {
                    sum += row[i] * positions[t * width + i];
                }
                projected[(h * length + t) * head_dim + k] = sum;
            }
        }
    }
    return fixed_input(c, ggml_new_tensor_3d(g, GGML_TYPE_F32, head_dim, length, heads), std::move(projected));
}

ggml_tensor * conv_edge_scale(xasr_context * c, ggml_context * g, ggml_tensor * weight, int seq, int dim, int kernel) {
    const auto edges = to_f32(weight);
    std::vector<float> scale(size_t(seq) * dim);
    for (int t = 0; t < seq; ++t) {
        for (int d = 0; d < dim; ++d) {
            const int right = t + kernel - seq;
            const float left_scale = t < kernel ? edges[d * kernel + t] : 0.0f;
            const float right_scale = right >= 0 ? edges[(dim + d) * kernel + right] : 0.0f;
            scale[t * dim + d] = 1.0f + (left_scale + right_scale);
        }
    }
    return fixed_input(c, ggml_new_tensor_2d(g, GGML_TYPE_F32, dim, seq), std::move(scale));
}

// One Zipformer2 layer, streaming_forward (streaming block).
ggml_tensor * zipformer_layer(xasr_context * c, ggml_context * g, ggml_cgraph * gf, int s, const zlayer & L, int li,
                             ggml_tensor * x, const std::vector<float> & positions, ggml_tensor * mask, int seq, int left) {
    const auto & hp = c->model.hp;
    const int d = hp.dims[s], H = hp.n_heads[s], qd = hp.qd[s], vd = hp.vd[s], pd = hp.pd[s], K = hp.kernel[s];
    const int hid = 3 * d / 4, pad = K / 2, Tk = left + seq;
    const auto batch = x->ne[2];
    ggml_tensor * src_orig = x;

    // 1. attention weights
    ggml_tensor * in = lin(g, x, L.aw_in_w, L.aw_in_b); // (H*(2qd+pd), seq)
    ggml_tensor * q = ggml_reshape_4d(g, rows(g, in, 0, H * qd), qd, H, seq, batch);
    ggml_tensor * k = rows(g, in, H * qd, H * qd);
    ggml_tensor * p = ggml_reshape_4d(g, rows(g, in, 2 * H * qd, H * pd), pd, H, seq, batch);
    ggml_tensor * ck = ggml_new_tensor_3d(g, GGML_TYPE_F32, H * qd, left, batch);
    in_tensor(ck, lname("ck", li));
    ggml_tensor * kfull = ggml_concat(g, ck, k, 1); // (H*qd, Tk)
    cache_write(c, g, gf, kfull, left, lname("ck", li));
    ggml_tensor * Q = ggml_cont(g, ggml_permute(g, q, 0, 2, 1, 3));                                     // (qd, seq, H)
    ggml_tensor * Kh = ggml_cont(g, ggml_permute(g, ggml_reshape_4d(g, kfull, qd, H, Tk, batch), 0, 2, 1, 3)); // (qd, Tk, H)
    ggml_tensor * sc = ggml_mul_mat(g, Kh, Q);                                                          // (Tk, seq, H)
    ggml_tensor * PE = project_positions(c, g, positions, L.aw_pos_w, pd, H);
    ggml_tensor * P = ggml_cont(g, ggml_permute(g, p, 0, 2, 1, 3)); // (pd, seq, H)
    ggml_tensor * ps = ggml_mul_mat(g, PE, P);                      // (L2, seq, H)
    // rel -> abs: out[j, t] = ps[j + seq-1-t, t] (icefall's as_strided)
    ggml_tensor * shifted =
        ggml_view_4d(g, ps, Tk, seq, H, batch, ps->nb[1] - ps->nb[0], ps->nb[2], ps->nb[3], (size_t)(seq - 1) * ps->nb[0]);
    sc = ggml_add(g, sc, ggml_cont(g, shifted));
    sc = ggml_add(g, sc, mask);            // (Tk) broadcast: -inf on not-yet-filled left context
    ggml_tensor * W = ggml_soft_max(g, sc); // (Tk, seq, H)

    // 2. FF1
    x = ggml_add(g, x, ffn(g, x, L.ff_in_w[0], L.ff_in_b[0], L.ff_out_w[0], L.ff_out_b[0]));

    // 3. NonlinAttention with head 0's weights
    {
        ggml_tensor * t = lin(g, x, L.na_in_w, L.na_in_b); // (3*hid, seq)
        ggml_tensor * sgate = ggml_tanh(g, rows(g, t, 0, hid));
        ggml_tensor * xx = ggml_mul(g, rows(g, t, hid, hid), sgate);
        ggml_tensor * y = rows(g, t, 2 * hid, hid);
        ggml_tensor * cn = ggml_new_tensor_3d(g, GGML_TYPE_F32, hid, left, batch);
        in_tensor(cn, lname("cn", li));
        ggml_tensor * xp = ggml_concat(g, cn, xx, 1); // (hid, Tk)
        cache_write(c, g, gf, xp, left, lname("cn", li));
        ggml_tensor * W0 = ggml_view_3d(g, W, Tk, seq, batch, W->nb[1], W->nb[3], 0);
        ggml_tensor * o = ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, xp)), W0); // (hid, seq)
        x = ggml_add(g, x, lin(g, ggml_mul(g, o, y), L.na_out_w, L.na_out_b));
    }

    auto self_attn = [&](ggml_tensor * xin, ggml_tensor * wi, ggml_tensor * bi, ggml_tensor * wo, ggml_tensor * bo,
                         const char * cname) {
        ggml_tensor * v = lin(g, xin, wi, bi); // (H*vd, seq)
        ggml_tensor * cv = ggml_new_tensor_3d(g, GGML_TYPE_F32, H * vd, left, batch);
        in_tensor(cv, lname(cname, li));
        ggml_tensor * vf = ggml_concat(g, cv, v, 1); // (H*vd, Tk)
        cache_write(c, g, gf, vf, left, lname(cname, li));
        ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, ggml_reshape_4d(g, vf, vd, H, Tk, batch), 1, 2, 0, 3)); // (Tk, vd, H)
        ggml_tensor * o = ggml_mul_mat(g, Vt, W);                                                        // (vd, seq, H)
        o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3));                                               // (vd, H, seq)
        return lin(g, ggml_reshape_3d(g, o, H * vd, seq, batch), wo, bo);
    };

    auto conv = [&](ggml_tensor * xin, const conv_mod& C, const char * cname) {
        ggml_tensor * t = lin(g, xin, C.in_w, C.in_b); // (2d, seq)
        ggml_tensor * xg = ggml_mul(g, rows(g, t, 0, d), ggml_sigmoid(g, rows(g, t, d, d)));
        ggml_tensor * cc = ggml_new_tensor_3d(g, GGML_TYPE_F32, d, pad, batch);
        in_tensor(cc, lname(cname, li));
        ggml_tensor * xc = ggml_concat(g, cc, xg, 1); // (d, pad+seq)
        cache_write(c, g, gf, xc, pad, lname(cname, li));
        ggml_tensor * causal = ggml_add(g, dwconv_valid(g, xc, C.causal_w), C.causal_b);
        ggml_tensor * chunk = dwconv_valid(g, ggml_pad_ext(g, xg, 0, 0, pad, pad, 0, 0, 0, 0), C.chunk_w);
        chunk = ggml_add(g, chunk, C.chunk_b);
        chunk = ggml_mul(g, chunk, conv_edge_scale(c, g, C.chunk_scale, seq, d, K));
        return lin(g, swoosh_r(g, ggml_add(g, chunk, causal)), C.out_w, C.out_b);
    };

    // 4.-10.
    x = ggml_add(g, x, self_attn(x, L.sa1_in_w, L.sa1_in_b, L.sa1_out_w, L.sa1_out_b, "cv1"));
    x = ggml_add(g, x, conv(x, L.cv[0], "cc1"));
    x = ggml_add(g, x, ffn(g, x, L.ff_in_w[1], L.ff_in_b[1], L.ff_out_w[1], L.ff_out_b[1]));
    x = bypass(g, src_orig, x, L.bypass_mid);
    x = ggml_add(g, x, self_attn(x, L.sa2_in_w, L.sa2_in_b, L.sa2_out_w, L.sa2_out_b, "cv2"));
    x = ggml_add(g, x, conv(x, L.cv[1], "cc2"));
    x = ggml_add(g, x, ffn(g, x, L.ff_in_w[2], L.ff_in_b[2], L.ff_out_w[2], L.ff_out_b[2]));
    // 11.-12.
    x = bias_norm(g, x, L.norm_bias, L.norm_scale);
    return bypass(g, src_orig, x, L.bypass);
}

// Weighted mean over each group of `ds` frames of (C, T): (C, T/ds).
ggml_tensor * simple_downsample(ggml_context * g, ggml_tensor * x, const std::vector<float> & w) {
    const int ds = (int)w.size(), C = (int)x->ne[0], Tout = (int)x->ne[1] / ds;
    ggml_tensor * acc = nullptr;
    for (int j = 0; j < ds; j++) {
        ggml_tensor * v = ggml_view_3d(g, x, C, Tout, x->ne[2], x->nb[1] * ds, x->nb[2], (size_t)j * x->nb[1]);
        ggml_tensor * t = ggml_scale(g, ggml_cont(g, v), w[(size_t)j]);
        acc = acc ? ggml_add(g, acc, t) : t;
    }
    return acc;
}

ggml_cgraph * build_chunk_graph(xasr_context * c, xasr_slot & slot) {
    auto & m = c->model;
    const auto & hp = m.hp;
    const geom & G = c->g;
    const int batch = slot.batch_size;
    ggml_init_params ip = {slot.meta.size(), slot.meta.data(), true};
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, graph_nodes, false);
    c->building = &slot;

    ggml_tensor * feats = ggml_new_tensor_4d(g, GGML_TYPE_F32, hp.feature_dim, G.T, 1, batch);
    in_tensor(feats, "feats");
    // Conv2dSubsampling: ggml (W=freq, H=time, C, N)
    auto cbias = [&](ggml_tensor * b) { return ggml_reshape_3d(g, b, 1, 1, b->ne[0]); };
    ggml_tensor * x = ggml_conv_2d(g, m.conv_w[0], feats, 1, 1, 1, 0, 1, 1); // pad (freq 1, time 0)
    x = swoosh_r(g, ggml_add(g, x, cbias(m.conv_b[0])));
    x = ggml_conv_2d(g, m.conv_w[1], x, 2, 2, 0, 0, 1, 1);
    x = swoosh_r(g, ggml_add(g, x, cbias(m.conv_b[1])));
    x = ggml_conv_2d(g, m.conv_w[2], x, 2, 1, 0, 0, 1, 1); // stride (freq 2, time 1)
    x = swoosh_r(g, ggml_add(g, x, cbias(m.conv_b[2])));   // (F', chunk+3, 128)
    const int Fp = (int)x->ne[0], Tc = (int)x->ne[1], Ch = (int)x->ne[2];
    GGML_ASSERT(Tc == G.chunk + 3);
    // ConvNeXt, streaming: 3 cached time frames on the left, no time padding
    ggml_tensor * ecache = ggml_new_tensor_4d(g, GGML_TYPE_F32, Fp, 3, Ch, batch);
    in_tensor(ecache, "emb_cache");
    ggml_tensor * xcat = ggml_concat(g, ecache, x, 1); // (F', chunk+6, 128)
    ggml_tensor * ewin = ggml_view_4d(g, xcat, Fp, 3, Ch, batch, xcat->nb[1], xcat->nb[2], xcat->nb[3], (size_t)G.chunk * xcat->nb[1]);
    if (cache_fold()) {
        queue_cache(c, ewin, "emb_cache");
    } else {
        out_tensor(gf, ggml_cont(g, ewin), "o_emb_cache");
    }
    ggml_tensor * y = ggml_conv_2d_dw_direct(g, m.cnx_dw_w, xcat, 1, 1, 3, 0, 1, 1); // (F', chunk, 128)
    y = ggml_add(g, y, cbias(m.cnx_dw_b));
    y = ggml_cont(g, ggml_permute(g, y, 1, 2, 0, 3)); // (128, F', chunk)
    y = ggml_reshape_3d(g, y, Ch, Fp * G.chunk, batch);
    y = lin(g, y, m.cnx_pw1_q8 ? m.cnx_pw1_q8 : ggml_reshape_2d(g, m.cnx_pw1_w, Ch, m.cnx_pw1_w->ne[3]), m.cnx_pw1_b);
    y = swoosh_l(g, y);
    y = lin(g, y, m.cnx_pw2_q8 ? m.cnx_pw2_q8 : ggml_reshape_2d(g, m.cnx_pw2_w, m.cnx_pw2_w->ne[2], Ch), m.cnx_pw2_b); // (128, F'*chunk)
    ggml_tensor * byp = ggml_view_4d(g, x, Fp, G.chunk, Ch, batch, x->nb[1], x->nb[2], x->nb[3], 0);
    byp = ggml_cont(g, ggml_permute(g, byp, 1, 2, 0, 3)); // (128, F', chunk)
    y = ggml_add(g, ggml_reshape_4d(g, y, Ch, Fp, G.chunk, batch), byp);
    // (b, c, t, f) -> (b, t, c*f): feature index c*F' + f
    y = ggml_cont(g, ggml_permute(g, y, 1, 0, 2, 3)); // (F', 128, chunk)
    y = ggml_reshape_3d(g, y, Fp * Ch, G.chunk, batch);
    x = lin(g, y, m.emb_out_w, m.emb_out_b);
    x = bias_norm(g, x, m.emb_norm_bias, m.emb_norm_scale); // (dims[0], chunk)

    const int S = (int)hp.dims.size();
    std::vector<ggml_tensor *> outs((size_t)S);
    int li = 0;
    for (int s = 0; s < S; s++) {
        const int dim = hp.dims[s], ds = hp.downsample[s];
        const int cur = (int)x->ne[0];
        if (dim <= cur)
            x = rows(g, x, 0, dim);
        else
            x = ggml_pad_ext(g, x, 0, dim - cur, 0, 0, 0, 0, 0, 0);
        ggml_tensor * src_orig = x;
        const int seq = G.chunk / ds, left = G.left / ds;
        if (ds > 1)
            x = simple_downsample(g, x, m.stacks[(size_t)s].ds_w);
        const auto pe = compact_pe(hp.pos_dim, seq, left);
        ggml_tensor * mask = ggml_new_tensor_4d(g, GGML_TYPE_F32, left + seq, 1, 1, batch);
        in_tensor(mask, "mask_" + std::to_string(s));
        for (auto & L : m.stacks[(size_t)s].layers)
            x = zipformer_layer(c, g, gf, s, L, li++, x, pe, mask, seq, left);
        if (ds > 1) {
            x = ggml_repeat(g, ggml_reshape_4d(g, x, dim, 1, seq, batch), ggml_new_tensor_4d(g, GGML_TYPE_F32, dim, ds, seq, batch));
            x = ggml_reshape_3d(g, x, dim, G.chunk, batch);
            x = bypass(g, src_orig, x, m.stacks[(size_t)s].combiner);
        }
        outs[(size_t)s] = x;
    }
    // _get_full_dim_output: the last stack, plus the higher channels of earlier, wider stacks
    ggml_tensor * full = outs[(size_t)S - 1];
    int cur = hp.dims[(size_t)S - 1];
    for (int s = S - 2; s >= 0; s--) {
        if (hp.dims[s] > cur) {
            full = ggml_concat(g, full, rows(g, outs[(size_t)s], cur, hp.dims[s] - cur), 0);
            cur = hp.dims[s];
        }
    }
    full = simple_downsample(g, full, m.out_ds_w); // (max dim, chunk/2)
    out_tensor(gf, lin(g, full, m.enc_proj_w, m.enc_proj_b), "enc_out");
    auto add_cache = [&](const std::string & name) {
        auto * input = ggml_graph_get_tensor(gf, name.c_str());
        ggml_set_output(input);
        c->building->caches.push_back(input);
        // Folded writes point straight at the strided window, so the tail copy is
        // the only one that moves these bytes.
        ggml_tensor * output = nullptr;
        if (cache_fold()) {
            for (auto & folded : c->building->folded_caches) {
                if (folded.second == name) {
                    output = folded.first;
                    break;
                }
            }
        }
        if (!output) {
            output = ggml_graph_get_tensor(gf, ("o_" + name).c_str());
        }
        // All reads of the old cache are in the graph before this write.
        ggml_build_forward_expand(gf, ggml_cpy(g, output, input));
    };
    add_cache("emb_cache");
    for (int l = 0; l < li; ++l) {
        for (const char * name : {"ck", "cn", "cv1", "cv2", "cc1", "cc2"}) {
            add_cache(lname(name, l));
        }
    }
    c->building = nullptr;
    ggml_free(g);
    return gf;
}

bool prepare_slot(xasr_context * c, xasr_slot & slot) {
    slot.graph = build_chunk_graph(c, slot);
    ggml_backend_t backends[] = {c->backend, c->backend_cpu};
    slot.sched = ggml_backend_sched_new(backends, nullptr, c->backend == c->backend_cpu ? 1 : 2, graph_nodes, false, false);
    if (!slot.sched) {
        return false;
    }
    for (auto * cache : slot.caches) {
        ggml_backend_sched_set_tensor_backend(slot.sched, cache, c->backend);
    }
    for (const auto & constant : slot.constants) {
        ggml_backend_sched_set_tensor_backend(slot.sched, constant.first, c->backend);
    }
    if (!ggml_backend_sched_alloc_graph(slot.sched, slot.graph)) {
        return false;
    }
    slot.feats = ggml_graph_get_tensor(slot.graph, "feats");
    slot.output = ggml_graph_get_tensor(slot.graph, "enc_out");
    size_t cache_bytes = 0;
    for (auto * tensor : slot.caches) {
        cache_bytes += ggml_nbytes(tensor);
    }
    for (const auto & [tensor, values] : slot.constants) {
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }
    slot.constants.clear();
    for (size_t s = 0; s < c->model.stacks.size(); ++s) {
        slot.masks.push_back(ggml_graph_get_tensor(slot.graph, ("mask_" + std::to_string(s)).c_str()));
    }
    int total = 0;
    int on_target = 0;
    for (int i = 0; i < ggml_graph_n_nodes(slot.graph); ++i) {
        auto * node = ggml_graph_node(slot.graph, i);
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
                continue;
            default:
                ++total;
        }
        if (ggml_backend_sched_get_tensor_backend(slot.sched, node) == c->backend) {
            ++on_target;
        } else {
            fprintf(stderr, "xasr: encoder fallback %s (%s)\n", ggml_op_name(node->op), node->name);
        }
    }
    fprintf(stderr, "xasr: encoder %s %d/%d ops, %d scheduler splits, %zu bytes of cache\n",
            ggml_backend_name(c->backend), on_target, total, ggml_backend_sched_get_n_splits(slot.sched), cache_bytes);
    return on_target == total;
}

bool load_hotwords(xasr_model & m, const char * path) {
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "xasr: cannot open hotword file %s\n", path);
        return false;
    }

    char line[512];
    size_t line_no = 0;
    while (file.getline(line, sizeof(line))) {
        ++line_no;
        const std::string text(line);
        const auto first = text.find_first_not_of(" \t\r");
        if (first == std::string::npos || text[first] == '#') {
            continue;
        }

        xasr_hotword word;
        std::istringstream record(text);
        if (!(record >> word.bonus) || !std::isfinite(word.bonus) ||
            word.bonus <= 0 || word.bonus > 3 || m.hotwords.size() == 128) {
            fprintf(stderr, "xasr: invalid hotword gain or count at line %zu\n", line_no);
            return false;
        }
        for (std::string piece; record >> piece;) {
            const auto match = std::find(m.vocab.begin(), m.vocab.end(), piece);
            const int id = static_cast<int>(match - m.vocab.begin());
            if (match == m.vocab.end() || id == m.hp.blank_id || id == m.hp.unk_id ||
                piece.front() == '<' || piece == "\xe2\x96\x81" || word.tokens.size() == 16) {
                fprintf(stderr, "xasr: invalid hotword token at line %zu\n", line_no);
                return false;
            }
            word.tokens.push_back(id);
        }
        if (word.tokens.size() < 2) {
            fprintf(stderr, "xasr: hotword at line %zu needs at least two tokens\n", line_no);
            return false;
        }
        m.hotwords.push_back(std::move(word));
    }
    if (!file.eof() || m.hotwords.empty()) {
        fprintf(stderr, "xasr: hotword file is empty, too long, or unreadable\n");
        return false;
    }
    fprintf(stderr, "xasr: loaded %zu hotwords\n", m.hotwords.size());
    return true;
}

bool is_speech_token(const xasr_model & m, int id) {
    return id >= 0 && id < m.hp.vocab && id != m.hp.blank_id && id != m.hp.unk_id &&
           m.vocab[id] != "<sos/eos>";
}

// Build the joiner graph once. The weight tensors stay in the model buffer on
// the same backend, so only the activation, the predictor state and the token
// ids cross the host boundary per call.
bool prepare_joiner(xasr_context * c) {
    if (c->joiner.ready) {
        return true;
    }
    if (!c->params.joiner_on_device) {
        return false;
    }
    ggml_tensor * weight = ggml_get_tensor(c->model.ctx, "join.output_linear.weight");
    ggml_tensor * bias   = ggml_get_tensor(c->model.ctx, "join.output_linear.bias");
    if (!weight || !bias) {
        return false;
    }
    const int J = c->model.hp.joiner_dim;
    const int T = 12; // encoder frames per chunk
    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false), nullptr, true };
    c->joiner.ctx = ggml_init(ip);
    if (!c->joiner.ctx) {
        return false;
    }
    ggml_context * g = c->joiner.ctx;
    ggml_tensor * act = ggml_new_tensor_2d(g, GGML_TYPE_F32, J, T);
    ggml_set_name(act, "j_act");
    ggml_set_input(act);
    ggml_tensor * pred = ggml_new_tensor_2d(g, GGML_TYPE_F32, J, 1);
    ggml_set_name(pred, "j_pred");
    ggml_set_input(pred);
    ggml_tensor * x = ggml_tanh(g, ggml_add(g, act, pred));
    x = ggml_add(g, ggml_mul_mat(g, weight, x), bias);
    ggml_tensor * raw_idx = ggml_argmax(g, x);
    ggml_tensor * idx = raw_idx;
    ggml_tensor * token_bonus = nullptr;
    if (!c->model.hotwords.empty()) {
        token_bonus = ggml_new_tensor_1d(g, GGML_TYPE_F32, c->model.hp.vocab);
        ggml_set_name(token_bonus, "j_token_bonus");
        ggml_set_input(token_bonus);
        ggml_set_name(raw_idx, "j_raw_idx");
        ggml_set_output(raw_idx);
        idx = ggml_argmax(g, ggml_add(g, x, token_bonus));
    }
    ggml_set_name(idx, "j_idx");
    ggml_set_output(idx);
    c->joiner.gf = ggml_new_graph_custom(g, 256, false);
    ggml_build_forward_expand(c->joiner.gf, idx);
    if (token_bonus) {
        ggml_build_forward_expand(c->joiner.gf, raw_idx);
    }
    ggml_backend_t backends[] = { c->backend, c->backend_cpu };
    c->joiner.sched = ggml_backend_sched_new(backends, nullptr, c->backend == c->backend_cpu ? 1 : 2, 256, false, false);
    if (!c->joiner.sched) {
        return false;
    }
    ggml_backend_sched_set_tensor_backend(c->joiner.sched, weight, c->backend);
    ggml_backend_sched_set_tensor_backend(c->joiner.sched, bias, c->backend);
    if (!ggml_backend_sched_alloc_graph(c->joiner.sched, c->joiner.gf)) {
        return false;
    }
    c->joiner.act = act;
    c->joiner.pred = pred;
    c->joiner.idx = idx;
    c->joiner.raw_idx = raw_idx;
    c->joiner.token_bonus = token_bonus;
    c->joiner.max_frames = T;
    c->joiner.ready = true;
    int nodes = ggml_graph_n_nodes(c->joiner.gf);
    int on_target = 0;
    for (int i = 0; i < nodes; ++i) {
        if (ggml_backend_sched_get_tensor_backend(c->joiner.sched, ggml_graph_node(c->joiner.gf, i)) == c->backend) {
            ++on_target;
        }
    }
    fprintf(stderr, "xasr: joiner %s %d/%d ops, weight %s %d x %d, %d frames per call\n",
            ggml_backend_name(c->backend), on_target, nodes, ggml_type_name(weight->type),
            (int) weight->ne[0], (int) weight->ne[1], T);
    return true;
}

bool joiner_compute(xasr_context * c, const float * act, const float * pred, const float * bonus, int32_t * idx) {
    xasr_joiner & j = c->joiner;
    ggml_backend_tensor_set(j.act, act, 0, ggml_nbytes(j.act));
    ggml_backend_tensor_set(j.pred, pred, 0, ggml_nbytes(j.pred));
    if (j.token_bonus) {
        ggml_backend_tensor_set(j.token_bonus, bonus, 0, ggml_nbytes(j.token_bonus));
    }
    if (ggml_backend_sched_graph_compute(j.sched, j.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    ggml_backend_tensor_get(j.idx, idx, 0, ggml_nbytes(j.idx));
    if (j.token_bonus) {
        int32_t raw[12];
        GGML_ASSERT(j.max_frames <= int(sizeof(raw) / sizeof(raw[0])));
        ggml_backend_tensor_get(j.raw_idx, raw, 0, ggml_nbytes(j.raw_idx));
        for (int f = 0; f < j.max_frames; ++f) {
            if (!is_speech_token(c->model, raw[f])) {
                idx[f] = raw[f];
            }
        }
    }
    return true;
}

void upload_encoder_lane(xasr_context * c, xasr_slot & slot, const float * feats, int pos, int lane, xasr_stats * st) {
    const auto & G = c->g;
    const size_t feature_bytes = ggml_nbytes(slot.feats) / slot.batch_size;
    auto up = xasr_clock::now();
    ggml_backend_tensor_set(slot.feats, feats, lane * feature_bytes, feature_bytes);
    st->feat_up_ms += span_ms(up);
    for (size_t s = 0; s < slot.masks.size(); ++s) {
        auto * tensor = slot.masks[s];
        const int ds = c->model.hp.downsample[s];
        const int unseen = G.left - pos / 2;
        up = xasr_clock::now();
        std::vector<float> mask(ggml_nelements(tensor) / slot.batch_size);
        for (int j = 0; j * ds < unseen; ++j) {
            mask[j] = -INFINITY;
        }
        st->mask_build_ms += span_ms(up);
        up = xasr_clock::now();
        const size_t mask_bytes = mask.size() * sizeof(float);
        ggml_backend_tensor_set(tensor, mask.data(), lane * mask_bytes, mask_bytes);
        st->mask_up_ms += span_ms(up);
    }
}

bool run_chunks(xasr_context * c, xasr_slot & slot, const float * feats, int n_rows, int & pos,
                std::vector<float> & enc_out, xasr_stats * st, bool commit_phase, bool single_chunk) {
    const auto & G = c->g;
    const int first = pos;
    for (; pos + G.T < n_rows; pos += G.decode_chunk_len) {
        if (single_chunk && pos != first) {
            break;
        }
        const auto chunk_begin = xasr_clock::now();
        upload_encoder_lane(c, slot, feats + size_t(pos) * c->model.hp.feature_dim, pos, 0, st);
        auto up = xasr_clock::now();
        double dsp_submit = 0, dsp_wait = 0;
        uint64_t dsp_batches = 0;
        if (c->dsp_stats) {
            c->dsp_stats(c->backend, &dsp_submit, &dsp_wait, &dsp_batches);
        }
        up = xasr_clock::now();
        const ggml_status status = ggml_backend_sched_graph_compute(slot.sched, slot.graph);
        st->compute_ms += span_ms(up);
        if (status != GGML_STATUS_SUCCESS) {
            return false;
        }
        const size_t start = enc_out.size();
        enc_out.resize(start + ggml_nelements(slot.output));
        up = xasr_clock::now();
        ggml_backend_tensor_get(slot.output, enc_out.data() + start, 0, ggml_nbytes(slot.output));
        st->readback_ms += span_ms(up);
        if (c->dsp_stats) {
            double submit_after = 0, wait_after = 0;
            uint64_t batches_after = 0;
            c->dsp_stats(c->backend, &submit_after, &wait_after, &batches_after);
            st->submit_ms += submit_after - dsp_submit;
            st->wait_ms += wait_after - dsp_wait;
            st->dsp_batches += (int) (batches_after - dsp_batches);
        }
        st->chunk_ms.push_back(span_ms(chunk_begin));
        if (commit_phase) {
            ++st->chunks_commit;
        } else {
            ++st->chunks_stream;
        }
    }
    if (c->params.verbosity > 1 && pos != first) {
        fprintf(stderr, "xasr: advanced %d feature frames\n", pos - first);
    }
    return true;
}

// ---- decoder / joiner (host) ------------------------------------------------

// decoder_proj(relu(grouped_conv(embedding(ctx tokens)))) for the last context_size tokens.
std::vector<float> decoder_out(const xasr_model & m, const std::vector<int> & hyp) {
    const auto & hp = m.hp;
    const int D = hp.decoder_dim, Kc = hp.context_size,
              gs = (int)(m.dec_conv.size() / ((size_t)D * Kc)); // in-ch per group
    std::vector<float> e((size_t)Kc * D, 0.0f);
    for (int k = 0; k < Kc; k++) {
        const int tok = hyp[hyp.size() - (size_t)Kc + (size_t)k];
        if (tok >= 0)
            std::memcpy(e.data() + (size_t)k * D, m.dec_emb.data() + (size_t)tok * D, (size_t)D * sizeof(float));
    }
    std::vector<float> h((size_t)D);
    for (int o = 0; o < D; o++) {
        const int g0 = (o / gs) * gs;
        float acc = 0.0f;
        for (int i = 0; i < gs; i++)
            for (int k = 0; k < Kc; k++)
                acc += m.dec_conv[((size_t)o * gs + i) * Kc + k] * e[(size_t)k * D + g0 + i];
        h[(size_t)o] = acc > 0.0f ? acc : 0.0f;
    }
    const int J = hp.joiner_dim;
    std::vector<float> out((size_t)J);
    for (int j = 0; j < J; j++) {
        float acc = m.dec_proj_b[(size_t)j];
        const float * w = m.dec_proj_w.data() + (size_t)j * D;
        for (int i = 0; i < D; i++)
            acc += w[i] * h[(size_t)i];
        out[(size_t)j] = acc;
    }
    return out;
}

// sherpa-onnx greedy search, resumable across chunks: tokens start as
// [-1]*(ctx-1) + [blank]; at most one symbol per encoder frame; blank and
// <unk> are never emitted and never re-run the decoder.
struct greedy_state {
    std::vector<int> hyp;
    std::vector<float> d; // decoder_proj output for the current context
    std::vector<int32_t> toks;
    std::vector<float> bonus;
};

void update_hotword_bonus(const xasr_model & m, greedy_state & gs) {
    if (gs.bonus.empty()) {
        return;
    }
    std::fill(gs.bonus.begin(), gs.bonus.end(), 0.0f);
    for (const auto & word : m.hotwords) {
        size_t prefix = std::min(gs.toks.size(), word.tokens.size() - 1);
        while (prefix && !std::equal(word.tokens.begin(), word.tokens.begin() + prefix, gs.toks.end() - prefix)) {
            --prefix;
        }
        const float gain = prefix ? word.bonus : 0.3f * word.bonus;
        auto & current = gs.bonus[word.tokens[prefix]];
        current = std::max(current, gain);
    }
}

void greedy_init(const xasr_model & m, greedy_state & gs) {
    gs.hyp.assign((size_t)m.hp.context_size, -1);
    gs.hyp.back() = m.hp.blank_id;
    gs.d = decoder_out(m, gs.hyp);
    gs.toks.clear();
    if (!m.hotwords.empty()) {
        gs.bonus.resize(m.hp.vocab);
        update_hotword_bonus(m, gs);
    }
}

void greedy_run(const xasr_model & m, greedy_state & gs, const float * enc, int n_enc) {
    const auto & hp = m.hp;
    const int J = hp.joiner_dim, V = hp.vocab;
    std::vector<float> t((size_t)J), logits((size_t)V);
    for (int f = 0; f < n_enc; f++) {
        const float * e = enc + (size_t)f * J;
        for (int j = 0; j < J; j++)
            t[(size_t)j] = std::tanh(e[j] + gs.d[(size_t)j]);
        int best = 0;
        for (int v = 0; v < V; v++) {
            float acc = m.join_out_b[(size_t)v];
            const float * w = m.join_out_w.data() + (size_t)v * J;
            for (int j = 0; j < J; j++)
                acc += w[j] * t[(size_t)j];
            logits[(size_t)v] = acc;
            if (acc > logits[(size_t)best])
                best = v;
        }
        if (!gs.bonus.empty() && is_speech_token(m, best)) {
            int biased = best;
            for (int v = 0; v < V; ++v) {
                if (logits[v] + gs.bonus[v] > logits[biased] + gs.bonus[biased]) {
                    biased = v;
                }
            }
            best = biased;
        }
        if (best < 0 || best >= V) {
            best = hp.blank_id;
        }
        if (best != hp.blank_id && best != hp.unk_id) {
            gs.toks.push_back(best);
            gs.hyp.push_back(best);
            gs.d = decoder_out(m, gs.hyp);
            update_hotword_bonus(m, gs);
        }
    }
}

// Same greedy rule as greedy_run, but the joiner projection and the argmax run
// on the backend in blocks of encoder frames. Tokens are taken in frame order;
// an emitted token updates the predictor and discards the remaining frames of
// the block, which are then recomputed with the new state.
bool greedy_run_dev(xasr_context * c, greedy_state & gs, const float * enc, int n_enc, xasr_stats * st) {
    const auto & hp = c->model.hp;
    const int J = hp.joiner_dim, T = c->joiner.max_frames;
    std::vector<float> act((size_t) T * J);
    std::vector<int32_t> idx((size_t) T);
    int f = 0;
    while (f < n_enc) {
        const int n = std::min(T, n_enc - f);
        for (int i = 0; i < T; ++i) {
            const int src = f + std::min(i, n - 1);
            std::memcpy(act.data() + (size_t) i * J, enc + (size_t) src * J, (size_t) J * sizeof(float));
        }
        const auto t = xasr_clock::now();
        if (!joiner_compute(c, act.data(), gs.d.data(), gs.bonus.data(), idx.data())) {
            return false;
        }
        st->joiner_dev_ms += span_ms(t);
        ++st->joiner_dev_calls;
        const int start = f;
        for (; f < start + n; ++f) {
            int best = idx[(size_t) (f - start)];
            // A degenerate input (digital silence) drives the normalisation to
            // NaN, and the argmax of NaN logits is not a valid token id.
            if (best < 0 || best >= hp.vocab) {
                best = hp.blank_id;
            }
            if (best != hp.blank_id && best != hp.unk_id) {
                gs.toks.push_back(best);
                gs.hyp.push_back(best);
                const auto p = xasr_clock::now();
                gs.d = decoder_out(c->model, gs.hyp);
                update_hotword_bonus(c->model, gs);
                st->joiner_pred_ms += span_ms(p);
                ++f;
                break;
            }
        }
    }
    return true;
}

// sherpa-onnx text-utils: IsCJK / IsPunct (code-point ranges) for RemoveSpaceBetweenCjk.
bool is_cjk(char32_t cp) {
    return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x2E80 && cp <= 0xA4CF) || (cp >= 0xA840 && cp <= 0xD7AF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF65 && cp <= 0xFFDC) ||
           (cp >= 0x20000 && cp <= 0x2FFFF);
}

bool is_punct(char32_t cp) {
    return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) || (cp >= 0x5B && cp <= 0x60) ||
           (cp >= 0x7B && cp <= 0x7E) || (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF01 && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65);
}

std::u32string utf8_to_u32(const std::string & s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = (uint8_t)s[i];
        const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
        for (int k = 1; k < n && i + (size_t)k < s.size(); k++)
            cp = (cp << 6) | ((uint8_t)s[i + (size_t)k] & 0x3F);
        out.push_back(cp);
        i += (size_t)n;
    }
    return out;
}

std::string u32_to_utf8(const std::u32string & u) {
    std::string out;
    for (char32_t cp : u) {
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// sherpa-onnx Convert(): join the symbols ('\u2581' prefix -> space, <unk> dropped),
// RemoveSpaceBetweenCjk (a space between two CJK characters or before
// punctuation goes), then drop leading spaces.
std::string detok(const xasr_model & m, const int32_t * toks, int n) {
    std::string s;
    for (int i = 0; i < n; i++) {
        std::string sym = m.vocab[(size_t)toks[i]];
        if (sym == "<unk>")
            continue;
        if (sym.size() >= 3 && (uint8_t)sym[0] == 0xe2 && (uint8_t)sym[1] == 0x96 && (uint8_t)sym[2] == 0x81)
            sym.replace(0, 3, " ");
        s += sym;
    }
    const std::u32string u = utf8_to_u32(s);
    std::u32string out;
    for (size_t i = 0; i < u.size(); i++) {
        if (i > 0 && u[i] == U' ' && i + 1 < u.size() && ((is_cjk(u[i - 1]) && is_cjk(u[i + 1])) || is_punct(u[i + 1])))
            continue;
        out.push_back(u[i]);
    }
    s = u32_to_utf8(out);
    const size_t b = s.find_first_not_of(' ');
    return b == std::string::npos ? std::string() : s.substr(b);
}

} // namespace

xasr_context * xasr_init_from_file(const char * path, xasr_context_params params) {
    std::unique_ptr<xasr_context, decltype(&xasr_free)> c(new xasr_context, xasr_free);
    if (!path || !params.device || params.n_threads < 1 || params.tail_pad_ms > 4000) {
        return nullptr;
    }
    c->params = params;
    c->backend = std::strcmp(params.device, "CPU") == 0 ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)
        : ggml_backend_init_by_name(params.device, nullptr);
    if (!c->backend) {
        fprintf(stderr, "xasr: cannot initialize device %s\n", params.device);
        return nullptr;
    }
    c->backend_cpu = ggml_backend_dev_type(ggml_backend_get_device(c->backend)) == GGML_BACKEND_DEVICE_TYPE_CPU
        ? c->backend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!c->backend_cpu) {
        return nullptr;
    }
    auto * dev_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(c->backend));
    c->dsp_stats = reinterpret_cast<void (*)(ggml_backend_t, double *, double *, uint64_t *)>(
        ggml_backend_reg_get_proc_address(dev_reg, "ggml_backend_hexagon_get_stats"));
    auto * cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(c->backend_cpu));
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads"));
    if (set_threads) {
        set_threads(c->backend_cpu, params.n_threads);
    }
    try {
        if (!load(c->model, path, c->backend)) {
            return nullptr;
        }
        if (params.hotwords && !load_hotwords(c->model, params.hotwords)) {
            return nullptr;
        }
        c->g.tail_pad_ms = params.tail_pad_ms;
        const int n_streams = params.n_streams > 0 ? params.n_streams : 1;
        for (int s = 0; s < n_streams; ++s) {
            auto slot = std::make_unique<xasr_slot>();
            slot->meta.resize(ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false));
            if (!prepare_slot(c.get(), *slot)) {
                fprintf(stderr, "xasr: selected backend must support the entire encoder graph\n");
                return nullptr;
            }
            c->slots.push_back(std::move(slot));
        }
        if (params.encoder_batch != 1 &&
            (!c->dsp_stats || (params.encoder_batch != 2 && params.encoder_batch != 4) || n_streams < params.encoder_batch)) {
            fprintf(stderr, "xasr: batch 2 or 4 requires HTP and at least that many streams\n");
            return nullptr;
        }
        for (int width = 2; width <= params.encoder_batch; width *= 2) {
            auto & slot = width == 2 ? c->batch_two : c->batch_four;
            slot = std::make_unique<xasr_slot>();
            slot->batch_size = width;
            slot->meta.resize(ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false));
            if (!prepare_slot(c.get(), *slot)) {
                fprintf(stderr, "xasr: encoder batch %d is unsupported on the selected device\n", width);
                return nullptr;
            }
        }
        if (params.encoder_batch > 1) {
            size_t cache_elements = 0;
            for (const auto * cache : c->slots.front()->caches) {
                cache_elements = std::max(cache_elements, size_t(ggml_nelements(cache)));
            }
            c->cache_copy.resize(cache_elements);
        }
        if (params.joiner_on_device && !prepare_joiner(c.get())) {
            fprintf(stderr, "xasr: joiner graph unavailable, using the host decoder\n");
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "xasr: %s\n", e.what());
        return nullptr;
    }
    return c.release();
}

void xasr_free(xasr_context * c) {
    if (!c) {
        return;
    }
    if (c->joiner.sched) {
        ggml_backend_sched_free(c->joiner.sched);
    }
    if (c->joiner.ctx) {
        ggml_free(c->joiner.ctx);
    }
    c->batch_four.reset();
    c->batch_two.reset();
    c->slots.clear();
    if (c->model.aux_buf) {
        ggml_backend_buffer_free(c->model.aux_buf);
    }
    if (c->model.aux_ctx) {
        ggml_free(c->model.aux_ctx);
    }
    if (c->model.buf) {
        ggml_backend_buffer_free(c->model.buf);
    }
    ggml_free(c->model.ctx);
    if (c->backend_cpu && c->backend_cpu != c->backend) {
        ggml_backend_free(c->backend_cpu);
    }
    if (c->backend) {
        ggml_backend_free(c->backend);
    }
    delete c;
}

// ---- streaming -------------------------------------------------------------
//
// Feature frames are computed as audio arrives, and are bit-identical to one
// fbank over the whole signal. With snip_edges=false frame i covers samples
// [160i - 120, 160i + 280), so it is final once 160i + 280 samples exist
// (frame 0 mirrors its left edge; later frames only read real samples). The
// tail, including the mirrored right edge, is computed on flush after the tail
// padding is appended.

struct xasr_stream {
    xasr_context * c = nullptr;
    xasr_slot * slot = nullptr;
    std::vector<float> pcm, feats;
    int pos = 0;
    greedy_state gs;
    bool finished = false;
    bool failed = false;
    bool stepping = false;
    bool step_flush = false;
    double step_ms = 0;
    xasr_stats stats;
};

namespace {

void stream_reset(xasr_stream * s) {
    s->pcm.clear();
    s->feats.clear();
    s->pos = 0;
    s->stats = xasr_stats{};
    for (auto * cache : s->slot->caches) {
        const std::vector<float> zero(ggml_nelements(cache));
        ggml_backend_tensor_set(cache, zero.data(), 0, ggml_nbytes(cache));
    }
    greedy_init(s->c->model, s->gs);
    s->finished = false;
}

void stream_feed(xasr_stream * s, const float * pcm, int n, bool flush) {
    if (n > 0) {
        s->pcm.insert(s->pcm.end(), pcm, pcm + n);
    }
    if (flush) {
        const size_t n_pcm = s->pcm.size();
        int n_rows = 0;
        if (s->c->g.tail_pad_ms >= 0) {
            s->pcm.resize(n_pcm + (size_t) s->c->g.tail_pad_ms * 16, 0.0f);
            n_rows = (int) ((s->pcm.size() + 80) / 160);
        } else {
            // A chunk at pos covers features pos..pos+60, so the chunk that
            // covers the last real frame needs 61 frames from its position plus
            // one more row to satisfy the readiness test.
            const int n_real = (int) ((n_pcm + 80) / 160);
            const int covering = n_real > 0 ? 48 * ((n_real - 1) / 48) : 0;
            const int target = covering + 62;
            n_rows = s->pos >= target ? n_real : std::max(n_real, target);
            const size_t want = n_rows > 0 ? (size_t) (160 * n_rows - 80) : 0;
            if (want > n_pcm) {
                s->pcm.resize(want, 0.0f);
            }
        }
        s->stats.pad_frames = n_rows - (int) ((n_pcm + 80) / 160);
    }
    auto t = xasr_clock::now();
    xasr_fbank_append(s->pcm, flush, s->feats);
    s->stats.fbank_ms += span_ms(t);
    const int n_frames = static_cast<int>(s->feats.size() / s->c->model.hp.feature_dim);
    s->stats.n_frames = n_frames;
}

bool stream_decode_projection(xasr_stream * s, const float * enc, int frames) {
    const auto t = xasr_clock::now();
    if (s->c->joiner.ready) {
        if (!greedy_run_dev(s->c, s->gs, enc, frames, &s->stats)) {
            s->failed = true;
            return false;
        }
    } else {
        greedy_run(s->c->model, s->gs, enc, frames);
    }
    s->stats.greedy_ms += span_ms(t);
    return true;
}

bool stream_decode(xasr_stream * s, bool flush, bool single_chunk) {
    const int n_frames = s->stats.n_frames;
    std::vector<float> enc;
    if (!run_chunks(s->c, *s->slot, s->feats.data(), n_frames, s->pos, enc, &s->stats, flush, single_chunk)) {
        s->failed = true;
        return false;
    }
    return stream_decode_projection(s, enc.data(), static_cast<int>(enc.size() / s->c->model.hp.joiner_dim));
}

bool stream_accept(xasr_stream * s, const float * pcm, int n, bool flush) {
    if (s->failed || s->stepping) {
        return false;
    }
    if (s->finished) {
        return true;
    }
    stream_feed(s, pcm, n, flush);
    if (!stream_decode(s, flush, false)) {
        return false;
    }
    s->finished = flush;
    return true;
}

std::string stream_text(const xasr_stream * s) {
    return detok(s->c->model, s->gs.toks.data(), (int)s->gs.toks.size());
}

} // namespace

xasr_stream * xasr_stream_init(xasr_context * c) {
    if (!c) {
        return nullptr;
    }
    xasr_slot * slot = nullptr;
    for (auto & candidate : c->slots) {
        if (!candidate->busy) {
            slot = candidate.get();
            break;
        }
    }
    if (!slot) {
        return nullptr;
    }
    auto s = std::make_unique<xasr_stream>();
    s->c = c;
    s->slot = slot;
    slot->busy = true;
    stream_reset(s.get());
    return s.release();
}

static bool valid_audio(const xasr_stream * s, const float * samples, int count) {
    constexpr size_t max_samples = 16000 * 600;
    return s && !s->finished && !s->failed && !s->stepping && count >= 0 &&
           (samples || count == 0) && s->pcm.size() <= max_samples && size_t(count) <= max_samples - s->pcm.size();
}

bool xasr_append(xasr_stream * s, const float * samples, int count) {
    if (!valid_audio(s, samples, count)) {
        return false;
    }
    const auto begin = xasr_clock::now();
    const bool ok = stream_accept(s, samples, count, false);
    s->stats.append_wall_ms += span_ms(begin);
    return ok;
}

static void stream_final(xasr_stream * s, std::string & text, double work_ms) {
    const auto t = xasr_clock::now();
    text = stream_text(s);
    s->stats.detok_ms = span_ms(t);
    s->stats.commit_wall_ms = work_ms + s->stats.detok_ms;
    if (timing_requested()) {
        const xasr_stats & st = s->stats;
        fprintf(stderr,
                R"(xasr-timing {"chunks":%d,"chunks_stream":%d,"chunks_commit":%d,"frames":%d,"pad_frames":%d,)"
                R"("fbank_ms":%.3f,"feat_up_ms":%.3f,"mask_build_ms":%.3f,"mask_up_ms":%.3f,)"
                R"("compute_ms":%.3f,"submit_ms":%.3f,"wait_ms":%.3f,"dsp_batches":%d,)"
                R"("readback_ms":%.3f,"greedy_ms":%.3f,"joiner_dev_ms":%.3f,"joiner_dev_calls":%d,)"
                R"("joiner_pred_ms":%.3f,"detok_ms":%.3f,)"
                R"("append_wall_ms":%.3f,"commit_wall_ms":%.3f,"chunk_ms":[)",
                st.chunks_stream + st.chunks_commit, st.chunks_stream, st.chunks_commit, st.n_frames, st.pad_frames,
                st.fbank_ms, st.feat_up_ms, st.mask_build_ms, st.mask_up_ms,
                st.compute_ms, st.submit_ms, st.wait_ms, st.dsp_batches,
                st.readback_ms, st.greedy_ms, st.joiner_dev_ms, st.joiner_dev_calls,
                st.joiner_pred_ms, st.detok_ms,
                st.append_wall_ms, st.commit_wall_ms);
        for (size_t i = 0; i < st.chunk_ms.size(); ++i) {
            fprintf(stderr, "%s%.3f", i ? "," : "", st.chunk_ms[i]);
        }
        fprintf(stderr, "]}\n");
    }
}

bool xasr_commit(xasr_stream * s, std::string & text) {
    const auto begin = xasr_clock::now();
    if (!s || !stream_accept(s, nullptr, 0, true)) {
        return false;
    }
    stream_final(s, text, span_ms(begin));
    return true;
}

bool xasr_begin_step(xasr_stream * s, const float * samples, int count, bool commit) {
    if (!s || s->failed || s->stepping || (!commit && !valid_audio(s, samples, count))) {
        return false;
    }
    if (commit && (samples || count != 0)) {
        return false;
    }
    const auto begin = xasr_clock::now();
    if (!s->finished) {
        stream_feed(s, samples, count, commit);
    }
    s->step_ms = span_ms(begin);
    s->step_flush = commit;
    s->stepping = true;
    return true;
}

bool xasr_step_ready(const xasr_stream * s) {
    return s && s->stepping && !s->finished && !s->failed && s->pos + s->c->g.T < s->stats.n_frames;
}

static xasr_step_result finish_step(xasr_stream * s, std::string & text) {
    if (s->failed) {
        s->stepping = false;
        return xasr_step_result::error;
    }
    if (xasr_step_ready(s)) {
        return xasr_step_result::pending;
    }
    s->stepping = false;
    if (s->step_flush) {
        s->finished = true;
        stream_final(s, text, s->step_ms);
    } else {
        s->stats.append_wall_ms += s->step_ms;
    }
    return xasr_step_result::done;
}

xasr_step_result xasr_step(xasr_stream * s, std::string & text) {
    if (!s || !s->stepping) {
        return xasr_step_result::error;
    }
    const auto begin = xasr_clock::now();
    if (xasr_step_ready(s)) {
        stream_decode(s, s->step_flush, true);
    }
    s->step_ms += span_ms(begin);
    return finish_step(s, text);
}

static void copy_batch_caches(xasr_context * c, xasr_slot & batch, xasr_stream * const * streams, bool gather) {
    for (int lane = 0; lane < batch.batch_size; ++lane) {
        const auto & caches = streams[lane]->slot->caches;
        for (size_t i = 0; i < caches.size(); ++i) {
            auto * single = caches[i];
            auto * combined = batch.caches[i];
            const size_t size = ggml_nbytes(single);
            const size_t offset = lane * size;
            GGML_ASSERT(size * batch.batch_size == ggml_nbytes(combined));
            if (gather) {
                ggml_backend_tensor_get(single, c->cache_copy.data(), 0, size);
                ggml_backend_tensor_set(combined, c->cache_copy.data(), offset, size);
            } else {
                ggml_backend_tensor_get(combined, c->cache_copy.data(), offset, size);
                ggml_backend_tensor_set(single, c->cache_copy.data(), 0, size);
            }
        }
    }
}

void xasr_step_batch(xasr_stream * const * streams, int count, xasr_step_result * results, std::string * texts) {
    if (count != 2 && count != 4) {
        return;
    }
    auto * c = streams[0] ? streams[0]->c : nullptr;
    auto * batch = c ? (count == 2 ? c->batch_two.get() : c->batch_four.get()) : nullptr;
    std::fill_n(results, count, xasr_step_result::error);
    if (!batch) {
        return;
    }
    for (int lane = 0; lane < count; ++lane) {
        if (!xasr_step_ready(streams[lane]) || streams[lane]->c != c ||
            std::find(streams, streams + lane, streams[lane]) != streams + lane) {
            return;
        }
    }

    const auto start = xasr_clock::now();
    copy_batch_caches(c, *batch, streams, true);
    const double gather_ms = span_ms(start);
    for (int lane = 0; lane < count; ++lane) {
        auto * s = streams[lane];
        upload_encoder_lane(c, *batch, s->feats.data() + size_t(s->pos) * c->model.hp.feature_dim,
                            s->pos, lane, &s->stats);
    }
    const auto compute = xasr_clock::now();
    const auto rc = ggml_backend_sched_graph_compute(batch->sched, batch->graph);
    const double compute_ms = span_ms(compute);
    if (rc != GGML_STATUS_SUCCESS) {
        for (int lane = 0; lane < count; ++lane) {
            streams[lane]->failed = true;
            streams[lane]->stepping = false;
        }
        return;
    }
    const auto scatter = xasr_clock::now();
    copy_batch_caches(c, *batch, streams, false);
    const double scatter_ms = span_ms(scatter);
    const double shared_ms = span_ms(start);
    if (timing_requested()) {
        fprintf(stderr, "xasr-batch size=%d gather_ms=%.3f compute_ms=%.3f scatter_ms=%.3f\n",
                count, gather_ms, compute_ms, scatter_ms);
    }

    const size_t out_elements = ggml_nelements(batch->output) / count;
    const size_t out_size = out_elements * sizeof(float);
    const int out_frames = static_cast<int>(out_elements / c->model.hp.joiner_dim);
    std::vector<float> projection(out_elements);
    for (int lane = 0; lane < count; ++lane) {
        const auto begin = xasr_clock::now();
        auto * s = streams[lane];
        ggml_backend_tensor_get(batch->output, projection.data(), lane * out_size, out_size);
        s->stats.readback_ms += span_ms(begin);
        s->stats.compute_ms += compute_ms;
        s->stats.chunk_ms.push_back(shared_ms + span_ms(begin));
        if (s->step_flush) {
            ++s->stats.chunks_commit;
        } else {
            ++s->stats.chunks_stream;
        }
        s->pos += c->g.decode_chunk_len;
        stream_decode_projection(s, projection.data(), out_frames);
        s->step_ms += shared_ms + span_ms(begin);
        results[lane] = finish_step(s, texts[lane]);
    }
}


const xasr_stats * xasr_stream_stats(const xasr_stream * s) {
    return s ? &s->stats : nullptr;
}

void xasr_stream_free(xasr_stream * s) {
    if (s) {
        if (s->slot) {
            s->slot->busy = false;
        }
        delete s;
    }
}


bool xasr_probe_encoder_batch(xasr_context * c, const std::vector<std::vector<float>> & audio) {
    const int batch = static_cast<int>(audio.size());
    if (!c || (batch != 2 && batch != 4) || c->slots.size() < audio.size()) {
        return false;
    }
    if (std::any_of(c->slots.begin(), c->slots.end(), [](const auto & slot) { return slot->busy; })) {
        return false;
    }
    std::vector<std::vector<float>> feats(batch);
    int chunks = 0;
    for (int lane = 0; lane < batch; ++lane) {
        if (audio[lane].empty() || audio[lane].size() > 16000 * 60) {
            return false;
        }
        xasr_fbank_append(audio[lane], true, feats[lane]);
        const int frames = static_cast<int>(feats[lane].size() / c->model.hp.feature_dim);
        const int ready = frames > c->g.T ? (frames - c->g.T - 1) / c->g.decode_chunk_len + 1 : 0;
        chunks = lane == 0 ? ready : std::min(chunks, ready);
    }
    if (chunks < 3) {
        fprintf(stderr, "xasr: batch probe needs at least three chunks per lane\n");
        return false;
    }

    xasr_slot combined;
    combined.batch_size = batch;
    combined.meta.resize(ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false));
    if (!prepare_slot(c, combined)) {
        return false;
    }
    auto reset = [](xasr_slot & slot) {
        for (auto * cache : slot.caches) {
            const std::vector<float> zero(ggml_nelements(cache));
            ggml_backend_tensor_set(cache, zero.data(), 0, ggml_nbytes(cache));
        }
    };

    const size_t lane_elements = ggml_nelements(combined.output) / batch;
    std::vector<float> reference(size_t(chunks) * batch * lane_elements);
    std::vector<float> actual(batch * lane_elements);
    std::vector<double> samples[2];
    double error = 0;
    double signal = 0;
    float max_diff = 0;
    xasr_stats stats;
    for (int arm = 0; arm < 2; ++arm) {
        reset(combined);
        for (int lane = 0; lane < batch; ++lane) {
            reset(*c->slots[lane]);
        }
        for (int index = 0; index < chunks; ++index) {
            const auto start = xasr_clock::now();
            const int pos = index * c->g.decode_chunk_len;
            const size_t offset = size_t(pos) * c->model.hp.feature_dim;
            float * expected = reference.data() + size_t(index) * actual.size();
            if (arm == 0) {
                for (int lane = 0; lane < batch; ++lane) {
                    auto & slot = *c->slots[lane];
                    upload_encoder_lane(c, slot, feats[lane].data() + offset, pos, 0, &stats);
                    if (ggml_backend_sched_graph_compute(slot.sched, slot.graph) != GGML_STATUS_SUCCESS) {
                        return false;
                    }
                    ggml_backend_tensor_get(slot.output, expected + lane * lane_elements, 0, ggml_nbytes(slot.output));
                }
            } else {
                for (int lane = 0; lane < batch; ++lane) {
                    upload_encoder_lane(c, combined, feats[lane].data() + offset, pos, lane, &stats);
                }
                if (ggml_backend_sched_graph_compute(combined.sched, combined.graph) != GGML_STATUS_SUCCESS) {
                    return false;
                }
                ggml_backend_tensor_get(combined.output, actual.data(), 0, ggml_nbytes(combined.output));
            }
            if (index >= 2) {
                samples[arm].push_back(span_ms(start));
            }
            if (arm == 1) {
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
                        return false;
                    }
                    const float diff = actual[i] - expected[i];
                    max_diff = std::max(max_diff, std::abs(diff));
                    error += double(diff) * diff;
                    signal += double(expected[i]) * expected[i];
                }
            }
        }
    }
    for (auto & values : samples) {
        std::sort(values.begin(), values.end());
    }
    const double relative_rms = std::sqrt(error / std::max(signal, 1e-30));
    printf("encoder_batch=%d chunks_per_lane=%d serial_median_ms=%.3f batch_median_ms=%.3f max_abs=%.6f relative_rms=%.6g\n",
           batch, chunks, samples[0][samples[0].size() / 2], samples[1][samples[1].size() / 2], max_diff, relative_rms);
    if (relative_rms >= 0.01) {
        fprintf(stderr, "xasr: batch projection relative RMS error exceeds 1%%\n");
        return false;
    }
    return true;
}
