// src/paraformer.cpp — FunASR Paraformer NAR-ASR runtime.
//
// Inference flow:
//   Kaldi fbank (80-mel) → LFR(7,6) → CMVN → 50 SANM blocks → CIF predictor
//   → 16 decoder blocks (FFN → FSMN → cross-attn) → argmax → characters
//
// The encoder reuses core_sanm::build_block(). See paraformer.h for overview.

#include "paraformer.h"

#include "core/kaldi_fbank.h"
#include "core/lfr.h"
#include "core/sanm.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (§232 paraformer GPU path)
#include "core/crispasr_env.h"
#include "core/parallel_for.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h" // core_cpu_backend::is_metal(§232 CUDA/Vulkan-default gate)
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <unordered_map>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"
#include "core/sched_prof.h"

// ===========================================================================
// Bench instrumentation — `PARAFORMER_BENCH=1` for per-stage timings.
// ===========================================================================

static bool paraformer_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_PARAFORMER_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct paraformer_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit paraformer_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~paraformer_bench_stage() {
        if (!paraformer_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  paraformer_bench: %-22s %.2f ms\n", name, ms);
    }
};

// Cumulative lap timer: prints time since origin, to expose graph-build and
// sched-alloc blind spots that the stage wrappers do not cover.
struct paraformer_bench_lap {
    std::chrono::steady_clock::time_point origin;
    paraformer_bench_lap() : origin(std::chrono::steady_clock::now()) {}
    void mark(const char* name) const {
        if (!paraformer_bench_enabled())
            return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - origin).count();
        std::fprintf(stderr, "  paraformer_lap:     %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Model definition
// ===========================================================================

struct paraformer_decoder_block {
    // Upstream order: norm1→FFN, norm2→FSMN, norm3→cross-attn
    // norm1 (before FFN)
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    // FFN: w_1 → relu → internal LN → w_2 (no w_2 bias)
    ggml_tensor *ffn_l1_w = nullptr, *ffn_l1_b = nullptr;
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor* ffn_l2_w = nullptr;
    // norm2 (before FSMN)
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    // FSMN depthwise conv (self-attn substitute)
    ggml_tensor* fsmn_w = nullptr; // (n_feat, K)
    // norm3 (before cross-attention)
    ggml_tensor *norm3_w = nullptr, *norm3_b = nullptr;
    // Cross-attention: Q from decoder, fused K+V from encoder
    ggml_tensor *cross_q_w = nullptr, *cross_q_b = nullptr;
    ggml_tensor *cross_kv_w = nullptr, *cross_kv_b = nullptr;
    ggml_tensor *cross_out_w = nullptr, *cross_out_b = nullptr;
};

struct paraformer_decoder_post {
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *ffn_l1_w = nullptr, *ffn_l1_b = nullptr;
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor* ffn_l2_w = nullptr;
};

struct paraformer_model {
    struct {
        uint32_t n_mels = 80, lfr_m = 7, lfr_n = 6;
        uint32_t sample_rate = 16000;
        uint32_t frame_length_ms = 25, frame_shift_ms = 10;
        uint32_t d_model = 512, n_heads = 4, ffn_dim = 2048;
        uint32_t n_enc_blocks0 = 1, n_enc_blocks = 49, sanm_kernel = 11;
        uint32_t n_dec_blocks = 16, dec_ffn_dim = 2048;
        uint32_t cif_conv_kernel = 3;
        uint32_t vocab_size = 8404;
        float ln_eps = 1e-5f;
    } hp;

    // CMVN
    ggml_tensor *cmvn_shift = nullptr, *cmvn_scale = nullptr;
    std::vector<float> cmvn_shift_v, cmvn_scale_v;

    // CIF cached weights
    std::vector<float> cif_conv_w_f32, cif_conv_b_f32, cif_out_w_f32;
    float cif_out_b_f32 = 0.0f;
    bool cif_weights_cached = false;

    // Encoder: entry block(s)
    std::vector<core_sanm::BlockWeights> enc0;
    // Encoder: main blocks
    std::vector<core_sanm::BlockWeights> enc;
    // Encoder: after_norm
    ggml_tensor *enc_after_norm_w = nullptr, *enc_after_norm_b = nullptr;

    // CIF predictor
    ggml_tensor *cif_conv_w = nullptr, *cif_conv_b = nullptr;
    ggml_tensor *cif_out_w = nullptr, *cif_out_b = nullptr;

    // Decoder blocks
    std::vector<paraformer_decoder_block> dec;
    paraformer_decoder_post dec_post;
    ggml_tensor *dec_after_norm_w = nullptr, *dec_after_norm_b = nullptr;
    ggml_tensor *dec_output_w = nullptr, *dec_output_b = nullptr;
    ggml_tensor* dec_embed_w = nullptr; // (vocab, d_model)
};

// One stage's cached graph and scheduler. A hit reuses the allocated graph, so
// the per-request cost falls to setting inputs and computing. Encoder and
// decoder keep separate schedulers: allocating a graph on a scheduler
// invalidates the other graph's buffers, which is what forced the old
// rebuild-every-call workaround.
// §21: span rewrite. A logit gain cannot change the number of tokens, so the
// BPE mismatches (footwall 3 tokens vs footwell 2) need a different tool: after
// argmax, an exact wrong-sequence match is replaced by the right one when the
// per-position logit average does not fall too far behind.
struct pf_repl {
    std::vector<int> wrong;
    std::vector<int> right;
    float margin = 0.0f;
};

struct pf_stage_cache {
    ggml_backend_sched_t sched = nullptr;
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    std::vector<char> meta;
    int k0 = -1, k1 = -1;
    uint64_t hits = 0, misses = 0;
};

struct paraformer_context {
    paraformer_model model;
    core_gguf::WeightLoad wl;
    std::string model_path;
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    pf_stage_cache enc_cache;
    pf_stage_cache dec_cache;
    paraformer_placement warmup_placement = {};
    int n_threads = 4;
    bool flash_attn = true;
    int verbosity = 0;
    std::vector<std::string> vocab; // token strings
    // §14.5: which sub-word convention this checkpoint's vocab uses. Detected
    // from the vocab itself so a new checkpoint needs no flag.
    int word_marker = 0; // 0 = trailing @@ (zh), 1 = leading U+2581 (en)
    std::vector<char> compute_meta;

    // §13: NAR logit bias for confusable whole-word tokens. Loaded once from
    // PF_HOTWORDS; empty by default, so the baseline is bit-identical. Each
    // entry adds a fixed gain to one token id at every CIF position (NAR has
    // no autoregressive prefix, so no head/prefix weighting is possible).
    std::vector<float> hotword_bonus; // size == vocab_size, 0 where unbiased
    // §13: prefix-conditional bias for BPE-fragment nouns. Rule applies gain to
    // `target` at position n only when the token emitted at n-1 was `prefix`.
    // e.g. se@@ -> at turns "se@@ ed" (seed) into "se@@ at" (seat), without
    // globally boosting `at` everywhere (which would corrupt other words).
    struct SeqRule { int prefix, target; float gain; };
    std::vector<SeqRule> hotword_seq;
    std::vector<pf_repl> hotword_repl;

    // Section 28: segment context. A trie of phrase token sequences (POI names,
    // contact names, cabin words). While decoding, the token that continues an
    // active phrase gets its logit boosted, so a phrase the model started is
    // much more likely to be finished as the domain spells it.
    struct CtxNode {
        std::vector<std::pair<int, int>> child; // token id -> node index
        float gain = 0.0f;                      // boost for entering this node
    };
    std::vector<CtxNode> ctx_trie;
    mutable std::unordered_map<std::string, int> vocab_index; // built lazily for phrase tokenisation
    int ctx_phrases = 0;
    bool ctx_loaded = false;
    int hotword_count = 0;
    bool hotword_loaded = false;

};

// Shared, read-only weight set loaded once from GGUF. A canonical copy lives
// in a CPU host buffer; each inference context clones it into its own backend
// buffer (host for CPU, tiled device buffer for HTP). This avoids re-reading
// the GGUF per context while keeping every context's tensors independent.
struct paraformer_weights {
    // One file read, reused by every context. Each context materializes its own
    // backend buffer from these bytes. A single GGUF-weight buffer cannot back
    // both HTP and CPU schedulers (ggml binds ops to the weight's physical
    // buffer; HTP also needs tiled repack), so per-context buffers are required
    // — what is shared is the load (disk I/O + GGUF parse), not the buffer.
    std::vector<char> gguf_bytes;
    std::string model_path;
    std::vector<std::string> vocab;
};

// ===========================================================================
// Helpers
// ===========================================================================

// mm_bias: matmul + optional bias add (shared helper).
static inline ggml_tensor* mm_bias(ggml_context* ctx0, ggml_tensor* W, ggml_tensor* x, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx0, W, x);
    return b ? ggml_add(ctx0, y, b) : y;
}

// §13: whole-word logit bias for confusable English function words.
// PF_HOTWORDS points to a file with one "<gain> <token>" per line; '#' comments
// and blank lines are ignored. Token strings must appear verbatim in the GGUF
// vocab (set/said/two/too/and/in are single tokens; seat/seed/footwell are BPE
// fragments and are intentionally not biassable this way). Gain is clamped to
// [0, 5]. Bias is applied to every CIF position; NAR decoding has no
// autoregressive prefix, so x-asr's head/prefix weighting does not apply.
// Empty/invalid config leaves hotword_bonus empty => bit-identical baseline.
struct pf_choice {
    int id;
    double start_row;
    double end_row;
    bool rewritten = false;

    size_t logits_row() const { return (size_t) std::lround(end_row); }
};

static void pf_apply_replacements(std::vector<pf_choice>& chosen, const float* logits, int V,
                                  const std::vector<pf_repl>& repls, const std::vector<std::string>& vocab, bool verbose) {
    if (verbose) {
        fprintf(stderr, "pf-repl: %zu rules, %zu chosen\n", repls.size(), chosen.size());
    }
    for (const auto& r : repls) {
        const size_t nw = r.wrong.size(), nr = r.right.size();
        if (verbose) {
            std::string w, c;
            for (int id : r.wrong) w += vocab[(size_t) id] + "|";
            for (size_t i = 0; i < chosen.size() && i < 8; ++i) {
                const int id = chosen[i].id;
                c += (id >= 0 && (size_t) id < vocab.size()) ? vocab[(size_t) id] + "|" : "?|";
            }
            fprintf(stderr, "pf-repl: rule [%s] chosen head [%s]\n", w.c_str(), c.c_str());
        }
        if (nw == 0 || nr == 0 || chosen.size() < nw) {
            continue;
        }
        size_t p = 0;
        while (p + nw <= chosen.size()) {
            // Compare token text, not ids: the vocab can hold duplicate strings.
            bool match = true;
            for (size_t i = 0; i < nw; ++i) {
                const int a = chosen[p + i].id, b = r.wrong[i];
                if (a < 0 || (size_t) a >= vocab.size() || (size_t) b >= vocab.size() ||
                    vocab[(size_t) a] != vocab[(size_t) b]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                ++p;
                continue;
            }
            double sw = 0.0, sr = 0.0;
            for (size_t i = 0; i < nw; ++i) {
                sw += logits[chosen[p + i].logits_row() * V + r.wrong[i]];
            }
            for (size_t i = 0; i < nr; ++i) {
                sr += logits[chosen[p + std::min(i, nw - 1)].logits_row() * V + r.right[i]];
            }
            const double avg_w = sw / (double) nw, avg_r = sr / (double) nr;
            if (verbose) {
                fprintf(stderr, "pf-repl: match at %zu, avg_wrong %.2f avg_right %.2f deficit %.2f\n", p, avg_w, avg_r,
                        avg_w - avg_r);
            }
            if (avg_r + r.margin < avg_w) {
                ++p;
                continue;
            }
            std::vector<pf_choice> replacement;
            replacement.reserve(nr);
            const double first = chosen[p].start_row, last = chosen[p + nw - 1].end_row;
            for (size_t i = 0; i < nr; ++i) {
                if (nw == nr) {
                    replacement.push_back({r.right[i], chosen[p + i].start_row, chosen[p + i].end_row, true});
                } else if (nr == 1) {
                    replacement.push_back({r.right[i], first, last, true});
                } else {
                    const double row = first + (last - first) * (double) i / (double) (nr - 1);
                    replacement.push_back({r.right[i], row, row, true});
                }
            }
            auto at = chosen.erase(chosen.begin() + p, chosen.begin() + p + nw);
            chosen.insert(at, replacement.begin(), replacement.end());
            if (verbose) {
                fprintf(stderr, "paraformer: span rewrite at %zu, avg %.2f -> %.2f\n", p, avg_w, avg_r);
            }
            p += nr;
        }
    }
}

// Greedy longest-match tokenisation of one phrase: Chinese characters and Latin
// BPE pieces both resolve against the model vocab. Explicit token sequences can
// be given as a+b+c when the greedy match is not what the model would emit.
static std::vector<int> pf_tokenize_phrase(const paraformer_context* ctx, const std::string& text) {
    std::vector<int> ids;
    if (text.find('+') != std::string::npos) {
        size_t i = 0;
        for (;;) {
            const size_t j = text.find('+', i);
            const std::string tok = text.substr(i, j == std::string::npos ? std::string::npos : j - i);
            if (!tok.empty()) {
                int id = -1;
                for (size_t k = 0; k < ctx->vocab.size(); ++k) {
                    if (ctx->vocab[k] == tok) {
                        id = (int) k;
                        break;
                    }
                }
                if (id < 0) {
                    fprintf(stderr, "paraformer: context token not in vocab: '%s'\n", tok.c_str());
                    return {};
                }
                ids.push_back(id);
            }
            if (j == std::string::npos) {
                break;
            }
            i = j + 1;
        }
        return ids;
    }
    // The GGUF carries no merge table (only tokenizer.tokens), so a true BPE
    // encode is impossible. Encode the way this vocab is built instead: cover
    // the text with vocab pieces where every non-final piece ends in "@@" and
    // the last one does not, preferring fewer pieces and then longer ones.
    if (ctx->vocab_index.empty()) {
        for (size_t k = 0; k < ctx->vocab.size(); ++k) {
            ctx->vocab_index.emplace(ctx->vocab[k], (int) k);
        }
    }
    // Objective: fewest pieces, then the longest piece. Exact for CJK (single
    // characters). For Latin words this cannot reproduce the model's merge
    // order (the GGUF has no merge table), so it is best-effort and the explicit
    // "a+b+c" form exists for cases that must match exactly. Measured: a
    // rank-minimising objective produced f@@|oo@@|tw@@|el|l for "footwell" and
    // corrupted the transcript to "footwwall".
    const size_t n = text.size();
    std::vector<int> cost(n + 1, INT32_MAX);
    std::vector<int> next(n + 1, -1);
    std::vector<int> tok_at(n + 1, -1);
    cost[n] = 0;
    for (size_t i = n; i-- > 0;) {
        const size_t max_len = std::min<size_t>(24, n - i);
        for (size_t len = 1; len <= max_len; ++len) {
            const size_t j = i + len;
            if (j < n && ((unsigned char) text[j] & 0xC0) == 0x80) {
                continue; // do not cut inside a UTF-8 sequence
            }
            if (cost[j] == INT32_MAX) {
                continue;
            }
            // Mid-word spans prefer the "@@" continuation form (that is how the
            // model emits English pieces); the final span and every CJK span use
            // the plain token, because Chinese characters carry no "@@".
            const char* forms[2] = { nullptr, nullptr };
            std::string piece_cont, piece_plain = text.substr(i, len);
            if (j < n) {
                piece_cont = piece_plain + "@@";
                forms[0] = piece_cont.c_str();
                forms[1] = piece_plain.c_str();
            } else {
                forms[0] = piece_plain.c_str();
            }
            for (int f = 0; f < 2 && forms[f]; ++f) {
                const auto it = ctx->vocab_index.find(forms[f]);
                if (it == ctx->vocab_index.end()) {
                    continue;
                }
                const int cand = cost[j] + 1;
                if (cand < cost[i] || (cand == cost[i] && len > (size_t) (next[i] - (int) i))) {
                    cost[i] = cand;
                    next[i] = (int) j;
                    tok_at[i] = it->second;
                }
            }
        }
    }
    if (cost[0] == INT32_MAX) {
        fprintf(stderr, "paraformer: cannot tokenise '%s'\n", text.c_str());
        return {};
    }
    for (size_t i = 0; i < n; i = (size_t) next[i]) {
        ids.push_back(tok_at[i]);
    }
    return ids;
}

// PF_CONTEXT: one "<gain> <phrase>" per line. Phrases share a prefix tree, so
// only the continuation of a started phrase is boosted.
static void paraformer_load_context(paraformer_context* ctx) {
    if (!ctx || ctx->ctx_loaded || ctx->vocab.empty()) {
        return;
    }
    ctx->ctx_loaded = true;
    const char* path = std::getenv("PF_CONTEXT");
    if (!path || !path[0]) {
        return;
    }
    std::FILE* f = std::fopen(path, "r");
    if (!f) {
        fprintf(stderr, "paraformer: cannot open PF_CONTEXT '%s'\n", path);
        return;
    }
    ctx->ctx_trie.clear();
    ctx->ctx_trie.emplace_back(); // root
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        std::string text(line);
        const size_t c = text.find('#');
        if (c != std::string::npos) {
            text.resize(c);
        }
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ' || text.back() == '\t')) {
            text.pop_back();
        }
        float gain = 0.0f;
        char body[400] = {0};
        if (std::sscanf(text.c_str(), "%f %399s", &gain, body) != 2 || gain <= 0.0f || gain > 8.0f) {
            continue;
        }
        const std::vector<int> ids = pf_tokenize_phrase(ctx, std::string(body));
        if (ids.empty()) {
            continue;
        }
        if (std::getenv("PF_DUMP_PHRASES")) {
            std::string seq;
            for (int id : ids) {
                seq += ctx->vocab[(size_t) id] + "|";
            }
            fprintf(stderr, "pf-phrase: %s -> %s\n", body, seq.c_str());
        }
        int node = 0;
        for (int id : ids) {
            int next = -1;
            for (const auto& kv : ctx->ctx_trie[(size_t) node].child) {
                if (kv.first == id) {
                    next = kv.second;
                    break;
                }
            }
            if (next < 0) {
                next = (int) ctx->ctx_trie.size();
                ctx->ctx_trie.emplace_back();
                ctx->ctx_trie[(size_t) next].gain = gain;
                ctx->ctx_trie[(size_t) node].child.push_back({id, next});
            }
            node = next;
        }
        ++ctx->ctx_phrases;
    }
    std::fclose(f);
    fprintf(stderr, "paraformer: loaded %d context phrases (%zu trie nodes)\n", ctx->ctx_phrases,
            ctx->ctx_trie.size());
}

static void paraformer_load_hotwords(paraformer_context* ctx) {
    if (!ctx || ctx->hotword_loaded || ctx->vocab.empty())
        return;
    ctx->hotword_loaded = true;
    const char* path = std::getenv("PF_HOTWORDS");
    if (!path || !path[0])
        return;

    std::FILE* f = std::fopen(path, "r");
    if (!f) {
        fprintf(stderr, "paraformer: cannot open PF_HOTWORDS '%s'\n", path);
        return;
    }
    ctx->hotword_bonus.assign(ctx->vocab.size(), 0.0f);

    auto find_token = [&](const std::string& s) -> int {
        for (size_t i = 0; i < ctx->vocab.size(); ++i)
            if (ctx->vocab[i] == s)
                return (int) i;
        return -1;
    };

    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        std::string text(line);
        const size_t c = text.find('#');
        if (c != std::string::npos)
            text.resize(c);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ' || text.back() == '\t'))
            text.pop_back();
        // "R <margin> <w1+w2+...> => <r1+r2+...>" is a span rewrite, not a gain.
        if (text.size() > 2 && (text[0] == 'R' || text[0] == 'r') && (text[1] == ' ' || text[1] == '\t')) {
            const size_t arrow = text.find("=>");
            if (arrow == std::string::npos)
                continue;
            float margin = 0.0f;
            char lhs[256] = { 0 };
            if (std::sscanf(text.c_str() + 2, "%f %255s", &margin, lhs) != 2)
                continue;
            std::string rhs = text.substr(arrow + 2);
            auto split = [&](std::string s) {
                std::vector<int> ids;
                for (size_t i = 0; i < s.size();) {
                    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
                        ++i;
                    size_t j = i;
                    while (j < s.size() && s[j] != '+')
                        ++j;
                    if (j > i) {
                        const int id = find_token(s.substr(i, j - i));
                        if (id < 0) {
                            fprintf(stderr, "paraformer: replacement token not in vocab: '%s'\n", s.substr(i, j - i).c_str());
                            return std::vector<int>();
                        }
                        ids.push_back(id);
                    }
                    i = j + (j < s.size() ? 1 : 0);
                }
                return ids;
            };
            pf_repl r;
            r.wrong = split(std::string(lhs));
            r.right = split(rhs);
            r.margin = margin;
            if (!r.wrong.empty() && !r.right.empty()) {
                ctx->hotword_repl.push_back(std::move(r));
                ++ctx->hotword_count;
            }
            continue;
        }
        float gain = 0.0f;
        char tokbuf[256] = { 0 };
        if (std::sscanf(text.c_str(), "%f %255s", &gain, tokbuf) != 2 || tokbuf[0] == '\0')
            continue;
        const std::string tok(tokbuf);
        if (gain <= 0.0f || gain > 5.0f || !std::isfinite(gain)) {
            fprintf(stderr, "paraformer: hotword gain out of range [0,5]: %s\n", text.c_str());
            continue;
        }
        // Two forms:
        //   "<gain> <token>"          global bias at every position (existing)
        //   "<gain> <prefix>><token>" conditional: only when previous token==prefix
        // The '>' form targets BPE word-tail fragments (se@@>at for "seat").
        const size_t arrow = tok.find('>');
        if (arrow != std::string::npos && arrow > 0 && arrow + 1 < tok.size()) {
            const std::string pfx = tok.substr(0, arrow);
            const std::string tgt = tok.substr(arrow + 1);
            const int pid = find_token(pfx), tid = find_token(tgt);
            if (pid < 0 || tid < 0) {
                fprintf(stderr, "paraformer: seq hotword token not in vocab: '%s'\n", tok.c_str());
                continue;
            }
            ctx->hotword_seq.push_back({ pid, tid, gain });
            ++ctx->hotword_count;
            continue;
        }
        const int id = find_token(tok);
        if (id < 0) {
            fprintf(stderr, "paraformer: hotword token not in vocab: '%s'\n", tok.c_str());
            continue;
        }
        ctx->hotword_bonus[(size_t) id] = std::max(ctx->hotword_bonus[(size_t) id], gain);
        ++ctx->hotword_count;
    }
    std::fclose(f);
    fprintf(stderr, "paraformer: loaded %d whole-word logit bias tokens from '%s'\n", ctx->hotword_count, path);
}

// Diagnostic: top-K logits (net of the applied bias) at one CIF position,
// printed as offsets from the winner. Enabled by PF_DUMP_LOGITS=<positions>;
// a value of 0 (default) prints nothing. This is how the real Set/said,
// two/too gaps are measured before choosing a gain — x-asr learned the hard
// way that a 0.9-logit cap cannot flip a 1.76–6.42 gap.
static void paraformer_dump_logits(const paraformer_context* ctx, const float* row, int V, int pos, int K) {
    int top[8] = { 0 };
    K = std::min(K, 8);
    for (int k = 0; k < K; ++k) {
        for (int s = k + 1; s < K; ++s)
            if (row[top[s]] > row[top[k]])
                std::swap(top[k], top[s]);
    }
    for (int v = K; v < V; ++v) {
        if (row[v] <= row[top[K - 1]])
            continue;
        int k = K - 1;
        while (k > 0 && row[v] > row[top[k - 1]]) {
            top[k] = top[k - 1];
            --k;
        }
        top[k] = v;
    }
    fprintf(stderr, "PF_LOGIT pos=%d", pos);
    for (int k = 0; k < K; ++k) {
        const int id = top[k];
        const char* tok = (id >= 0 && (size_t) id < ctx->vocab.size()) ? ctx->vocab[(size_t) id].c_str() : "?";
        fprintf(stderr, " | %d \"%s\" %.2f", id, tok, row[id] - row[top[0]]);
    }
    // Watch mode: PF_WATCH_TOKENS="set,two,and" appends the exact gap of named
    // tokens even when they are outside top-K. This measures the gap a bias
    // must overcome without inflating K or guessing from noisy top-5 output.
    const char* watch_env = std::getenv("PF_WATCH_TOKENS");
    if (watch_env && watch_env[0]) {
        std::string names(watch_env);
        size_t start = 0;
        while (start < names.size()) {
            const size_t comma = names.find(',', start);
            std::string name = names.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            start = (comma == std::string::npos) ? names.size() : comma + 1;
            for (size_t i = 0; i < ctx->vocab.size(); ++i) {
                if (ctx->vocab[i] == name) {
                    fprintf(stderr, " [watch %s %.2f]", name.c_str(), row[i] - row[top[0]]);
                    break;
                }
            }
        }
    }
    fprintf(stderr, "\n");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool paraformer_load_model(paraformer_context* ctx) {
    auto& m = ctx->model;
    auto& hp = m.hp;
    // Keep a reference to wl for require()
    struct {
        core_gguf::WeightLoad* tensors_ref;
    } wl_wrapper;

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = ctx->wl.tensors.find(name);
        if (it == ctx->wl.tensors.end()) {
            fprintf(stderr, "paraformer: missing tensor '%s'\n", name);
            return nullptr;
        }
        return it->second;
    };

    // Read hyperparameters from GGUF KV
    if (gguf_context* gctx = core_gguf::open_metadata(ctx->model_path.c_str())) {
        hp.d_model = core_gguf::kv_u32(gctx, "paraformer.d_model", hp.d_model);
        hp.n_heads = core_gguf::kv_u32(gctx, "paraformer.n_heads", hp.n_heads);
        hp.n_enc_blocks0 = core_gguf::kv_u32(gctx, "paraformer.n_enc_blocks0", hp.n_enc_blocks0);
        hp.n_enc_blocks = core_gguf::kv_u32(gctx, "paraformer.n_enc_blocks", hp.n_enc_blocks);
        hp.sanm_kernel = core_gguf::kv_u32(gctx, "paraformer.sanm_kernel", hp.sanm_kernel);
        hp.n_dec_blocks = core_gguf::kv_u32(gctx, "paraformer.n_dec_blocks", hp.n_dec_blocks);
        hp.vocab_size = core_gguf::kv_u32(gctx, "paraformer.vocab_size", hp.vocab_size);
        hp.lfr_m = core_gguf::kv_u32(gctx, "paraformer.lfr_m", hp.lfr_m);
        hp.lfr_n = core_gguf::kv_u32(gctx, "paraformer.lfr_n", hp.lfr_n);
        core_gguf::free_metadata(gctx);
    }

    // CMVN
    m.cmvn_shift = get("paraformer.cmvn_shift");
    m.cmvn_scale = get("paraformer.cmvn_scale");
    if (m.cmvn_shift && m.cmvn_scale) {
        int D_lfr = (int)m.hp.n_mels * (int)m.hp.lfr_m;
        m.cmvn_shift_v.resize((size_t)D_lfr);
        m.cmvn_scale_v.resize((size_t)D_lfr);
        ggml_backend_tensor_get(m.cmvn_shift, m.cmvn_shift_v.data(), 0, D_lfr * sizeof(float));
        ggml_backend_tensor_get(m.cmvn_scale, m.cmvn_scale_v.data(), 0, D_lfr * sizeof(float));
    }

    // Encoder: entry block(s)
    m.enc0.resize(hp.n_enc_blocks0);
    for (uint32_t i = 0; i < hp.n_enc_blocks0; i++) {
        char buf[128];
        auto& b = m.enc0[i];
        auto g = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "paraformer.enc0.blk.%u.%s", i, suf);
            return get(buf);
        };
        b.norm1_w = g("norm1.w");
        b.norm1_b = g("norm1.b");
        b.norm2_w = g("norm2.w");
        b.norm2_b = g("norm2.b");
        b.attn_qkv_w = g("attn.qkv.w");
        b.attn_qkv_b = g("attn.qkv.b");
        b.attn_out_w = g("attn.out.w");
        b.attn_out_b = g("attn.out.b");
        b.attn_fsmn_w = g("attn.fsmn.w");
        b.ffn_l1_w = g("ffn.l1.w");
        b.ffn_l1_b = g("ffn.l1.b");
        b.ffn_l2_w = g("ffn.l2.w");
        b.ffn_l2_b = g("ffn.l2.b");
    }

    // Encoder: main blocks
    m.enc.resize(hp.n_enc_blocks);
    for (uint32_t i = 0; i < hp.n_enc_blocks; i++) {
        char buf[128];
        auto& b = m.enc[i];
        auto g = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "paraformer.enc.blk.%u.%s", i, suf);
            return get(buf);
        };
        b.norm1_w = g("norm1.w");
        b.norm1_b = g("norm1.b");
        b.norm2_w = g("norm2.w");
        b.norm2_b = g("norm2.b");
        b.attn_qkv_w = g("attn.qkv.w");
        b.attn_qkv_b = g("attn.qkv.b");
        b.attn_out_w = g("attn.out.w");
        b.attn_out_b = g("attn.out.b");
        b.attn_fsmn_w = g("attn.fsmn.w");
        b.ffn_l1_w = g("ffn.l1.w");
        b.ffn_l1_b = g("ffn.l1.b");
        b.ffn_l2_w = g("ffn.l2.w");
        b.ffn_l2_b = g("ffn.l2.b");
    }

    m.enc_after_norm_w = get("paraformer.enc.after_norm.w");
    m.enc_after_norm_b = get("paraformer.enc.after_norm.b");

    // CIF predictor
    m.cif_conv_w = get("paraformer.cif.conv.w");
    m.cif_conv_b = get("paraformer.cif.conv.b");
    m.cif_out_w = get("paraformer.cif.out.w");
    m.cif_out_b = get("paraformer.cif.out.b");

    // Decoder blocks
    m.dec.resize(hp.n_dec_blocks);
    for (uint32_t i = 0; i < hp.n_dec_blocks; i++) {
        char buf[128];
        auto& b = m.dec[i];
        auto g = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "paraformer.dec.blk.%u.%s", i, suf);
            return get(buf);
        };
        b.norm1_w = g("norm1.w");
        b.norm1_b = g("norm1.b");
        b.fsmn_w = g("fsmn.w");
        b.norm2_w = g("norm2.w");
        b.norm2_b = g("norm2.b");
        b.cross_q_w = g("cross.q.w");
        b.cross_q_b = g("cross.q.b");
        b.cross_kv_w = g("cross.kv.w");
        b.cross_kv_b = g("cross.kv.b");
        b.cross_out_w = g("cross.out.w");
        b.cross_out_b = g("cross.out.b");
        b.norm3_w = g("norm3.w");
        b.norm3_b = g("norm3.b");
        b.ffn_l1_w = g("ffn.l1.w");
        b.ffn_l1_b = g("ffn.l1.b");
        b.ffn_norm_w = g("ffn.norm.w");
        b.ffn_norm_b = g("ffn.norm.b");
        b.ffn_l2_w = g("ffn.l2.w");
    }

    // Decoder: post block
    {
        auto g = [&](const char* suf) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "paraformer.dec.post.%s", suf);
            return get(buf);
        };
        m.dec_post.norm1_w = g("norm1.w");
        m.dec_post.norm1_b = g("norm1.b");
        m.dec_post.ffn_l1_w = g("ffn.l1.w");
        m.dec_post.ffn_l1_b = g("ffn.l1.b");
        m.dec_post.ffn_norm_w = g("ffn.norm.w");
        m.dec_post.ffn_norm_b = g("ffn.norm.b");
        m.dec_post.ffn_l2_w = g("ffn.l2.w");
    }

    m.dec_after_norm_w = get("paraformer.dec.after_norm.w");
    m.dec_after_norm_b = get("paraformer.dec.after_norm.b");
    m.dec_output_w = get("paraformer.dec.output.w");
    m.dec_output_b = get("paraformer.dec.output.b");
    m.dec_embed_w = get("paraformer.dec.embed.w");

    return true;
}

// ===========================================================================
// Feature extraction (fbank + LFR + CMVN)
// ===========================================================================

static std::vector<float> paraformer_compute_features(paraformer_context* ctx, const float* pcm, int n_samples,
                                                      int& T_lfr, int& D_lfr) {
    const auto& hp = ctx->model.hp;

    // Kaldi fbank
    core_kaldi::FbankParams fp;
    fp.sample_rate = (int)hp.sample_rate;
    fp.n_mels = (int)hp.n_mels;
    fp.frame_length_ms = (int)hp.frame_length_ms;
    fp.frame_shift_ms = (int)hp.frame_shift_ms;
    fp.int16_scale = true;
    fp.window_type = core_kaldi::WindowType::Hamming;

    int T_raw = 0;
    auto fbank = core_kaldi::compute_fbank(pcm, n_samples, fp, T_raw);
    if (T_raw == 0)
        return {};

    // LFR (stack lfr_m frames with stride lfr_n)
    auto lfr = core_lfr::stack(fbank.data(), T_raw, (int)hp.n_mels, (int)hp.lfr_m, (int)hp.lfr_n, T_lfr);
    D_lfr = (int)hp.n_mels * (int)hp.lfr_m; // 80*7 = 560

    // CMVN: x = (x + shift) * scale (AddShift + Rescale)
    if (!ctx->model.cmvn_shift_v.empty() && !ctx->model.cmvn_scale_v.empty()) {
        const float* shift_ptr = ctx->model.cmvn_shift_v.data();
        const float* scale_ptr = ctx->model.cmvn_scale_v.data();
        float* lfr_ptr = lfr.data();
        auto apply_cmvn = [&](int t0, int t1) {
            for (int t = t0; t < t1; t++) {
                float* row = lfr_ptr + (size_t)t * D_lfr;
                for (int d = 0; d < D_lfr; d++) {
                    row[d] = (row[d] + shift_ptr[d]) * scale_ptr[d];
                }
            }
        };
        if (T_lfr >= 16) {
            core_parallel::for_each_chunk_work(T_lfr, ctx->n_threads, (uint64_t)D_lfr * 2, 65536, apply_cmvn);
        } else {
            apply_cmvn(0, T_lfr);
        }
    } else if (ctx->model.cmvn_shift && ctx->model.cmvn_scale) {
        std::vector<float> shift_v((size_t)D_lfr);
        std::vector<float> scale_v((size_t)D_lfr);
        ggml_backend_tensor_get(ctx->model.cmvn_shift, shift_v.data(), 0, D_lfr * sizeof(float));
        ggml_backend_tensor_get(ctx->model.cmvn_scale, scale_v.data(), 0, D_lfr * sizeof(float));
        for (int t = 0; t < T_lfr; t++) {
            for (int d = 0; d < D_lfr; d++) {
                lfr[(size_t)t * D_lfr + d] = (lfr[(size_t)t * D_lfr + d] + shift_v[d]) * scale_v[d];
            }
        }
    }
    return lfr;
}

// ===========================================================================
// CIF predictor (CPU, not a ggml graph — simple sequential loop)
// ===========================================================================

static void cif_predict(paraformer_context* ctx, const float* enc_out, int T, int D,
                        std::vector<float>& acoustic_embeds, int& N_tokens, std::vector<int>* fire_frames = nullptr) {
    // Conv1d(D, D, 3, padding=1) → ReLU → Linear(D, 1) → sigmoid
    // Then CIF accumulation loop.

    // Read weights as F32. F32 is a direct copy; F16 and any block-quantized
    // type are dequantized per row via the type's to_float trait (native bytes
    // sized by ggml_nbytes, row stride ggml_row_size — never n_elem*4, which
    // reads garbage/asserts on quantized tensors). Shipped paraformer GGUFs keep
    // the CIF predictor weights at F16/F32 (the quantizer skips them: the '.w'
    // names miss the is_weight test and the 3-D/vector dims fail ok_dims), so the
    // quantized path is defensive — but without it a quantized tensor would leave
    // dst as uninitialized garbage, silently. Mirrors pcs.cpp's read helper.
    auto read_f32 = [](ggml_tensor* t, float* dst, size_t n_elem) {
        if (!t)
            return;
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, dst, 0, n_elem * sizeof(float));
            return;
        }
        const int64_t ne0 = t->ne[0];
        const int64_t n_rows = ne0 > 0 ? (int64_t)n_elem / ne0 : 0;
        const size_t row_size = ggml_row_size(t->type, ne0);
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        const ggml_type_traits* tr = ggml_get_type_traits(t->type);
        for (int64_t r = 0; r < n_rows; r++)
            tr->to_float(raw.data() + (size_t)r * row_size, dst + (size_t)r * ne0, ne0);
    };

    if (!ctx->model.cif_weights_cached) {
        ctx->model.cif_conv_w_f32.resize((size_t)D * D * 3);
        ctx->model.cif_conv_b_f32.resize((size_t)D);
        ctx->model.cif_out_w_f32.resize((size_t)D);
        read_f32(ctx->model.cif_conv_w, ctx->model.cif_conv_w_f32.data(), ctx->model.cif_conv_w_f32.size());
        read_f32(ctx->model.cif_conv_b, ctx->model.cif_conv_b_f32.data(), ctx->model.cif_conv_b_f32.size());
        read_f32(ctx->model.cif_out_w, ctx->model.cif_out_w_f32.data(), ctx->model.cif_out_w_f32.size());
        read_f32(ctx->model.cif_out_b, &ctx->model.cif_out_b_f32, 1);
        ctx->model.cif_weights_cached = true;
    }

    const float* conv_w = ctx->model.cif_conv_w_f32.data();
    const float* conv_b = ctx->model.cif_conv_b_f32.data();
    const float* out_w = ctx->model.cif_out_w_f32.data();
    const float out_b = ctx->model.cif_out_b_f32;

    // Apply conv1d with padding=1
    std::vector<float> conv_out((size_t)T * D, 0.0f);
    float* conv_out_ptr = conv_out.data();
    auto run_conv_chunk = [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            float* out_row = conv_out_ptr + (size_t)t * D;
            for (int o = 0; o < D; o++) {
                float val = conv_b[o];
                const float* w_o = conv_w + (size_t)o * D * 3;
                for (int k = 0; k < 3; k++) {
                    int ti = t - 1 + k; // padding=1: shift by 1
                    if (ti < 0 || ti >= T)
                        continue;
                    const float* x_ti = enc_out + (size_t)ti * D;
                    const float* w_k = w_o + k;
                    for (int d = 0; d < D; d++) {
                        // conv_w layout: (out_ch, in_ch, K) → w[o * D * 3 + d * 3 + k]
                        val += w_k[(size_t)d * 3] * x_ti[d];
                    }
                }
                out_row[o] = val > 0.0f ? val : 0.0f; // ReLU
            }
        }
    };
    if (T >= 8) {
        core_parallel::for_each_chunk_work(T, ctx->n_threads, (uint64_t)D * D * 3, 65536, run_conv_chunk);
    } else {
        run_conv_chunk(0, T);
    }

    // Linear(D, 1) → sigmoid → alphas
    std::vector<float> alphas((size_t)T);
    float* alphas_ptr = alphas.data();
    auto run_linear_chunk = [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            float val = out_b;
            const float* in_row = conv_out_ptr + (size_t)t * D;
            for (int d = 0; d < D; d++) {
                val += out_w[d] * in_row[d];
            }
            alphas_ptr[t] = 1.0f / (1.0f + std::exp(-val)); // sigmoid
        }
    };
    if (T >= 16) {
        core_parallel::for_each_chunk_work(T, ctx->n_threads, D, 65536, run_linear_chunk);
    } else {
        run_linear_chunk(0, T);
    }

    // CIF accumulation: integrate alphas, fire when ≥ 1.0
    const float tail_threshold = 0.45f;
    float accum = 0.0f;
    std::vector<float> fired; // (D,) vectors, one per fired token
    std::vector<float> partial((size_t)D, 0.0f);

    for (int t = 0; t < T; t++) {
        float alpha = alphas[t];
        accum += alpha;
        if (accum >= 1.0f) {
            // Fire: blend current frame
            float remainder = 1.0f - (accum - alpha);
            for (int d = 0; d < D; d++) {
                partial[d] += remainder * enc_out[(size_t)t * D + d];
            }
            fired.insert(fired.end(), partial.begin(), partial.end());
            if (fire_frames)
                fire_frames->push_back(t);
            // Start new token with leftover
            float leftover = accum - 1.0f;
            for (int d = 0; d < D; d++) {
                partial[d] = leftover * enc_out[(size_t)t * D + d];
            }
            accum = leftover;
        } else {
            for (int d = 0; d < D; d++) {
                partial[d] += alpha * enc_out[(size_t)t * D + d];
            }
        }
    }
    // Tail handling: if remaining accum > threshold, fire one more
    if (accum > tail_threshold) {
        fired.insert(fired.end(), partial.begin(), partial.end());
        if (fire_frames)
            fire_frames->push_back(T - 1);
    }

    N_tokens = (int)(fired.size() / (size_t)D);
    acoustic_embeds = std::move(fired);
}

// ===========================================================================
// Decoder graph build (FSMN + cross-attention + FFN-with-LN)
// ===========================================================================

static ggml_tensor* build_decoder_block(ggml_context* ctx0, ggml_tensor* cur, ggml_tensor* enc_out, int N, int T_enc,
                                        const paraformer_decoder_block& b, int D, int n_heads, int head_dim, float eps,
                                        bool flash_attn) {
    // Upstream order (DecoderLayerSANM.forward):
    //   1. norm1 → FFN (no residual yet)
    //   2. norm2 → FSMN (with internal conv+identity residual) → add original input
    //   3. norm3 → cross-attn → add residual

    // 1. FFN: norm1 → w_1 → relu → internal LN → w_2
    ggml_tensor* residual = cur;
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, b.norm1_w, b.norm1_b, eps);
    x = mm_bias(ctx0, b.ffn_l1_w, x, b.ffn_l1_b); // (ffn_dim, N)
    x = ggml_relu(ctx0, x);
    x = ggml_norm_affine(ctx0, x, b.ffn_norm_w, b.ffn_norm_b, eps); // internal LN after activation
    x = ggml_mul_mat(ctx0, b.ffn_l2_w, x);                          // no bias on w_2
    // No residual addition here — residual spans FFN + FSMN together.

    // 2. FSMN self-attention: norm2 → depthwise conv + internal residual → add original input
    ggml_tensor* ffn_out = x;
    {
        ggml_tensor* normed = ggml_norm_affine(ctx0, ffn_out, b.norm2_w, b.norm2_b, eps);

        const int K = (int)b.fsmn_w->ne[0]; // kernel width; ne = (K, D)
        ggml_tensor* w4 = ggml_cast(ctx0, b.fsmn_w, GGML_TYPE_F32);
        w4 = ggml_reshape_4d(ctx0, w4, K, 1, 1, D);

        ggml_tensor* v4 = ggml_cont(ctx0, ggml_transpose(ctx0, normed)); // (N, D)
        v4 = ggml_reshape_4d(ctx0, v4, N, 1, D, 1);
        ggml_tensor* y = ggml_conv_2d_dw_direct(ctx0, w4, v4, 1, 1, (K - 1) / 2, 0, 1, 1);
        y = ggml_cont(ctx0, ggml_permute(ctx0, y, 1, 2, 0, 3));
        y = ggml_reshape_2d(ctx0, y, D, N);
        x = ggml_add(ctx0, y, normed); // FSMN internal residual: conv(normed) + normed
    }
    cur = ggml_add(ctx0, residual, x); // add original input residual

    // 3. Cross-attention: norm3 → Q from decoder, K+V from encoder
    residual = cur;
    x = ggml_norm_affine(ctx0, cur, b.norm3_w, b.norm3_b, eps);

    {
        // Q = linear_q(decoder_hidden): (D, N) → (D, N)
        ggml_tensor* Q = mm_bias(ctx0, b.cross_q_w, x, b.cross_q_b);

        // K, V = split(linear_k_v(encoder_out), D): (2D, T_enc) → K (D, T), V (D, T)
        ggml_tensor* kv = mm_bias(ctx0, b.cross_kv_w, enc_out, b.cross_kv_b); // (2D, T_enc)
        const size_t row_bytes = kv->nb[1];
        ggml_tensor* K_ = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, D, T_enc, row_bytes, 0));
        ggml_tensor* V = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, D, T_enc, row_bytes, (size_t)D * sizeof(float)));

        // Reshape for multi-head attention
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, N);
        K_ = ggml_reshape_3d(ctx0, K_, head_dim, n_heads, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));   // (hd, N, n_heads)
        K_ = ggml_cont(ctx0, ggml_permute(ctx0, K_, 0, 2, 1, 3)); // (hd, T_enc, n_heads)
        ggml_tensor* V_h = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        const float scale = 1.0f / std::sqrt((float)head_dim);

        ggml_tensor* attn;
        if (flash_attn) {
            attn = ggml_flash_attn_ext(ctx0, Q, K_, V_h, nullptr, scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, D, N);
        } else {
            ggml_tensor* scores = ggml_mul_mat(ctx0, K_, Q); // (T_enc, N, n_heads)
            scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);
            ggml_tensor* V_p = ggml_cont(ctx0, ggml_permute(ctx0, V_h, 1, 0, 2, 3));
            attn = ggml_mul_mat(ctx0, V_p, scores);
            attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(ctx0, attn, D, N);
        }
        x = mm_bias(ctx0, b.cross_out_w, attn, b.cross_out_b);
    }
    cur = ggml_add(ctx0, residual, x);

    return cur;
}

// Post-processing block (decoders3): norm1 → w_1 → relu → LN → w_2
// Upstream: self_attn=None, src_attn=None, so no residual addition.
static ggml_tensor* build_decoder_post(ggml_context* ctx0, ggml_tensor* cur, const paraformer_decoder_post& b,
                                       float eps) {
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, b.norm1_w, b.norm1_b, eps);
    x = mm_bias(ctx0, b.ffn_l1_w, x, b.ffn_l1_b);
    x = ggml_relu(ctx0, x);
    x = ggml_norm_affine(ctx0, x, b.ffn_norm_w, b.ffn_norm_b, eps); // LN after activation
    x = ggml_mul_mat(ctx0, b.ffn_l2_w, x);
    return x; // no residual — upstream decoders3 has no self_attn so residual is never added
}

// ===========================================================================
// Full inference
// ===========================================================================

static ggml_backend_sched_t paraformer_new_sched(paraformer_context* ctx) {
    ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    return ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
}

static bool paraformer_ensure_sched(paraformer_context* ctx, pf_stage_cache& cache) {
    if (!cache.sched) cache.sched = paraformer_new_sched(ctx);
    return cache.sched != nullptr;
}

static void paraformer_reset_stage(pf_stage_cache& cache) {
    if (cache.sched) ggml_backend_sched_reset(cache.sched);
    ggml_free(cache.ctx);
    cache.ctx = nullptr;
    cache.gf = nullptr;
}

static void paraformer_count_placement(const char* stage, ggml_backend_sched_t sched, ggml_cgraph* graph,
                                      ggml_backend_t backend, int& device_ops, int& total_ops) {
    device_ops = total_ops = 0;
    int other_ops[GGML_OP_COUNT] = {};
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto* node = ggml_graph_node(graph, i);
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
                continue;
            default:
                ++total_ops;
        }
        if (ggml_backend_sched_get_tensor_backend(sched, node) == backend) {
            ++device_ops;
        } else {
            ++other_ops[node->op];
        }
    }
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        if (other_ops[op]) {
            fprintf(stderr, "paraformer: warmup %s %s outside %s: %d\n", stage, ggml_op_name((ggml_op) op),
                    ggml_backend_name(backend), other_ops[op]);
        }
    }
}

// §15: one decoded token with the time and confidence stitch needs.
// Produced only when the caller asks for it (long-audio path); the plain
// string path skips it, so single-utterance output stays bit-identical.
struct pf_token {
    std::string text;     // as emitted (may end in @@ for BPE continuation)
    int64_t start_ms;
    int64_t t_ms;
    float posterior;      // softmax probability of the chosen token
};

static ggml_cgraph* paraformer_build_encoder(paraformer_context* ctx, ggml_context* ctx0, int T_lfr,
                                             int D_lfr_actual, int batch, bool stage) {
    const auto& hp = ctx->model.hp;
    const int D = (int) hp.d_model;
    auto* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input tensor
    ggml_tensor* inp = batch == 1 ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D_lfr_actual, T_lfr)
                                  : ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D_lfr_actual, T_lfr, batch);
    ggml_set_name(inp, "features");
    ggml_set_input(inp);
    // SANMEncoder with input_layer "pe": xs *= sqrt(output_size), then
    // SinusoidalPositionEncoder adds sin|cos of positions 1..T over the input
    // width. Missing before 2026-09: encoder_layer_0 sat at cos ~0.95 against
    // upstream while the text still came out right.
    ggml_tensor* pe_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D_lfr_actual, T_lfr);
    ggml_set_name(pe_in, "enc_pe");
    ggml_set_input(pe_in);

    ggml_tensor* cur = ggml_add(ctx0, ggml_scale(ctx0, inp, std::sqrt((float)D)), pe_in);

    // Encoder: entry block(s)
    int enc_layer_idx = 0;
    for (uint32_t i = 0; i < hp.n_enc_blocks0; i++) {
        core_sanm::BlockParams p;
        p.in_size = (i == 0) ? D_lfr_actual : D;
        p.size = D;
        p.n_heads = (int)hp.n_heads;
        p.head_dim = D / (int)hp.n_heads;
        p.kernel = (int)hp.sanm_kernel;
        p.ln_eps = hp.ln_eps;
        p.flash_attn = ctx->flash_attn;
        cur = core_sanm::build_block(ctx0, cur, T_lfr, ctx->model.enc0[i], p, /*apply_attn_residual=*/false);
        if (stage) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "encoder_layer_%d", enc_layer_idx);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
        }
        enc_layer_idx++;
    }

    // Encoder: main blocks
    for (uint32_t i = 0; i < hp.n_enc_blocks; i++) {
        core_sanm::BlockParams p;
        p.in_size = D;
        p.size = D;
        p.n_heads = (int)hp.n_heads;
        p.head_dim = D / (int)hp.n_heads;
        p.kernel = (int)hp.sanm_kernel;
        p.ln_eps = hp.ln_eps;
        p.flash_attn = ctx->flash_attn;
        cur = core_sanm::build_block(ctx0, cur, T_lfr, ctx->model.enc[i], p, /*apply_attn_residual=*/true);
        if (stage) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "encoder_layer_%d", enc_layer_idx);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
        }
        enc_layer_idx++;
    }

    // Encoder: after_norm
    cur = ggml_norm_affine(ctx0, cur, ctx->model.enc_after_norm_w, ctx->model.enc_after_norm_b, hp.ln_eps);
    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

static std::vector<float> paraformer_encoder_pe(const paraformer_context* ctx, int T_lfr, int Dp) {
    // sinusoidal PE (FunASR SinusoidalPositionEncoder): positions 1..T,
    // depth = input width; first half sin, second half cos
    const int half = Dp / 2;
    const float log_inc = std::log(10000.0f) / (float)(half - 1);
    std::vector<float> inv_freq((size_t)half);
    for (int i = 0; i < half; i++) {
        inv_freq[i] = std::exp(-log_inc * (float)i);
    }
    std::vector<float> pe((size_t)Dp * (size_t)T_lfr);
    float* pe_ptr = pe.data();
    const float* inv_freq_ptr = inv_freq.data();
    auto compute_pe = [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            float* row_sin = pe_ptr + (size_t)t * Dp;
            float* row_cos = row_sin + half;
            const float t_plus_1 = (float)(t + 1);
            for (int i = 0; i < half; i++) {
                const float a = t_plus_1 * inv_freq_ptr[i];
                row_sin[i] = std::sin(a);
                row_cos[i] = std::cos(a);
            }
        }
    };
    if (T_lfr >= 16) {
        core_parallel::for_each_chunk_work(T_lfr, ctx->n_threads, (uint64_t)Dp * 4, 65536, compute_pe);
    } else {
        compute_pe(0, T_lfr);
    }
    return pe;
}

bool paraformer_bench_encoder_pair(paraformer_context* ctx, const float* pcm_a, const float* pcm_b, int n_samples, int reps) {
    if (!ctx || !pcm_a || !pcm_b || n_samples <= 0 || reps <= 0) return false;
    int T = 0, D = 0, T_b = 0, D_b = 0;
    const auto a = paraformer_compute_features(ctx, pcm_a, n_samples, T, D);
    const auto b = paraformer_compute_features(ctx, pcm_b, n_samples, T_b, D_b);
    if (a.empty() || T != T_b || D != D_b || a.size() != b.size()) return false;
    const auto pe = paraformer_encoder_pe(ctx, T, D);

    struct EncoderGraph {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{nullptr, &ggml_free};
        std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched{nullptr, &ggml_backend_sched_free};
        ggml_cgraph* gf = nullptr;
        ggml_tensor* features = nullptr;
        ggml_tensor* positions = nullptr;
        ggml_tensor* output = nullptr;
    } serial, paired;

    auto prepare = [&](EncoderGraph& graph, int width) {
        graph.ctx.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        graph.sched.reset(paraformer_new_sched(ctx));
        if (!graph.ctx || !graph.sched) return false;
        graph.gf = paraformer_build_encoder(ctx, graph.ctx.get(), T, D, width, false);
        if (!ggml_backend_sched_alloc_graph(graph.sched.get(), graph.gf)) return false;
        graph.features = ggml_graph_get_tensor(graph.gf, "features");
        graph.positions = ggml_graph_get_tensor(graph.gf, "enc_pe");
        graph.output = ggml_graph_get_tensor(graph.gf, "encoder_output");
        int on_target, total;
        paraformer_count_placement("encoder probe", graph.sched.get(), graph.gf, ctx->backend, on_target, total);
        printf("encoder_pair width=%d frames=%d device=%s ops=%d/%d splits=%d\n", width, T,
               ggml_backend_name(ctx->backend), on_target, total, ggml_backend_sched_get_n_splits(graph.sched.get()));
        return true;
    };
    if (!prepare(serial, 1) || !prepare(paired, 2)) return false;

    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    struct Timing { double wall_ms = 0; double compute_ms = 0; };
    auto run = [&](EncoderGraph& graph, const std::vector<float>& first, const std::vector<float>* second,
                   float* output, Timing& timing) {
        const auto begin = Clock::now();
        ggml_backend_tensor_set(graph.features, first.data(), 0, first.size() * sizeof(float));
        if (second) {
            ggml_backend_tensor_set(graph.features, second->data(), first.size() * sizeof(float), second->size() * sizeof(float));
        }
        ggml_backend_tensor_set(graph.positions, pe.data(), 0, pe.size() * sizeof(float));
        const auto ready = Clock::now();
        if (ggml_backend_sched_graph_compute(graph.sched.get(), graph.gf) != GGML_STATUS_SUCCESS) return false;
        const auto done = Clock::now();
        ggml_backend_tensor_get(graph.output, output, 0, ggml_nbytes(graph.output));
        timing.wall_ms += elapsed_ms(begin, Clock::now());
        timing.compute_ms += elapsed_ms(ready, done);
        return true;
    };

    const size_t stride = (size_t) ctx->model.hp.d_model * T;
    std::vector<float> reference(2 * stride), output(2 * stride);
    for (int i = -2; i < reps; ++i) {
        Timing one, two;
        const auto& first = i % 2 ? b : a;
        const auto& second = i % 2 ? a : b;
        auto run_serial = [&] {
            return run(serial, first, nullptr, reference.data(), one) &&
                   run(serial, second, nullptr, reference.data() + stride, one);
        };
        auto run_batch = [&] { return run(paired, first, &second, output.data(), two); };
        if (!(i % 2 ? run_batch() && run_serial() : run_serial() && run_batch())) return false;

        double rel_rms[2] = {};
        double max_abs[2] = {};
        for (size_t lane = 0; lane < 2; ++lane) {
            double ref_sum = 0, err_sum = 0;
            for (size_t j = lane * stride; j < (lane + 1) * stride; ++j) {
                const double v = reference[j], error = output[j] - v;
                if (!std::isfinite(v) || !std::isfinite(error)) return false;
                ref_sum += v * v;
                err_sum += error * error;
                max_abs[lane] = std::max(max_abs[lane], std::abs(error));
            }
            rel_rms[lane] = std::sqrt(err_sum / std::max(ref_sum, 1e-30));
        }
        if (i >= 0) {
            printf("encoder_pair rep=%d serial_ms=%.4f batch_ms=%.4f serial_compute_ms=%.4f batch_compute_ms=%.4f "
                   "rel_rms_a=%.8g rel_rms_b=%.8g max_abs_a=%.8g max_abs_b=%.8g\n",
                   i, one.wall_ms, two.wall_ms, one.compute_ms, two.compute_ms,
                   rel_rms[i % 2], rel_rms[1 - i % 2], max_abs[i % 2], max_abs[1 - i % 2]);
        }
        if (std::max(rel_rms[0], rel_rms[1]) > 1e-3) {
            fprintf(stderr, "paraformer: encoder pair parity failed: %.8g %.8g (limit 0.001)\n", rel_rms[0], rel_rms[1]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// §14.5 Token text assembly, shared by the single-utterance and long-audio paths.
//
// The two Paraformer checkpoints use opposite conventions for "this token does
// not start a word", and getting it wrong is invisible on Chinese:
//
//   paraformer-zh (vocab  8404)  trailing "@@"   ->  win@@ | do@@ | w
//   paraformer-en (vocab 10020)  leading U+2581  ->  "▁recirculation"
//
// In the zh vocab "▁" appears zero times and in the en vocab "@@" appears zero
// times, so the convention is a property of the checkpoint, not a guess: scan
// the vocab once at load and record which marker it uses.
//
// The two conventions invert the meaning of an unmarked token:
//   @@ vocab: no trailing @@  ->  the token ends a word
//   ▁  vocab: no leading  ▁   ->  the token *continues* a word ("rec" "ir")
// Reading the en vocab with zh rules splits "rec ir rc ulation" into four words
// and scores four errors where there was one; reading it correctly is the
// difference between 35.9% and 41.2% WER on the same audio.
// ---------------------------------------------------------------------------

enum pf_word_marker {
    PF_MARK_ATAT = 0, // paraformer-zh: trailing "@@" marks continuation
    PF_MARK_SPM  = 1, // paraformer-en: leading U+2581 marks a word start
};

static const char PF_SPM_MARK[] = "\xe2\x96\x81"; // U+2581 LOWER ONE EIGHTH BLOCK

// Decide which convention a vocab follows by counting markers over the whole
// table. Counts, not a single probe: "▁" may also appear inside a few zh tokens
// by accident, and one stray "@@" in an en vocab would flip a first-match test.
// The real tables are 0-vs-thousands, so the majority wins unambiguously.
static pf_word_marker pf_detect_word_marker(const std::vector<std::string>& vocab) {
    size_t spm = 0, atat = 0;
    for (const auto& t : vocab) {
        if (t.compare(0, 3, PF_SPM_MARK) == 0) {
            ++spm;
        }
        if (t.size() >= 2 && t.substr(t.size() - 2) == "@@") {
            ++atat;
        }
    }
    return spm > atat ? PF_MARK_SPM : PF_MARK_ATAT;
}

// Read tokenizer.tokens out of the GGUF and set the marker convention. Both
// init paths need this and they used to do it inline, so a change to one
// silently skipped the other (that is how the English vocab ended up scored
// with zh stitching rules).
static void pf_load_vocab(paraformer_context* ctx) {
    if (gguf_context* gctx = core_gguf::open_metadata(ctx->model_path.c_str())) {
        const int idx = gguf_find_key(gctx, "tokenizer.tokens");
        if (idx >= 0) {
            const int n = gguf_get_arr_n(gctx, idx);
            ctx->vocab.resize((size_t) n);
            for (int i = 0; i < n; i++) {
                ctx->vocab[(size_t) i] = gguf_get_arr_str(gctx, idx, i);
            }
        }
        core_gguf::free_metadata(gctx);
    }
    ctx->word_marker = (int) pf_detect_word_marker(ctx->vocab);
}

// True when tok leaves the current word open (a following token continues it).
static bool pf_token_continues(const std::string& tok, pf_word_marker mark) {
    if (tok.empty()) {
        return false;
    }
    if (mark == PF_MARK_SPM) {
        if (tok.size() >= 2 && tok[0] == '<' && tok.back() == '>') {
            return false; // <blank>/<s>/</s> are dropped, never spliced
        }
        // SentencePiece: the word-start marker is what opens a word; its
        // absence means this piece continues the previous one.
        return tok.compare(0, 3, PF_SPM_MARK) != 0;
    }
    return tok.size() >= 2 && tok.substr(tok.size() - 2) == "@@";
}

// Append one decoded token to out, inserting a space only at a real word
// boundary. prev_continues is pf_token_continues() of the previous token.
static void pf_append_token(std::string& out, const std::string& tok, bool prev_continues, pf_word_marker mark) {
    if (tok.empty()) {
        return;
    }
    if (tok.size() >= 2 && tok[0] == '<' && tok.back() == '>') {
        return; // <blank>/<s>/</s>
    }
    std::string text = tok;
    bool starts_word = false;
    if (text.compare(0, 3, PF_SPM_MARK) == 0) {
        text.erase(0, 3);
        starts_word = true;
    }
    if (text.empty()) {
        return;
    }
    const bool is_cont = mark == PF_MARK_ATAT && text.size() >= 2 &&
                         text.substr(text.size() - 2) == "@@";
    if (is_cont) {
        text.resize(text.size() - 2);
    }
    if (!text.empty() && !out.empty()) {
        const unsigned char first_byte = (unsigned char) text[0];
        const unsigned char last_byte  = (unsigned char) out.back();
        const bool cur_latin  = (first_byte >= 'a' && first_byte <= 'z') ||
                                (first_byte >= 'A' && first_byte <= 'Z');
        const bool prev_latin = (last_byte >= 'a' && last_byte <= 'z') ||
                                (last_byte >= 'A' && last_byte <= 'Z');
        bool space = false;
        if (mark == PF_MARK_SPM) {
            // The ▁ prefix *is* the word boundary; nothing else is. "▁a | f |
            // way" is the word "afway", so continuity must not depend on
            // whether the previous token happened to open a word.
            space = starts_word && cur_latin;
        } else {
            // @@ vocab: a space separates two Latin runs unless the previous
            // token ended in @@ and therefore continues the same word.
            space = cur_latin && prev_latin && !prev_continues;
        }
        if (space) {
            out += ' ';
        }
    }
    out += text;
}

static std::optional<std::string> paraformer_transcribe_impl(paraformer_context* ctx, const float* pcm, int n_samples,
                                              std::vector<float>* staged_out = nullptr, const char* stage = nullptr,
                                              std::vector<pf_token>* out_tokens = nullptr,
                                              paraformer_placement* warmup = nullptr) {
    const auto& hp = ctx->model.hp;
    const int D = (int)hp.d_model;
    paraformer_bench_lap _lap;
    // §13: load whole-word logit bias once (env PF_HOTWORDS); no-op by default.
    paraformer_load_hotwords(ctx);
    paraformer_load_context(ctx);

    // 1. Feature extraction
    int T_lfr = 0, D_lfr_actual = 0;
    std::vector<float> lfr;
    {
        paraformer_bench_stage _b("fbank+lfr");
        lfr = paraformer_compute_features(ctx, pcm, n_samples, T_lfr, D_lfr_actual);
        _lap.mark("feat done");
    }
    if (lfr.empty())
        return warmup ? std::nullopt : std::optional<std::string>("");

    // 2. Build and run encoder graph
    // §176s: reuse cached encoder graph when T_lfr matches (skip graph build).
    // Only for normal inference (stage==null); diff-harness calls rebuild.
    ggml_context* ctx0 = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* cur = nullptr;
    // Tensor metadata only: no_alloc keeps tensor data in the scheduler buffer,
    // so this arena holds objects and the graph node array, not activations.
    const size_t meta_sz = 32 * 1024 * 1024;

    // Encoder graph cache, keyed by the exact LFR length. A hit reuses the
    // allocated graph, so the request pays only input upload and compute.
    pf_stage_cache& ec = ctx->enc_cache;
    const bool can_cache = (stage == nullptr);
    const bool enc_hit = can_cache && ec.gf && ec.k0 == T_lfr;
    if (enc_hit) {
        ++ec.hits;
        gf = ec.gf;
        _lap.mark("enc graph cached");
    } else {
        ++ec.misses;
        _lap.mark("enc graph miss");
        paraformer_reset_stage(ec);
        if (can_cache) {
            if (ec.meta.empty())
                ec.meta.resize(meta_sz);
            ggml_init_params cache_ip = {ec.meta.size(), ec.meta.data(), true};
            ctx0 = ggml_init(cache_ip);
        } else {
            if (ctx->compute_meta.empty())
                ctx->compute_meta.resize(meta_sz);
            ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
            ctx0 = ggml_init(ip);
        }
        gf = paraformer_build_encoder(ctx, ctx0, T_lfr, D_lfr_actual, 1, stage != nullptr);

    }
    _lap.mark("enc graph built");

    // Allocate and run encoder
    if (!paraformer_ensure_sched(ctx, ec)) {
        if (ctx0)
            ggml_free(ctx0);
        return std::nullopt;
    }
    if (!enc_hit) {
        if (!ggml_backend_sched_alloc_graph(ec.sched, gf)) {
            fprintf(stderr, "paraformer: sched alloc encoder graph failed\n");
            paraformer_reset_stage(ec);
            ggml_free(ctx0);
            return std::nullopt;
        }
        if (can_cache) {
            ec.ctx = ctx0;
            ec.gf = gf;
            ec.k0 = T_lfr;
            ctx0 = nullptr;
        }
    }
    _lap.mark("enc sched alloc");
    {
        ggml_tensor* inp = ggml_graph_get_tensor(gf, "features");
        const size_t inp_bytes = ggml_nbytes(inp);
        const size_t lfr_bytes = lfr.size() * sizeof(float);
        if (lfr_bytes > inp_bytes) {
            fprintf(stderr, "paraformer: input tensor too small: %zu bytes vs %zu lfr bytes (T=%d, D=%d)\n", inp_bytes,
                    lfr_bytes, T_lfr, D_lfr_actual);
            paraformer_reset_stage(ec);
            ggml_free(ctx0);
            return std::nullopt;
        }
        if (ctx->verbosity >= 2)
            fprintf(stderr, "paraformer: encoder input OK: T_lfr=%d, D_lfr=%d, bytes=%zu\n", T_lfr, D_lfr_actual,
                    lfr_bytes);
        ggml_backend_tensor_set(inp, lfr.data(), 0, lfr_bytes);
        const auto pe = paraformer_encoder_pe(ctx, T_lfr, D_lfr_actual);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_pe"), pe.data(), 0, pe.size() * sizeof(float));
    }
    if (ctx->verbosity >= 2)
        fprintf(stderr, "paraformer: running encoder graph...\n");
    {
        paraformer_bench_stage _b("encoder");
        if (core_sched_prof::compute(ec.sched, gf, "paraformer") != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "paraformer: encoder graph compute failed\n");
            paraformer_reset_stage(ec);
            ggml_free(ctx0);
            return std::nullopt;
        }
    }
    if (warmup) {
        paraformer_count_placement("encoder", ec.sched, gf, ctx->backend, warmup->encoder_device_ops,
                                   warmup->encoder_total_ops);
    }
    if (ctx->verbosity >= 2)
        fprintf(stderr, "paraformer: encoder done\n");
    _lap.mark("enc compute done");

    // Extract encoder output
    ggml_tensor* enc_result = ggml_graph_get_tensor(gf, "encoder_output");
    if (!enc_result) {
        fprintf(stderr, "paraformer: encoder_output tensor not found in graph\n");
        paraformer_reset_stage(ec);
        ggml_free(ctx0);
        return std::nullopt;
    }
    const size_t enc_bytes = ggml_nbytes(enc_result);
    const size_t enc_expect = (size_t)D * T_lfr * sizeof(float);
    if (ctx->verbosity >= 2)
        fprintf(stderr, "paraformer: enc_result ne=(%lld, %lld), type=%d, nbytes=%zu, expect=%zu\n",
                (long long)enc_result->ne[0], (long long)enc_result->ne[1], (int)enc_result->type, enc_bytes,
                enc_expect);
    if (enc_bytes < enc_expect) {
        fprintf(stderr, "paraformer: encoder_output size mismatch: got %zu, expected %zu (D=%d, T=%d)\n", enc_bytes,
                enc_expect, D, T_lfr);
        paraformer_reset_stage(ec);
        ggml_free(ctx0);
        return std::nullopt;
    }
    std::vector<float> enc_out((size_t)D * T_lfr);
    ggml_backend_tensor_get(enc_result, enc_out.data(), 0, enc_expect);

    // Stage capture from encoder graph
    if (stage && staged_out) {
        std::string sn(stage);
        if (sn == "mel_features") {
            // lfr is the post-LFR feature vector (T_lfr, D_lfr=560) in row-major
            *staged_out = lfr;
            ggml_free(ctx0);
            return "";
        }
        if (sn == "encoder_output") {
            *staged_out = enc_out;
            ggml_free(ctx0);
            return "";
        }
        if (sn.rfind("encoder_layer_", 0) == 0) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, stage);
            if (t) {
                staged_out->resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, staged_out->data(), 0, ggml_nbytes(t));
            }
            ggml_free(ctx0);
            return "";
        }
    }

    if (ctx0)
        ggml_free(ctx0);

    // 3. CIF predictor (CPU)
    // enc_out flat storage: ggml tensor ne=(D, T_lfr), element (d,t) at
    // flat index t*D + d.  This IS row-major (T, D), so no transpose needed.
    std::vector<float> acoustic_embeds;
    int N_tokens = 0;
    std::vector<int> fire_frames_vec;
    {
        paraformer_bench_stage _b("cif_predict");
        // §15: tokens need their fire frames too, so request them when either
        // the caller asked for frames or for token records.
        cif_predict(ctx, enc_out.data(), T_lfr, D, acoustic_embeds, N_tokens, out_tokens ? &fire_frames_vec : nullptr);
        _lap.mark("cif done");
    }
    if (ctx->verbosity >= 1)
        fprintf(stderr, "paraformer: CIF predicted %d tokens from T_lfr=%d\n", N_tokens, T_lfr);
    if (N_tokens <= 0) {
        if (!warmup) return "";
        N_tokens = 1;
        acoustic_embeds.assign(D, 0.0f);
    }

    // Stage capture: acoustic_embeds (row-major (N, D))
    if (stage && staged_out && std::strcmp(stage, "acoustic_embeds") == 0) {
        *staged_out = acoustic_embeds;
        return "";
    }

    // 4. Decoder graph, cached by the exact pair of shapes it embeds.
    pf_stage_cache& dc = ctx->dec_cache;
    const bool dec_cacheable = (stage == nullptr);
    const bool dec_hit = dec_cacheable && dc.gf && dc.k0 == T_lfr && dc.k1 == N_tokens;
    ggml_tensor* dec_inp = nullptr;
    ggml_tensor* enc_tensor = nullptr;
    if (dec_hit) {
        ++dc.hits;
        gf = dc.gf;
        dec_inp = ggml_graph_get_tensor(gf, "acoustic_embeds");
        enc_tensor = ggml_graph_get_tensor(gf, "enc_for_decoder");
    } else {
        ++dc.misses;
        paraformer_reset_stage(dc);
        if (dec_cacheable) {
            if (dc.meta.empty())
                dc.meta.resize(meta_sz);
            ggml_init_params cache_ip = {dc.meta.size(), dc.meta.data(), true};
            ctx0 = ggml_init(cache_ip);
        } else {
            ggml_init_params ip2 = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
            ctx0 = ggml_init(ip2);
        }
        gf = ggml_new_graph_custom(ctx0, 32768, false);

        dec_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, N_tokens);
        ggml_set_name(dec_inp, "acoustic_embeds");
        ggml_set_input(dec_inp);

        enc_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_lfr);
        ggml_set_name(enc_tensor, "enc_for_decoder");
        ggml_set_input(enc_tensor);
    }

    const int n_heads = (int)hp.n_heads;
    const int head_dim = D / n_heads;

    if (!dec_hit) {
        cur = dec_inp;
        for (uint32_t i = 0; i < hp.n_dec_blocks; i++) {
            cur = build_decoder_block(ctx0, cur, enc_tensor, N_tokens, T_lfr, ctx->model.dec[i], D, n_heads, head_dim,
                                      hp.ln_eps, ctx->flash_attn);
            if (stage) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "decoder_layer_%u", i);
                ggml_set_name(cur, nm);
                ggml_set_output(cur);
            }
        }

        // Post block
        if (ctx->model.dec_post.norm1_w) {
            cur = build_decoder_post(ctx0, cur, ctx->model.dec_post, hp.ln_eps);
        }

        // after_norm → output_layer
        cur = ggml_norm_affine(ctx0, cur, ctx->model.dec_after_norm_w, ctx->model.dec_after_norm_b, hp.ln_eps);
        if (stage) {
            ggml_set_name(cur, "decoder_output");
            ggml_set_output(cur);
        }
        cur = mm_bias(ctx0, ctx->model.dec_output_w, cur, ctx->model.dec_output_b); // (vocab, N)
        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
        _lap.mark("dec graph built");
    } else {
        _lap.mark("dec graph cached");
    }

    // Allocate and run decoder
    if (!paraformer_ensure_sched(ctx, dc)) {
        if (ctx0)
            ggml_free(ctx0);
        return std::nullopt;
    }
    if (!dec_hit) {
        if (!ggml_backend_sched_alloc_graph(dc.sched, gf)) {
            fprintf(stderr, "paraformer: sched alloc decoder graph failed\n");
            paraformer_reset_stage(dc);
            ggml_free(ctx0);
            return std::nullopt;
        }
        if (dec_cacheable) {
            dc.ctx = ctx0;
            dc.gf = gf;
            dc.k0 = T_lfr;
            dc.k1 = N_tokens;
            ctx0 = nullptr;
        }
    }
    _lap.mark("dec sched alloc");

    // Set decoder inputs: acoustic_embeds from CIF is row-major (N, D).
    // ggml tensor ne=(D, N) stores element (d,n) at flat index n*D+d, which
    // is the same layout. No transposition needed.
    ggml_backend_tensor_set(dec_inp, acoustic_embeds.data(), 0, (size_t)D * N_tokens * sizeof(float));
    ggml_backend_tensor_set(enc_tensor, enc_out.data(), 0, enc_out.size() * sizeof(float));
    {
        paraformer_bench_stage _b("decoder");
        if (core_sched_prof::compute(dc.sched, gf, "paraformer") != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "paraformer: decoder graph compute failed\n");
            paraformer_reset_stage(dc);
            ggml_free(ctx0);
            return std::nullopt;
        }
    }
    if (warmup) {
        paraformer_count_placement("decoder", dc.sched, gf, ctx->backend, warmup->decoder_device_ops,
                                   warmup->decoder_total_ops);
    }

    // Stage capture from decoder graph
    if (stage && staged_out) {
        std::string sn(stage);
        if (sn == "decoder_output" || sn.rfind("decoder_layer_", 0) == 0) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, stage);
            if (t) {
                staged_out->resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, staged_out->data(), 0, ggml_nbytes(t));
            }
            ggml_free(ctx0);
            return "";
        }
    }

    _lap.mark("dec compute done");

    // 5. Argmax decoding
    paraformer_bench_stage _b_argmax("argmax");
    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    const int V = (int)hp.vocab_size;
    std::vector<float> logits_data((size_t)V * N_tokens);
    ggml_backend_tensor_get(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));
    _lap.mark("logits readback");

    // §13: optional diagnostics — how many positions to dump top-K logits at.
    const char* dump_env = std::getenv("PF_DUMP_LOGITS");
    const int dump_n = dump_env ? std::atoi(dump_env) : 0;
    const bool apply_bias = !ctx->hotword_bonus.empty() || !ctx->hotword_seq.empty() || !ctx->ctx_trie.empty();
    std::vector<float> biased_row; // only allocated when dumping with bias on
    if (dump_n > 0 && apply_bias)
        biased_row.resize(V);

    // Section 28: active context-trie nodes. The root stays active so a phrase
    // can start anywhere; its children carry a configurable start gain (default
    // 0) while continuations carry the full phrase gain.
    float ctx_start_gain = 0.0f;
    if (const char* sg = std::getenv("PF_CONTEXT_START")) {
        ctx_start_gain = (float) std::atof(sg);
    }
    std::vector<int> ctx_active;
    if (!ctx->ctx_trie.empty()) {
        ctx_active.push_back(0);
    }
    std::vector<std::pair<int, float>> ctx_here;

    std::string result;
    const pf_word_marker mark = (pf_word_marker) ctx->word_marker;
    bool prev_was_bpe_cont = false; // true if previous token left the word open
    int prev_token = -1;            // §13: token chosen at n-1, for seq rules
    // Choose first, rewrite spans, then render. Without replacement rules the
    // output is byte-identical to the single-pass version.
    std::vector<pf_choice> chosen;
    chosen.reserve(N_tokens);
    std::vector<float> pos_posterior(out_tokens ? (size_t) N_tokens : 0, 0.0f);
    for (int n = 0; n < N_tokens; n++) {
        const float* row = logits_data.data() + (size_t)n * V;
        // §13: prefix-conditional gains for this position (depends on prev token).
        std::vector<std::pair<int, float>> seq_here;
        if (apply_bias && prev_token >= 0) {
            for (const auto& r : ctx->hotword_seq)
                if (r.prefix == prev_token)
                    seq_here.push_back({ r.target, r.gain });
        }
        // §28: gains for tokens that continue an active context phrase. Root
        // children (phrase starts) take the maximum, not the sum, so the start
        // bias stays bounded no matter how many phrases are registered.
        ctx_here.clear();
        std::vector<std::pair<int, float>> ctx_start;
        for (int node : ctx_active) {
            for (const auto& kv : ctx->ctx_trie[(size_t) node].child) {
                const float g = ctx->ctx_trie[(size_t) kv.second].gain;
                if (node == 0) {
                    bool merged = false;
                    for (auto& s : ctx_start) {
                        if (s.first == kv.first) {
                            s.second = std::max(s.second, g);
                            merged = true;
                            break;
                        }
                    }
                    if (!merged) {
                        ctx_start.push_back({ kv.first, g });
                    }
                } else if (g > 0.0f) {
                    ctx_here.push_back({ kv.first, g });
                }
            }
        }
        if (ctx_start_gain > 0.0f) {
            // Absolute logits, not a multiplier: the start bias must stay small
            // or it dominates the acoustic scores (measured: 3x gain destroyed
            // every transcript).
            for (const auto& s : ctx_start) {
                ctx_here.push_back({ s.first, ctx_start_gain });
            }
        }
        auto bias_of = [&](int v) -> float {
            float b = ctx->hotword_bonus.empty() ? 0.0f : ctx->hotword_bonus[(size_t) v];
            for (const auto& kv : seq_here)
                if (kv.first == v)
                    b += kv.second;
            for (const auto& kv : ctx_here)
                if (kv.first == v)
                    b += kv.second;
            return b;
        };
        int best = 0;
        float best_val = -1e30f;
        for (int v = 0; v < V; v++) {
            float val = row[v];
            if (apply_bias)
                val += bias_of(v);
            if (val > best_val) {
                best_val = val;
                best = v;
            }
        }
        if (dump_n > 0 && n < dump_n) {
            if (apply_bias) {
                for (int v = 0; v < V; ++v)
                    biased_row[(size_t) v] = row[v] + bias_of(v);
                paraformer_dump_logits(ctx, biased_row.data(), V, n, 5);
            } else {
                paraformer_dump_logits(ctx, row, V, n, 5);
            }
        }
        if (!ctx->ctx_trie.empty()) {
            std::vector<int> next_active;
            next_active.push_back(0); // root: a new phrase may start later
            for (int node : ctx_active) {
                for (const auto& kv : ctx->ctx_trie[(size_t) node].child) {
                    if (kv.first == best) {
                        next_active.push_back(kv.second);
                    }
                }
            }
            std::sort(next_active.begin(), next_active.end());
            next_active.erase(std::unique(next_active.begin(), next_active.end()), next_active.end());
            ctx_active.swap(next_active);
        }
        prev_token = best;
        chosen.push_back({best, (double) n, (double) n});
        // Skip special tokens: 0=<blank>, 1=<s>, 2=</s>, last=<unk>
        if (best <= 2 || best == V - 1)
            continue;
        if ((size_t)best < ctx->vocab.size()) {
            // §15: record time/confidence for the long-audio stitcher. Posterior
            // is a numerically stable softmax over the (biased) row; cost is
            // only paid when out_tokens is requested, so the plain path is
            // unchanged.
            if (out_tokens) {
                double sum = 0.0;
                for (int v = 0; v < V; ++v)
                    sum += std::exp((double) (row[v] + (apply_bias ? bias_of(v) : 0.0f) - best_val));
                pos_posterior[(size_t) n] = (float) (1.0 / sum);
            }
        }
    }

    if (!ctx->hotword_repl.empty()) {
        const char* dbg = std::getenv("PF_REPLACE_LOG");
        pf_apply_replacements(chosen, logits_data.data(), V, ctx->hotword_repl, ctx->vocab, dbg != nullptr);
    }

    if (out_tokens) {
        const double frame_step_ms = (double) hp.lfr_n * hp.frame_shift_ms;
        auto time_at = [&](double row) {
            const size_t lo = (size_t) row;
            const size_t hi = std::min(lo + 1, fire_frames_vec.size() - 1);
            const double frame = fire_frames_vec[lo] + (row - lo) * (fire_frames_vec[hi] - fire_frames_vec[lo]);
            return (int64_t) std::lround(frame * frame_step_ms);
        };
        for (const auto& choice : chosen) {
            const int id = choice.id;
            if (id <= 2 || id == V - 1 || (size_t) id >= ctx->vocab.size()) {
                continue;
            }
            const size_t src = choice.logits_row();
            float posterior = pos_posterior[src];
            if (choice.rewritten) {
                const float* row = logits_data.data() + src * V;
                const float mx = *std::max_element(row, row + V);
                double sum = 0.0;
                for (int v = 0; v < V; ++v) {
                    sum += std::exp((double) (row[v] - mx));
                }
                posterior = (float) (std::exp((double) (row[id] - mx)) / sum);
            }
            out_tokens->push_back({ctx->vocab[(size_t) id], time_at(choice.start_row), time_at(choice.end_row), posterior});
        }
    }

    if (const char* vsub = std::getenv("PF_DUMP_VOCAB")) {
        for (size_t i = 0; i < ctx->vocab.size(); ++i) {
            if (ctx->vocab[i].find(vsub) != std::string::npos)
                fprintf(stderr, "pf-vocab: %zu %s\n", i, ctx->vocab[i].c_str());
        }
    }

    if (std::getenv("PF_DUMP_TOKENS")) {
        std::string ids;
        for (size_t n = 0; n < chosen.size(); ++n) {
            const int id = chosen[n].id;
            if (id >= 0 && (size_t) id < ctx->vocab.size())
                ids += ctx->vocab[(size_t) id] + "|";
        }
        fprintf(stderr, "pf-tokens: %s\n", ids.c_str());
    }

    for (size_t n = 0; n < chosen.size(); n++) {
        const int best = chosen[n].id;
        if (best < 0 || best <= 2 || best == V - 1 || (size_t) best >= ctx->vocab.size())
            continue;
        const std::string& tok = ctx->vocab[best];
        pf_append_token(result, tok, prev_was_bpe_cont, mark);
        prev_was_bpe_cont = pf_token_continues(tok, mark);
    }

    ggml_free(ctx0);

    return result;
}

// ===========================================================================
// §15 Long audio: window segmentation + token stitching
//
// A UDP is not one utterance. The decoder's CIF and cross-attention cost grow
// with length and the encoder's global attention makes very long inputs
// unreliable, so audio beyond PARAFORMER_LONG_MS is cut into overlapping
// windows, each decoded independently, then stitched by global token time.
//
// Design follows design doc §2.1/§2.3 but only the parts that survive
// measurement:
//   * fixed window W with overlap O (default 8 s / 1 s), each window decoded
//     with its own CIF fire positions;
//   * tokens are placed on a global time axis; a token whose centre falls in a
//     window's *core* (not its overlap margin) is kept; duplicates in the
//     overlap are resolved by keeping the higher-posterior one.
//   * BPE words are never split: a token ending in @@ drags its continuation
//     with it, and merging is only allowed at word boundaries.
//
// Windows are decoded in the same process/context sequentially; parallelism is
// out of scope here (the server already parallelises whole utterances).
// ===========================================================================

struct pf_long_cfg {
    bool enabled = true;     // PARAFORMER_LONG=0 disables, restoring single-shot
    int win_ms = 8000;       // window length
    int overlap_ms = 1000;   // overlap between adjacent windows
    int min_ms = 15000;      // only engage beyond this length
};

static pf_long_cfg paraformer_long_cfg() {
    pf_long_cfg c;
    if (const char* e = std::getenv("PARAFORMER_LONG"))
        c.enabled = std::atoi(e) != 0;
    if (const char* e = std::getenv("PARAFORMER_WIN_MS"))
        c.win_ms = std::max(2000, std::atoi(e));
    if (const char* e = std::getenv("PARAFORMER_OVERLAP_MS"))
        c.overlap_ms = std::max(0, std::atoi(e));
    if (const char* e = std::getenv("PARAFORMER_LONG_MIN_MS"))
        c.min_ms = std::max(1000, std::atoi(e));
    if (c.overlap_ms >= c.win_ms)
        c.overlap_ms = c.win_ms / 4;
    return c;
}


// Decode a long utterance window-by-window and stitch the transcripts. Returns
// the joined text; `out_tokens` (optional) receives the merged token stream.
//
// Ownership is per *token*: a token belongs to the window whose core contains
// it. The cores tile the timeline exactly — core_e of window i equals core_s of
// window i+1 — so every token is kept by exactly one window and none is lost.
//
// A per-*word* rule looks tidier but is wrong, and measuring it proved it: a
// word whose first token sits just inside core_e while its tail spills past it
// is rejected as "still open, the next window owns it", yet the next window's
// core starts at core_e, so nobody owns it and the word vanishes (19.8 s English
// clip: `ro@@ of` -> dropped, `on` -> dropped, "sun roof a little" came out
// "sun of the little", 106 -> 103 tokens).
//
// BPE words cut at a boundary survive anyway: if window i keeps `se@@` and
// window i+1 keeps `at`, the plain @@ join below reassembles `seat`. The two
// halves may come from different windows and disagree (`seed` vs `seat`), which
// is the documented cost of context windows, not a stitch bug.
static std::optional<std::string> paraformer_transcribe_long(paraformer_context* ctx, const float* pcm, int n_samples,
                                                            const pf_long_cfg& cfg, std::vector<pf_token>* out_tokens) {
    const pf_word_marker mark = (pf_word_marker) ctx->word_marker;
    const int sr = 16000;
    const int64_t total_ms = (int64_t) n_samples * 1000 / sr;
    const int64_t win = cfg.win_ms, ov = cfg.overlap_ms;
    const int64_t step = win - ov;
    if (win <= 0 || step <= 0 || total_ms <= 0)
        return {};

    // Window plan: [start, end) in ms. Each window is decoded whole; its core
    // is [start + ov/2, end - ov/2] except at the utterance edges, so overlap
    // tokens are only kept from one side.
    struct Win { int64_t s, e, core_s, core_e; };
    std::vector<Win> wins;
    for (int64_t s = 0; s < total_ms; s += step) {
        int64_t e = std::min(total_ms, s + win);
        Win w;
        w.s = s;
        w.e = e;
        w.core_s = (s == 0) ? 0 : s + ov / 2;
        w.core_e = (e >= total_ms) ? total_ms : e - ov / 2;
        wins.push_back(w);
        if (e >= total_ms)
            break;
    }

    // Decode every window, offset its tokens into global time, keep those in
    // the window's core.
    std::vector<pf_token> merged;
    std::vector<size_t> seams; // merged index where each window's block starts
    for (size_t wi = 0; wi < wins.size(); ++wi) {
        const Win& w = wins[wi];
        const int off = (int) ((int64_t) w.s * sr / 1000);
        const int len = (int) (((int64_t) w.e - w.s) * sr / 1000);
        if (off < 0 || len <= 0 || off + len > n_samples)
            continue;
        std::vector<pf_token> toks;
        if (!paraformer_transcribe_impl(ctx, pcm + off, len, nullptr, nullptr, &toks)) {
            return std::nullopt;
        }
        seams.push_back(merged.size());
        // PF_DUMP_WINDOWS=1 prints each window's own hypothesis, so a bad
        // stitch can be told apart from a bad window decode. Must run before
        // the keep loop: that loop moves tokens out and leaves the retained
        // ones empty in `toks`.
        if (const char* dump = std::getenv("PF_DUMP_WINDOWS")) {
            if (std::atoi(dump) != 0) {
                std::string all;
                bool pc = false;
                for (const auto& t : toks) {
                    pf_append_token(all, t.text, pc, mark);
                    pc = pf_token_continues(t.text, mark);
                }
                fprintf(stderr, "paraformer:   win%zu raw (%zu tok): %s\n", wi + 1, toks.size(), all.c_str());
            }
        }
        size_t kept = 0;
        for (auto& t : toks) {
            t.start_ms += w.s;
            t.t_ms += w.s;
            if (t.t_ms < w.core_s || t.t_ms >= w.core_e)
                continue;
            merged.push_back(std::move(t));
            kept++;
        }
        if (ctx->verbosity >= 1)
            fprintf(stderr, "paraformer: long window %zu/%zu [%lld,%lld) core [%lld,%lld) -> %zu tokens, %zu kept\n",
                    wi + 1, wins.size(), (long long) w.s, (long long) w.e, (long long) w.core_s,
                    (long long) w.core_e, toks.size(), kept);
    }

    // Cores tile the timeline exactly, so a token is normally owned by exactly
    // one window. The remaining seam risk is the same speech decoded twice with
    // fire times straddling the boundary: if both copies carry the same text
    // within 120 ms, keep the more confident one.
    //
    // Two guards keep this from damaging real output:
    //   * only standalone words are eligible — a token ending in @@ or one that
    //     follows a @@ belongs to a BPE word, and deleting half of it is worse
    //     than the duplicate;
    //   * only the first couple of tokens of each window are compared against
    //     the last couple of the previous one, so genuine repeats inside a
    //     window ("谢谢", "看看") are never touched.
    if (merged.size() > 1 && seams.size() > 1) {
        const auto is_cont = [](const std::string& t) {
            return t.size() >= 2 && t.compare(t.size() - 2, 2, "@@") == 0;
        };
        const auto is_standalone = [&](size_t i) {
            if (is_cont(merged[i].text))
                return false;
            return i == 0 || !is_cont(merged[i - 1].text);
        };
        std::vector<char> drop(merged.size(), 0);
        for (size_t si = 1; si < seams.size(); ++si) {
            const size_t base = seams[si];
            if (base == 0 || base >= merged.size())
                continue;
            for (size_t j = base; j < std::min(merged.size(), base + 2); ++j) {
                if (drop[j] || !is_standalone(j))
                    continue;
                const size_t lo = base > 2 ? base - 2 : 0;
                for (size_t i = base; i-- > lo;) {
                    if (drop[i] || !is_standalone(i) || merged[i].text != merged[j].text)
                        continue;
                    if (std::llabs((long long) (merged[j].t_ms - merged[i].t_ms)) >= 120)
                        continue;
                    if (merged[j].posterior > merged[i].posterior)
                        drop[i] = 1;
                    else
                        drop[j] = 1;
                    break;
                }
            }
        }
        size_t removed = 0;
        std::vector<pf_token> survivors;
        survivors.reserve(merged.size());
        for (size_t i = 0; i < merged.size(); ++i) {
            if (drop[i]) {
                removed++;
                continue;
            }
            survivors.push_back(std::move(merged[i]));
        }
        merged.swap(survivors);
        if (removed && ctx->verbosity >= 1)
            fprintf(stderr, "paraformer: stitch dropped %zu duplicate token(s) at seams\n", removed);
    }

    // Sort by global time so the stitch order matches speech order.
    std::stable_sort(merged.begin(), merged.end(), [](const pf_token& a, const pf_token& b) { return a.t_ms < b.t_ms; });
    std::vector<pf_token>& keep = merged;

    std::string out;
    bool prev_cont = false;
    for (const auto& t : keep) {
        pf_append_token(out, t.text, prev_cont, mark);
        prev_cont = pf_token_continues(t.text, mark);
    }
    if (out_tokens)
        *out_tokens = std::move(keep);
    return out;
}

// ===========================================================================
// Public API
// ===========================================================================

// Single entry point for every caller: pick single-shot or windowed decoding
// from the clip length. Short clips take exactly the old path, so their
// transcripts stay bit-identical; PARAFORMER_LONG=0 forces single-shot
// everywhere.
static bool paraformer_use_long(const pf_long_cfg& cfg, int n_samples) {
    if (!cfg.enabled || n_samples <= 0)
        return false;
    const int64_t total_ms = (int64_t) n_samples * 1000 / 16000;
    return total_ms >= cfg.min_ms;
}

static std::optional<std::string> paraformer_run(paraformer_context* ctx, const float* pcm, int n_samples,
                                                std::vector<pf_token>* tokens = nullptr) {
    const pf_long_cfg lc = paraformer_long_cfg();
    if (paraformer_use_long(lc, n_samples))
        return paraformer_transcribe_long(ctx, pcm, n_samples, lc, tokens);
    return paraformer_transcribe_impl(ctx, pcm, n_samples, nullptr, nullptr, tokens);
}

paraformer_context_params paraformer_context_default_params() {
    return {4, 0, true, true};
}

// ---------------------------------------------------------------------------
// Shared-weight API: load the GGUF once, then create independent contexts.
// ---------------------------------------------------------------------------

paraformer_weights* paraformer_weights_load(const char* path, int verbosity) {
    if (!path || !path[0])
        return nullptr;

    auto* w = new paraformer_weights();
    w->model_path = path;
    if (!core_gguf::read_file(path, w->gguf_bytes)) {
        fprintf(stderr, "paraformer: shared read failed for '%s'\n", path);
        delete w;
        return nullptr;
    }

    if (gguf_context* gctx = core_gguf::open_metadata(path)) {
        int idx = gguf_find_key(gctx, "tokenizer.tokens");
        if (idx >= 0) {
            int n = gguf_get_arr_n(gctx, idx);
            w->vocab.resize(n);
            for (int i = 0; i < n; i++)
                w->vocab[i] = gguf_get_arr_str(gctx, idx, i);
        }
        core_gguf::free_metadata(gctx);
    }

    if (verbosity >= 1)
        fprintf(stderr, "paraformer: shared weights loaded (%zu bytes, vocab %zu)\n", w->gguf_bytes.size(),
                w->vocab.size());
    return w;
}

void paraformer_weights_free(paraformer_weights* w) {
    delete w;
}

paraformer_context* paraformer_init_shared(paraformer_weights* w, paraformer_context_params params) {
    if (!w)
        return nullptr;
    auto* ctx = new paraformer_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->flash_attn = params.flash_attn;
    ctx->verbosity = params.verbosity;

    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "paraformer: shared ctx failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);
    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : ctx->backend_cpu;
    if (!ctx->backend) {
        paraformer_free(ctx);
        return nullptr;
    }

    ctx->model_path = w->model_path;
    // Materialize this context's weights from the single in-memory GGUF byte
    // buffer into its own backend (HTP tiled device buffer, or host for CPU).
    if (!core_gguf::load_weights_bytes(w->gguf_bytes, ctx->backend, "paraformer", ctx->wl)) {
        fprintf(stderr, "paraformer: shared ctx failed to materialize weights\n");
        if (ctx->backend && ctx->backend != ctx->backend_cpu)
            ggml_backend_free(ctx->backend);
        ggml_backend_free(ctx->backend_cpu);
        delete ctx;
        return nullptr;
    }

    if (!paraformer_load_model(ctx)) {
        paraformer_free(ctx);
        return nullptr;
    }
    ctx->vocab = w->vocab; // cheap: ~8404 short strings copied once at init
    ctx->word_marker = (int) pf_detect_word_marker(ctx->vocab);
    if (ctx->verbosity >= 1)
        fprintf(stderr, "paraformer: word marker = %s (vocab %zu)\n",
                ctx->word_marker == (int) PF_MARK_SPM ? "leading-U+2581" : "trailing-@@",
                ctx->vocab.size());
    return ctx;
}

paraformer_context* paraformer_init_from_file(const char* path, paraformer_context_params params) {
    if (!path || !path[0])
        return nullptr;
    auto* ctx = new paraformer_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->flash_attn = params.flash_attn;
    ctx->verbosity = params.verbosity;

    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "paraformer: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);
    // Backend selection (§232). Weights already load onto ctx->backend via
    // core_gguf::load_weights, so pointing that at a GPU backend is the whole fix.
    //   * CRISPASR_PARAFORMER_GPU=1 forces GPU on ANY backend; =0 forces CPU.
    //   * default: GPU on CUDA/Vulkan, CPU on Metal. Kaggle P100 A/B: identical
    //     transcript, 2.15x wall (slow OpenBLAS baseline). On M1 (Accelerate) it
    //     is neutral — small model / short audio, launch-bound (LEARNING 34) — so
    //     Metal stays CPU unless forced. Mirrors LEARNING 34's is_metal gate.
    const char* gpu_env = std::getenv("CRISPASR_PARAFORMER_GPU");
    const bool force_gpu = gpu_env && std::atoi(gpu_env) != 0;
    const bool force_cpu = gpu_env && std::atoi(gpu_env) == 0;
    ctx->backend = ctx->backend_cpu;
    if (!force_cpu && (force_gpu || params.use_gpu)) {
        ggml_backend_t gpu = crispasr_init_gpu_backend();
        if (gpu) {
            bool is_metal = false;
#if defined(GGML_USE_METAL)
            is_metal = core_cpu_backend::is_metal(gpu);
#endif
            if (!is_metal || force_gpu) {
                ctx->backend = gpu;
                if (ctx->verbosity >= 1)
                    fprintf(stderr, "paraformer: GPU backend enabled (%s)\n", ggml_backend_name(ctx->backend));
            } else {
                ggml_backend_free(gpu);
                if (ctx->verbosity >= 1)
                    fprintf(stderr, "paraformer: GPU default limited to CUDA/Vulkan (Metal neutral); set "
                                    "CRISPASR_PARAFORMER_GPU=1 to force\n");
            }
        }
    }

    ctx->model_path = path;
    if (!core_gguf::load_weights(path, ctx->backend, "paraformer", ctx->wl)) {
        fprintf(stderr, "paraformer: failed to load '%s'\n", path);
        paraformer_free(ctx);
        return nullptr;
    }

    if (!paraformer_load_model(ctx)) {
        paraformer_free(ctx);
        return nullptr;
    }

    pf_load_vocab(ctx);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "paraformer: loaded %zu tensors, vocab %zu (%s), enc %u+%u blocks, dec %u blocks\n",
                ctx->wl.tensors.size(), ctx->vocab.size(),
                ctx->word_marker == (int) PF_MARK_SPM ? "leading-U+2581" : "trailing-@@",
                ctx->model.hp.n_enc_blocks0, ctx->model.hp.n_enc_blocks, ctx->model.hp.n_dec_blocks);
    }

    return ctx;
}

void paraformer_cache_stats(const paraformer_context* ctx, unsigned long long* enc_hit, unsigned long long* enc_miss,
                            unsigned long long* dec_hit, unsigned long long* dec_miss) {
    if (!ctx) {
        return;
    }
    if (enc_hit) *enc_hit = ctx->enc_cache.hits;
    if (enc_miss) *enc_miss = ctx->enc_cache.misses;
    if (dec_hit) *dec_hit = ctx->dec_cache.hits;
    if (dec_miss) *dec_miss = ctx->dec_cache.misses;
}

void paraformer_free(paraformer_context* ctx) {
    if (!ctx)
        return;
    for (pf_stage_cache* c : {&ctx->enc_cache, &ctx->dec_cache}) {
        if (c->ctx)
            ggml_free(c->ctx);
        if (c->sched)
            ggml_backend_sched_free(c->sched);
    }
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

bool paraformer_warmup(paraformer_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0) return false;
    paraformer_placement placement = {};
    if (!paraformer_transcribe_impl(ctx, samples, n_samples, nullptr, nullptr, nullptr, &placement) ||
        placement.encoder_device_ops == 0 || placement.decoder_device_ops == 0) {
        return false;
    }
    ctx->warmup_placement = placement;
    return true;
}

const char* paraformer_backend_name(const paraformer_context* ctx) {
    return ggml_backend_dev_name(ggml_backend_get_device(ctx->backend));
}

paraformer_placement paraformer_warmup_placement(const paraformer_context* ctx) {
    return ctx->warmup_placement;
}

char* paraformer_transcribe(paraformer_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    auto result = paraformer_run(ctx, samples, n_samples);
    if (!result) return nullptr;
    const auto& s = *result;
    char* buf = (char*)std::malloc(s.size() + 1);
    if (buf) {
        std::memcpy(buf, s.data(), s.size());
        buf[s.size()] = '\0';
    }
    return buf;
}

paraformer_result* paraformer_transcribe_with_timestamps(paraformer_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    std::vector<pf_token> tokens;
    auto result = paraformer_run(ctx, samples, n_samples, &tokens);
    if (!result) return nullptr;
    const auto& s = *result;

    auto* r = (paraformer_result*)std::malloc(sizeof(paraformer_result));
    if (!r)
        return nullptr;
    r->text = (char*)std::malloc(s.size() + 1);
    r->n_chars = (int) tokens.size();
    r->char_times_cs = r->n_chars ? (int32_t*) std::malloc(tokens.size() * sizeof(int32_t)) : nullptr;
    if (!r->text || (r->n_chars && !r->char_times_cs)) {
        paraformer_result_free(r);
        return nullptr;
    }
    std::memcpy(r->text, s.c_str(), s.size() + 1);
    for (int i = 0; i < r->n_chars; ++i) {
        r->char_times_cs[i] = (int32_t) (tokens[i].t_ms / 10);
    }
    return r;
}

paraformer_tokens* paraformer_transcribe_tokens(paraformer_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    std::vector<pf_token> toks;
    auto result = paraformer_run(ctx, samples, n_samples, &toks);
    if (!result) return nullptr;
    const auto& s = *result;
    auto* r = (paraformer_tokens*) std::malloc(sizeof(paraformer_tokens));
    if (!r)
        return nullptr;
    r->text = (char*) std::malloc(s.size() + 1);
    if (r->text) {
        std::memcpy(r->text, s.data(), s.size());
        r->text[s.size()] = '\0';
    }
    r->n_tokens = (int) toks.size();
    r->tokens = r->n_tokens ? (paraformer_token*) std::calloc((size_t) r->n_tokens, sizeof(paraformer_token)) : nullptr;
    if (!r->text || (r->n_tokens && !r->tokens)) {
        std::free(r->tokens);
        std::free(r->text);
        std::free(r);
        return nullptr;
    }
    for (int i = 0; i < r->n_tokens; ++i) {
        r->tokens[i].text = (char*) std::malloc(toks[(size_t) i].text.size() + 1);
        if (!r->tokens[i].text) {
            paraformer_tokens_free(r);
            return nullptr;
        }
        std::memcpy(r->tokens[i].text, toks[(size_t) i].text.data(), toks[(size_t) i].text.size());
        r->tokens[i].text[toks[(size_t) i].text.size()] = '\0';
        r->tokens[i].t_ms = (int32_t) toks[(size_t) i].t_ms;
        r->tokens[i].start_ms = (int32_t) toks[(size_t) i].start_ms;
        r->tokens[i].posterior = toks[(size_t) i].posterior;
    }
    return r;
}

void paraformer_tokens_free(paraformer_tokens* r) {
    if (!r)
        return;
    for (int i = 0; i < r->n_tokens; ++i)
        std::free(r->tokens[i].text);
    std::free(r->tokens);
    std::free(r->text);
    std::free(r);
}

void paraformer_result_free(paraformer_result* r) {
    if (!r)
        return;
    std::free(r->text);
    std::free(r->char_times_cs);
    std::free(r);
}

float* paraformer_extract_stage(paraformer_context* ctx, const float* samples, int n_samples, const char* stage_name,
                                int* n_out) {
    if (n_out)
        *n_out = 0;
    if (!ctx || !samples || n_samples <= 0 || !stage_name)
        return nullptr;

    if (std::strcmp(stage_name, "generated_text") == 0) {
        auto result = paraformer_run(ctx, samples, n_samples);
        if (!result) return nullptr;
        const auto& txt = *result;
        char* buf = (char*)std::malloc(txt.size() + 1);
        if (!buf)
            return nullptr;
        std::memcpy(buf, txt.data(), txt.size());
        buf[txt.size()] = '\0';
        if (n_out)
            *n_out = (int)txt.size();
        // cppcheck-suppress invalidPointerCast
        return (float*)buf; // caller casts back to char* (generated_text path)
    }

    std::vector<float> staged;
    (void)paraformer_transcribe_impl(ctx, samples, n_samples, &staged, stage_name);
    if (staged.empty())
        return nullptr;
    float* out = (float*)std::malloc(staged.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, staged.data(), staged.size() * sizeof(float));
    if (n_out)
        *n_out = (int)staged.size();
    return out;
}
