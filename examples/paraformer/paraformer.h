// src/paraformer.h — FunASR Paraformer (NAR ASR) runtime.
//
// Paraformer is a non-autoregressive ASR model:
//   Kaldi-fbank + LFR + CMVN
//   → 50-block SANM encoder (1 entry block 560→512 + 49 main blocks)
//   → CIF predictor (Conv1d + Linear → sigmoid → accumulation)
//   → 16-block NAR decoder (FSMN self-attn + cross-attn + FFN)
//   → argmax over vocab (8404 characters)
//
// The encoder reuses core_sanm::build_block(). The decoder is new — each
// block has three sub-layers (upstream order):
//   1. norm1 → FFN (w_1 → ReLU → LayerNorm → w_2)
//   2. norm2 → FSMN depthwise conv (no Q/K/V self-attention)
//   3. norm3 → cross-attention (Q from decoder, fused K+V from encoder)

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct paraformer_context;
struct paraformer_keyword_bias;

// Graph-cache counters: encoder hits/misses, decoder hits/misses.
void paraformer_cache_stats(const paraformer_context* ctx, unsigned long long* enc_hit, unsigned long long* enc_miss,
                            unsigned long long* dec_hit, unsigned long long* dec_miss);
struct paraformer_weights;

struct paraformer_context_params {
    int n_threads;
    int verbosity; // 0 = silent, 1 = normal, 2 = debug
    bool flash_attn;
    bool use_gpu;
};

paraformer_context_params paraformer_context_default_params();

// Load a GGUF file once into a canonical host-buffer weight set (weights +
// vocabulary + model path). Multiple inference contexts can be created from it
// without re-reading the file. The canonical weights stay read-only.
paraformer_weights* paraformer_weights_load(const char* path, int verbosity);
void paraformer_weights_free(paraformer_weights* w);

// Create an inference context backed by a shared weight set. Weights are
// cloned into the context's own backend buffer (host for CPU, tiled device
// buffer for HTP), so contexts are independent and safe to run in parallel.
// When params.use_gpu is true, initialization requires the selected device.
paraformer_context* paraformer_init_shared(paraformer_weights* w, paraformer_context_params params);

paraformer_context* paraformer_init_from_file(const char* path, paraformer_context_params params);
void paraformer_free(paraformer_context* ctx);

// Transcribe audio (16 kHz mono PCM). NULL means failure. Empty text is valid.
// The caller frees the returned string.
char* paraformer_transcribe(paraformer_context* ctx, const float* samples, int n_samples);

// Build immutable Mandarin phrase hints against the model vocabulary. Returns NULL on invalid input.
struct paraformer_keyword_bias* paraformer_keyword_bias_create(const struct paraformer_context* ctx,
                                                              const char* const* words, size_t count);
void paraformer_keyword_bias_free(struct paraformer_keyword_bias* bias);
char* paraformer_transcribe_biased(struct paraformer_context* ctx, const float* samples, int n_samples,
                                 const struct paraformer_keyword_bias* bias);

// Literal phrase rewrites applied to the final transcript (TSV: from <TAB> to).
// A rule fires only on an exact case-insensitive match, so it cannot invent a
// rewrite on unrelated text. Returns the number of rules loaded.
int paraformer_set_text_rules(paraformer_context* ctx, const char* path);

// Warm up the runtime outside the measured path: creates the HTP session and
// registers the encoder/decoder graphs once, so the first real request does not
// pay the one-time DSP graph-dispatch cost (~1 s on SM6650). Samples should be
// representative of the expected audio length. Safe to call more than once.
bool paraformer_warmup(paraformer_context* ctx, const float* samples, int n_samples);

struct paraformer_placement {
    int encoder_device_ops;
    int encoder_total_ops;
    int decoder_device_ops;
    int decoder_total_ops;
};

// Backend name and compute placement from the last successful warmup.
const char* paraformer_backend_name(const paraformer_context* ctx);
struct paraformer_placement paraformer_warmup_placement(const paraformer_context* ctx);

// Measure two equal-length 16 kHz utterances with serial and width-two encoder graphs.
bool paraformer_bench_encoder_pair(paraformer_context* ctx, const float* pcm_a, const float* pcm_b, int n_samples, int reps);

// End timestamps for the emitted tokens, in centiseconds from audio start.
struct paraformer_result {
    char* text;
    int32_t* char_times_cs; // token end times in centiseconds
    int n_chars;            // number of entries in char_times_cs
};
struct paraformer_result* paraformer_transcribe_with_timestamps(paraformer_context* ctx, const float* samples,
                                                                int n_samples);
void paraformer_result_free(struct paraformer_result* r);

// Token-level output: one entry per emitted token with its CIF fire time and
// posterior. Chinese tokens are characters, English ones are BPE pieces (a piece
// ending in "@@" continues the same word), so callers can build word timestamps
// and clause boundaries from the gaps between entries.
struct paraformer_token {
    char* text;
    int32_t t_ms;       // End of the source CIF span, in milliseconds.
    float posterior;   // For a rewrite, acoustic probability at its nearest original CIF position.
    int32_t start_ms;   // Start of the source CIF span, in milliseconds.
};

struct paraformer_tokens {
    char* text; // the same transcript the plain API returns
    struct paraformer_token* tokens;
    int n_tokens;
};

// Returns NULL on failure. Free with paraformer_tokens_free().
struct paraformer_tokens* paraformer_transcribe_tokens(paraformer_context* ctx, const float* samples, int n_samples);
struct paraformer_tokens* paraformer_transcribe_tokens_biased(struct paraformer_context* ctx, const float* samples,
                                                             int n_samples, const struct paraformer_keyword_bias* bias);
void paraformer_tokens_free(struct paraformer_tokens* r);

// Stage-capture API for the diff harness.
float* paraformer_extract_stage(paraformer_context* ctx, const float* samples, int n_samples, const char* stage_name,
                                int* n_out);

#ifdef __cplusplus
}
#endif
