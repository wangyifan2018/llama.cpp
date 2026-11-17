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
// params.use_gpu selects HTP vs CPU (subject to the same env gates as
// paraformer_init_from_file).
paraformer_context* paraformer_init_shared(paraformer_weights* w, paraformer_context_params params);

paraformer_context* paraformer_init_from_file(const char* path, paraformer_context_params params);
void paraformer_free(paraformer_context* ctx);

// Transcribe audio (16 kHz mono PCM). Returns a malloc'd string; caller frees.
char* paraformer_transcribe(paraformer_context* ctx, const float* samples, int n_samples);

// Warm up the runtime outside the measured path: creates the HTP session and
// registers the encoder/decoder graphs once, so the first real request does not
// pay the one-time DSP graph-dispatch cost (~1 s on SM6650). Samples should be
// representative of the expected audio length. Safe to call more than once.
void paraformer_warmup(paraformer_context* ctx, const float* samples, int n_samples);

// Variant that returns per-character CIF timestamps (centiseconds from audio start).
// Each emitted character maps to the encoder frame where CIF fired.
struct paraformer_result {
    char* text;
    int32_t* char_times_cs; // per-character end time in centiseconds
    int n_chars;            // number of entries in char_times_cs
};
struct paraformer_result* paraformer_transcribe_with_timestamps(paraformer_context* ctx, const float* samples,
                                                                int n_samples);
void paraformer_result_free(struct paraformer_result* r);

// Stage-capture API for the diff harness.
float* paraformer_extract_stage(paraformer_context* ctx, const float* samples, int n_samples, const char* stage_name,
                                int* n_out);

#ifdef __cplusplus
}
#endif
