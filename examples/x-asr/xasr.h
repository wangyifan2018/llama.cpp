#pragma once

#include <string>
#include <vector>

struct xasr_context;
struct xasr_stream;

struct xasr_context_params {
    const char * device = "CPU";
    int n_threads = 2;
    int n_streams = 1;      // independent encoder state slots in one context
    int tail_pad_ms = 1200; // negative selects the minimal phase-aware tail padding
    int verbosity = 1;
    bool joiner_on_device = true; // run the joiner projection on the selected backend
};

// Per-request timing, all in milliseconds of in-process wall time.
struct xasr_stats {
    double fbank_ms = 0;      // incremental feature extraction
    double feat_up_ms = 0;    // feature window upload
    double mask_build_ms = 0; // attention mask fill
    double mask_up_ms = 0;    // attention mask upload
    double compute_ms = 0;    // encoder graph submit plus any completion wait
    double submit_ms = 0;     // DSP batch hand-off inside compute_ms
    double wait_ms = 0;       // DSP completion wait inside compute_ms
    double readback_ms = 0;   // encoder projection download
    double greedy_ms = 0;     // predictor, joiner and greedy selection
    double detok_ms = 0;      // token to text
    double append_wall_ms = 0;
    double commit_wall_ms = 0;
    int chunks_stream = 0;    // chunks that ran before commit
    int chunks_commit = 0;    // chunks triggered by the tail padding
    int dsp_batches = 0;      // DSP command batches submitted
    int joiner_dev_calls = 0; // joiner graph runs on the backend
    double joiner_dev_ms = 0; // time inside those runs
    double joiner_pred_ms = 0; // predictor updates on the host
    int pad_frames = 0;       // feature frames added by the commit tail padding
    int n_frames = 0;         // feature frames at commit
    std::vector<double> chunk_ms; // per-chunk upload, compute and readback
};

// Valid until the stream is freed. Reset when the stream is created.
const xasr_stats * xasr_stream_stats(const xasr_stream * stream);

// Call ggml_backend_load_all before loading. Destroy streams before the context.
xasr_context * xasr_init_from_file(const char * path, xasr_context_params params);
void xasr_free(xasr_context * context);

// Up to n_streams active streams. Serialize calls on a context.
// Audio is normalized mono PCM at 16 kHz.
xasr_stream * xasr_stream_init(xasr_context * context);
bool xasr_append(xasr_stream * stream, const float * samples, int count);
bool xasr_commit(xasr_stream * stream, std::string & text);
// Start an append or commit. While pending, only step or destroy this stream.
// Other streams in the context can run between steps.
enum class xasr_step_result { pending, done, error };
bool xasr_begin_step(xasr_stream * stream, const float * samples, int count, bool commit);
xasr_step_result xasr_step(xasr_stream * stream, std::string & text);
void xasr_stream_free(xasr_stream * stream);
