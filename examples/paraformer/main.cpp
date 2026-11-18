#include "paraformer.h"
#include "core/segment_limits.h"

#include "ggml-backend.h"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using pf_clock = std::chrono::steady_clock;

static bool read_wav(const char * path, std::vector<float> & samples) {
    ma_decoder decoder;
    const auto config = ma_decoder_config_init(ma_format_f32, 1, 16000);
    if (ma_decoder_init_file(path, &config, &decoder) != MA_SUCCESS) {
        fprintf(stderr, "paraformer: cannot decode %s\n", path);
        return false;
    }
    const std::unique_ptr<ma_decoder, decltype(&ma_decoder_uninit)> audio(&decoder, ma_decoder_uninit);
    samples.clear();
    samples.reserve(16000 * 30);
    for (;;) {
        float block[4096];
        ma_uint64 count = 0;
        const auto res = ma_decoder_read_pcm_frames(audio.get(), block, 4096, &count);
        if (res != MA_SUCCESS && res != MA_AT_END) {
            fprintf(stderr, "paraformer: read error on %s\n", path);
            return false;
        }
        samples.insert(samples.end(), block, block + count);
        if (res == MA_AT_END || count == 0)
            break;
    }
    return !samples.empty();
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s MODEL.gguf WAV... [--device NAME] [--threads N] [--warmup FILE] [--timestamps] [--quiet]\n"
                "       [--hotwords FILE] [--text-rules FILE] [--dump-logits N] [--bench-encoder-pair REPS]\n",
                argv[0]);
        return 1;
    }

    paraformer_context_params params = paraformer_context_default_params();
    params.use_gpu = false;
    params.flash_attn = false;
    params.verbosity = 0;   // per-clip results go to stdout; stderr stays quiet
    bool timestamps = false;
    int encoder_pair_reps = 0;
    const char * warmup_file = nullptr;
    const char * requested_device = nullptr;
    std::vector<const char *> waves;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            requested_device = argv[++i];
            setenv("PARAFORMER_DEVICE", requested_device, 1);
            setenv("CRISPASR_PARAFORMER_GPU", "1", 1);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            params.n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_file = argv[++i];
        } else if (std::strcmp(argv[i], "--timestamps") == 0) {
            timestamps = true;
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            params.verbosity = 0;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            params.verbosity = 1;
        } else if (std::strcmp(argv[i], "--hotwords") == 0 && i + 1 < argc) {
            setenv("PF_HOTWORDS", argv[++i], 1);
        } else if (std::strcmp(argv[i], "--text-rules") == 0 && i + 1 < argc) {
            setenv("PF_TEXT_RULES", argv[++i], 1);
        } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            setenv("PF_DUMP_LOGITS", argv[++i], 1);
        } else if (std::strcmp(argv[i], "--bench-encoder-pair") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            const long reps = std::strtol(value, &end, 10);
            if (end == value || *end || reps < 1 || reps > 1000) return 1;
            encoder_pair_reps = (int) reps;
        } else {
            waves.push_back(argv[i]);
        }
    }
    if (waves.empty()) {
        return 1;
    }

    ggml_backend_load_all();
    const auto init_begin = pf_clock::now();
    std::unique_ptr<paraformer_context, decltype(&paraformer_free)> model(
        paraformer_init_from_file(argv[1], params), paraformer_free);
    if (!model) {
        return 2;
    }
    // The device request is checked against what actually initialised, so a
    // typo fails loudly instead of silently running on CPU. Compare device
    // names, not the raw request: crispasr_init_gpu_backend() resolves the
    // virtual-session form `HTP0:0` to the registered device `HTP0` and picks
    // the session through GGML_HEXAGON_DEVICES, so an exact strcmp rejects a
    // request it just honoured. That failure looks like a transport error --
    // the session is up and all the ggml-hex logging is normal -- and it was
    // recorded as an HTP outage once already; see design doc §31.
    if (requested_device) {
        const char* actual = paraformer_backend_name(model.get());
        const size_t prefix = std::strcspn(requested_device, ":");
        const bool same_device = std::strncmp(requested_device, actual, prefix) == 0 && actual[prefix] == '\0';
        if (!same_device) {
            fprintf(stderr, "paraformer: requested %s, initialized %s\n", requested_device, actual);
            return 2;
        }
    }
    const double init_ms =
        std::chrono::duration<double, std::milli>(pf_clock::now() - init_begin).count();
    fprintf(stderr, "paraformer: init %.1f ms\n", init_ms);

    if (encoder_pair_reps) {
        std::vector<float> a, b;
        if (waves.size() != 2 || !read_wav(waves[0], a) || !read_wav(waves[1], b) ||
            a.size() != b.size() || a.size() > pf_segment::max_samples) {
            fprintf(stderr, "paraformer: encoder pair probe needs two equal-length WAVs of at most 30 seconds\n");
            return 3;
        }
        return paraformer_bench_encoder_pair(model.get(), a.data(), b.data(), (int) a.size(), encoder_pair_reps) ? 0 : 6;
    }

    if (warmup_file) {
        std::vector<float> warm;
        if (!read_wav(warmup_file, warm)) {
            return 3;
        }
        const auto wbegin = pf_clock::now();
        if (!paraformer_warmup(model.get(), warm.data(), (int) warm.size())) {
            fprintf(stderr, "paraformer: warmup failed on %s\n", paraformer_backend_name(model.get()));
            return 6;
        }
        const double wms =
            std::chrono::duration<double, std::milli>(pf_clock::now() - wbegin).count();
        fprintf(stderr, "paraformer: warmup %.1f ms\n", wms);
    }

    for (const char * wave : waves) {
        std::vector<float> samples;
        if (!read_wav(wave, samples)) {
            return 3;
        }

        const auto begin = pf_clock::now();
        std::unique_ptr<char, decltype(&std::free)> text(nullptr, &std::free);
        std::unique_ptr<paraformer_result, decltype(&paraformer_result_free)> result(nullptr, paraformer_result_free);
        std::unique_ptr<paraformer_tokens, decltype(&paraformer_tokens_free)> toks(nullptr, paraformer_tokens_free);
        if (timestamps) {
            toks.reset(paraformer_transcribe_tokens(model.get(), samples.data(), (int) samples.size()));
            if (toks) {
                text.reset(toks->text);
                toks->text = nullptr;
            }
        } else {
            text.reset(paraformer_transcribe(model.get(), samples.data(), (int) samples.size()));
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(pf_clock::now() - begin).count();
        if (!text) {
            fprintf(stderr, "paraformer: inference failed on %s\n", wave);
            return 6;
        }
        const double audio_ms = samples.size() * 1000.0 / 16000.0;
        printf("%s\taudio_ms=%.3f\twall_ms=%.3f\trtf=%.4f", wave, audio_ms, wall_ms,
               audio_ms > 0 ? wall_ms / audio_ms : 0.0);
        if (timestamps && toks) {
            printf("\tn_tokens=%d", toks->n_tokens);
            if (toks->n_tokens > 0) {
                printf("\tts_cs=");
                for (int i = 0; i < toks->n_tokens; ++i)
                    printf("%s%d", i ? "," : "", toks->tokens[i].t_ms);
                printf("\ttok=");
                for (int i = 0; i < toks->n_tokens; ++i)
                    printf("%s%s", i ? "|" : "", toks->tokens[i].text);
            }
        }
        printf("\t%s\n", text.get());
    }
    if (getenv("PF_CACHE_STATS")) {
        unsigned long long eh = 0, em = 0, dh = 0, dm = 0;
        paraformer_cache_stats(model.get(), &eh, &em, &dh, &dm);
        fprintf(stderr, "pf-cache: enc hit=%llu miss=%llu dec hit=%llu miss=%llu\n", eh, em, dh, dm);
    }
    return 0;
}
