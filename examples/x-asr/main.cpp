#include "xasr.h"

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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include <time.h>
#include <sys/resource.h>

using xasr_clock = std::chrono::steady_clock;

static double process_cpu_ms() {
    timespec t{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf DEVICE WAV... [--realtime] [--pad-ms N] [--threads N] [--streams N] [--joiner-host] [--probe-encoder-batch]\n", argv[0]);
        return 1;
    }
    xasr_context_params params;
    params.device = argv[2];
    bool realtime = false;
    bool concurrent = false;
    bool probe_batch = false;
    int hold_ms = 0;
    std::vector<const char *> waves;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--realtime") == 0) {
            realtime = true;
        } else if (std::strcmp(argv[i], "--pad-ms") == 0 && i + 1 < argc) {
            params.tail_pad_ms = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            params.n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
            params.n_streams = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--probe-encoder-batch") == 0) {
            probe_batch = true;
        } else if (std::strcmp(argv[i], "--concurrent") == 0) {
            concurrent = true;
        } else if (std::strcmp(argv[i], "--hold-ms") == 0 && i + 1 < argc) {
            hold_ms = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--joiner-device") == 0) {
            params.joiner_on_device = true;
        } else if (std::strcmp(argv[i], "--joiner-host") == 0) {
            params.joiner_on_device = false;
        } else {
            waves.push_back(argv[i]);
        }
    }
    if (waves.empty()) {
        return 1;
    }
    if (probe_batch) {
        if (concurrent || realtime || (waves.size() != 2 && waves.size() != 4)) {
            return 1;
        }
        params.n_streams = static_cast<int>(waves.size());
        params.joiner_on_device = false;
    }
    ggml_backend_load_all();
    const auto init_begin = xasr_clock::now();
    const std::unique_ptr<xasr_context, decltype(&xasr_free)> model(xasr_init_from_file(argv[1], params), xasr_free);
    if (!model) {
        return 2;
    }
    fprintf(stderr, "xasr: init %.1f ms\n", std::chrono::duration<double, std::milli>(xasr_clock::now() - init_begin).count());
    if (probe_batch) {
        std::vector<std::vector<float>> pcm;
        for (const char * path : waves) {
            ma_decoder wav;
            auto config = ma_decoder_config_init(ma_format_f32, 1, 16000);
            if (ma_decoder_init_file(path, &config, &wav) != MA_SUCCESS) {
                fprintf(stderr, "xasr: cannot decode %s\n", path);
                return 3;
            }
            const std::unique_ptr<ma_decoder, decltype(&ma_decoder_uninit)> audio(&wav, ma_decoder_uninit);
            auto & samples = pcm.emplace_back();
            while (samples.size() < 16000 * 60) {
                float block[4096];
                ma_uint64 count = 0;
                const size_t capacity = std::min<size_t>(4096, 16000 * 60 - samples.size());
                const auto result = ma_decoder_read_pcm_frames(audio.get(), block, capacity, &count);
                if (result != MA_SUCCESS && result != MA_AT_END) {
                    return 5;
                }
                samples.insert(samples.end(), block, block + count);
                if (result == MA_AT_END || count == 0) {
                    break;
                }
            }
        }
        return xasr_probe_encoder_batch(model.get(), pcm) ? 0 : 8;
    }
    if (concurrent) {
        // Feed every stream the same packets, then commit them all at the same
        // endpoint. Streams are served in order, so stream i pays the commits of
        // the streams before it; that is the head-of-line cost of one worker.
        const int n = params.n_streams;
        for (const char * wave : waves) {
            ma_decoder wav;
            auto config = ma_decoder_config_init(ma_format_f32, 1, 16000);
            if (ma_decoder_init_file(wave, &config, &wav) != MA_SUCCESS) {
                fprintf(stderr, "xasr: cannot decode %s\n", wave);
                return 3;
            }
            const std::unique_ptr<ma_decoder, decltype(&ma_decoder_uninit)> audio(&wav, ma_decoder_uninit);
            std::vector<std::unique_ptr<xasr_stream, decltype(&xasr_stream_free)>> streams;
            for (int i = 0; i < n; ++i) {
                streams.emplace_back(xasr_stream_init(model.get()), xasr_stream_free);
                if (!streams.back()) {
                    fprintf(stderr, "xasr: cannot open stream %d of %d\n", i, n);
                    return 4;
                }
            }
            const auto start = xasr_clock::now();
            ma_uint64 frames = 0;
            ma_uint64 total = 0;
            if (ma_decoder_get_length_in_pcm_frames(audio.get(), &total) != MA_SUCCESS || total == 0) {
                total = 0;
            }
            const ma_uint64 split = total > (ma_uint64) (hold_ms * 16) ? total - (ma_uint64) (hold_ms * 16) : total;
            std::vector<double> endpoint_ms((size_t) n, 0.0);
            std::vector<std::string> texts((size_t) n);
            bool split_done = false;
            for (;;) {
                float samples[320];
                ma_uint64 n_read = 0;
                const ma_result result = ma_decoder_read_pcm_frames(audio.get(), samples, 320, &n_read);
                if (result != MA_SUCCESS && result != MA_AT_END) {
                    fprintf(stderr, "xasr: cannot read %s\n", wave);
                    return 5;
                }
                if (n_read > 0) {
                    frames += n_read;
                    if (realtime) {
                        std::this_thread::sleep_until(start + std::chrono::microseconds(frames * 1000000 / 16000));
                    }
                    // Until the split, every stream receives audio. After it only
                    // the last stream does, while the others are committed.
                    const int active = (hold_ms > 0 && frames > split && split_done) ? 1 : n;
                    for (int i = n - active; i < n; ++i) {
                        if (!xasr_append(streams[(size_t) i].get(), samples, static_cast<int>(n_read))) {
                            return 6;
                        }
                    }
                    if (hold_ms > 0 && !split_done && frames >= split) {
                        split_done = true;
                        const auto ep = xasr_clock::now();
                        for (int i = 0; i < n - 1; ++i) {
                            if (!xasr_commit(streams[(size_t) i].get(), texts[(size_t) i])) {
                                return 7;
                            }
                            endpoint_ms[(size_t) i] = std::chrono::duration<double, std::milli>(xasr_clock::now() - ep).count();
                        }
                    }
                }
                if (result == MA_AT_END || n_read == 0) {
                    break;
                }
            }
            const auto endpoint = xasr_clock::now();
            for (int i = (hold_ms > 0 ? n - 1 : 0); i < n; ++i) {
                if (!xasr_commit(streams[(size_t) i].get(), texts[(size_t) i])) {
                    return 7;
                }
                endpoint_ms[(size_t) i] = std::chrono::duration<double, std::milli>(xasr_clock::now() - endpoint).count();
            }
            for (int i = 0; i < n; ++i) {
                printf("%s\tstream=%d\tendpoint_to_final_ms=%.3f\t%s\n", wave, i, endpoint_ms[(size_t) i], texts[(size_t) i].c_str());
            }
            rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            printf("%s\tstreams=%d\taudio_ms=%.3f\tsession_ms=%.3f\trss_kib=%ld\n", wave, n, frames / 16.0,
                   std::chrono::duration<double, std::milli>(xasr_clock::now() - start).count(), (long) usage.ru_maxrss);
        }
        return 0;
    }
    for (const char * wave : waves) {
        ma_decoder wav;
        auto config = ma_decoder_config_init(ma_format_f32, 1, 16000);
        if (ma_decoder_init_file(wave, &config, &wav) != MA_SUCCESS) {
            fprintf(stderr, "xasr: cannot decode %s\n", wave);
            return 3;
        }
        const std::unique_ptr<ma_decoder, decltype(&ma_decoder_uninit)> audio(&wav, ma_decoder_uninit);
        const auto stream_begin = xasr_clock::now();
        const std::unique_ptr<xasr_stream, decltype(&xasr_stream_free)> stream(xasr_stream_init(model.get()), xasr_stream_free);
        if (!stream) {
            return 4;
        }
        if (std::getenv("XASR_TIMING")) {
            fprintf(stderr, "xasr: stream init %.2f ms\n", std::chrono::duration<double, std::milli>(xasr_clock::now() - stream_begin).count());
        }
        const double cpu_begin = process_cpu_ms();
        const auto start = xasr_clock::now();
        ma_uint64 frames = 0;
        for (;;) {
            float samples[320];
            ma_uint64 n_read = 0;
            const ma_result result = ma_decoder_read_pcm_frames(audio.get(), samples, 320, &n_read);
            if (result != MA_SUCCESS && result != MA_AT_END) {
                fprintf(stderr, "xasr: cannot read %s\n", wave);
                return 5;
            }
            if (n_read > 0) {
                frames += n_read;
                if (realtime) {
                    std::this_thread::sleep_until(start + std::chrono::microseconds(frames * 1000000 / 16000));
                }
                if (!xasr_append(stream.get(), samples, static_cast<int>(n_read))) {
                    return 6;
                }
            }
            if (result == MA_AT_END || n_read == 0) {
                break;
            }
        }
        const auto commit = xasr_clock::now();
        std::string text;
        if (!xasr_commit(stream.get(), text)) {
            return 7;
        }
        const auto now = xasr_clock::now();
        const double audio_ms = frames / 16.0;
        const double total_ms = std::chrono::duration<double, std::milli>(now - start).count();
        const double flush_ms = std::chrono::duration<double, std::milli>(now - commit).count();
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        printf("%s\taudio_ms=%.3f\twall_ms=%.3f\tflush_ms=%.3f\tcpu_ms=%.3f\trss_kib=%ld",
               wave, audio_ms, total_ms, flush_ms, process_cpu_ms() - cpu_begin, (long) usage.ru_maxrss);
        if (realtime) {
            printf("\ttail_ms=%.3f", total_ms - audio_ms);
        } else if (frames > 0) {
            printf("\trtf=%.4f", total_ms / audio_ms);
        }
        printf("\t%s\n", text.c_str());
    }
}
