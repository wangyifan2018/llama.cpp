// sherpa-onnx control for the X-ASR streaming benchmark. Same 20 ms packet
// pacing, 1,200 ms tail padding and greedy text as the llama-x-asr example.

#include "c-api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static std::string quoted(const std::string & text) {
    std::string s = "\"";
    for (unsigned char c : text) {
        if (c == '\\' || c == '"') {
            s += '\\';
            s += (char) c;
        } else if (c < 32) {
            s += ' ';
        } else {
            s += (char) c;
        }
    }
    return s + "\"";
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL_DIR THREADS PACED WAV...\n", argv[0]);
        return 1;
    }
    const std::string root   = argv[1];
    const int         threads = std::atoi(argv[2]);
    const bool        paced   = std::atoi(argv[3]) != 0;

    const std::string tokens  = root + "/tokens.txt";
    const std::string encoder = root + "/encoder.int8.onnx";
    const std::string decoder = root + "/decoder.onnx";
    const std::string joiner  = root + "/joiner.int8.onnx";

    const auto init_begin = Clock::now();
    SherpaOnnxOnlineRecognizerConfig config;
    memset(&config, 0, sizeof(config));
    config.feat_config = { 16000, 80 };
    config.model_config.tokens = tokens.c_str();
    config.model_config.num_threads = threads;
    config.model_config.provider = "cpu";
    config.model_config.transducer = { encoder.c_str(), decoder.c_str(), joiner.c_str() };
    config.model_config.model_type = "zipformer2";
    config.decoding_method = "greedy_search";
    config.enable_endpoint = 0;
    const SherpaOnnxOnlineRecognizer * rec = SherpaOnnxCreateOnlineRecognizer(&config);
    if (!rec) {
        fprintf(stderr, "onnx-bench: cannot create recognizer\n");
        return 2;
    }
    fprintf(stderr, "{\"init_ms\":%.3f,\"kind\":\"xasr-onnx\",\"threads\":%d}\n", ms(init_begin), threads);

    for (int arg = 4; arg < argc; ++arg) {
        const SherpaOnnxWave * wave = SherpaOnnxReadWave(argv[arg]);
        if (!wave || wave->num_samples <= 0) {
            fprintf(stderr, "onnx-bench: cannot read %s\n", argv[arg]);
            return 3;
        }
        const double duration = double(wave->num_samples) / wave->sample_rate;
        const int    step     = wave->sample_rate / 50;
        const auto   begin    = Clock::now();
        double       compute = 0, max_decode = 0;
        int          chunks = 0, chunks_stream = 0, chunks_commit = 0;
        bool         in_flush = false;
        std::vector<double> chunk_ms;
        std::string  text;

        const SherpaOnnxOnlineStream * stream = SherpaOnnxCreateOnlineStream(rec);
        auto drain = [&]() {
            while (SherpaOnnxIsOnlineStreamReady(rec, stream)) {
                const auto t = Clock::now();
                SherpaOnnxDecodeOnlineStream(rec, stream);
                const double elapsed = ms(t);
                compute += elapsed;
                max_decode = std::max(max_decode, elapsed);
                chunk_ms.push_back(elapsed);
                ++chunks;
                (in_flush ? chunks_commit : chunks_stream)++;
                const SherpaOnnxOnlineRecognizerResult * r = SherpaOnnxGetOnlineStreamResult(rec, stream);
                if (r) {
                    text = r->text;
                    SherpaOnnxDestroyOnlineRecognizerResult(r);
                }
            }
        };
        for (int at = 0; at < wave->num_samples; at += step) {
            const int n = std::min(step, wave->num_samples - at);
            const double consumed = 1000.0 * (at + n) / wave->sample_rate;
            if (paced) {
                std::this_thread::sleep_until(begin + std::chrono::microseconds((long long) (consumed * 1000)));
            }
            SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, wave->samples + at, n);
            drain();
        }
        // 1,200 ms of virtual silence, the same tail padding as llama-x-asr.
        in_flush = true;
        const auto commit_begin = Clock::now();
        std::vector<float> pad((size_t) wave->sample_rate * 6 / 5, 0.0f);
        SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, pad.data(), pad.size());
        SherpaOnnxOnlineStreamInputFinished(stream);
        drain();
        const double commit_ms = ms(commit_begin);
        const double wall_ms   = ms(begin);
        SherpaOnnxDestroyOnlineStream(stream);

        printf("{\"kind\":\"xasr-onnx\",\"threads\":%d,\"wave\":%s,\"paced\":%s,"
               "\"audio_ms\":%.3f,\"wall_ms\":%.3f,\"tail_ms\":%.3f,\"commit_ms\":%.3f,"
               "\"compute_ms\":%.3f,\"max_decode_ms\":%.3f,\"chunks\":%d,\"chunks_stream\":%d,"
               "\"chunks_commit\":%d,\"text\":%s,\"chunk_ms\":[",
               threads, quoted(argv[arg]).c_str(), paced ? "true" : "false",
               duration * 1000.0, wall_ms, wall_ms - duration * 1000.0, commit_ms,
               compute, max_decode, chunks, chunks_stream, chunks_commit, quoted(text).c_str());
        for (size_t i = 0; i < chunk_ms.size(); ++i) {
            printf("%s%.3f", i ? "," : "", chunk_ms[i]);
        }
        printf("]}\n");
        fflush(stdout);
        SherpaOnnxFreeWave(wave);
    }
    SherpaOnnxDestroyOnlineRecognizer(rec);
    return 0;
}
