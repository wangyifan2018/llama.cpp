// Speaker embedding via sherpa-onnx's extractor.
//
// The device already ships libsherpa-onnx-c-api.so and libonnxruntime.so, so the
// gate can run there without pulling in onnxruntime headers. sherpa's extractor
// does the fbank itself and expects a 3D-Speaker style model; the CAM++ ONNX we
// export uses the same input/output names ("feats" -> "embedding").
//
//   spk_embed_sherpa MODEL.onnx WAV... [--repeat N]

#include "c-api.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool read_wav(const char* path, std::vector<float>& out, int& rate) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> bytes((size_t) n);
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0) {
        return false;
    }
    size_t pos = 12;
    int channels = 0, bits = 0;
    const char* data = nullptr;
    size_t data_bytes = 0;
    while (pos + 8 <= bytes.size()) {
        char id[5] = {0};
        std::memcpy(id, bytes.data() + pos, 4);
        uint32_t size = 0;
        std::memcpy(&size, bytes.data() + pos + 4, 4);
        size_t body = pos + 8;
        if (body + size > bytes.size()) {
            break;
        }
        if (std::strcmp(id, "fmt ") == 0 && size >= 16) {
            uint16_t ch = 0, bps = 0;
            std::memcpy(&ch, bytes.data() + body + 2, 2);
            std::memcpy(&rate, bytes.data() + body + 4, 4);
            std::memcpy(&bps, bytes.data() + body + 14, 2);
            channels = ch;
            bits = bps;
        } else if (std::strcmp(id, "data") == 0) {
            data = bytes.data() + body;
            data_bytes = size;
        }
        pos = body + size + (size & 1);
    }
    if (!data || channels != 1 || bits != 16 || rate != 16000) {
        return false;
    }
    const size_t cnt = data_bytes / 2;
    out.resize(cnt);
    const int16_t* p = reinterpret_cast<const int16_t*>(data);
    for (size_t i = 0; i < cnt; ++i) {
        out[i] = (float) p[i] / 32768.0f;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.onnx WAV... [--repeat N]\n", argv[0]);
        return 1;
    }
    int repeat = 1;
    bool dump = false;
    std::vector<const char*> wavs;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--dump") == 0) {
            dump = true;
        } else {
            wavs.push_back(argv[i]);
        }
    }

    SherpaOnnxSpeakerEmbeddingExtractorConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.model = argv[1];
    cfg.num_threads = 1;
    cfg.debug = 0;
    cfg.provider = "cpu";
    const SherpaOnnxSpeakerEmbeddingExtractor* ex = SherpaOnnxCreateSpeakerEmbeddingExtractor(&cfg);
    if (!ex) {
        std::fprintf(stderr, "cannot create extractor for %s\n", cfg.model);
        return 2;
    }
    const int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(ex);
    std::printf("extractor ready, dim=%d\n", dim);

    for (const char* path : wavs) {
        std::vector<float> samples;
        int rate = 0;
        if (!read_wav(path, samples, rate)) {
            std::fprintf(stderr, "cannot read %s (need 16 kHz mono 16-bit PCM)\n", path);
            return 3;
        }
        double total_ms = 0;
        std::vector<float> emb;
        for (int r = 0; r < repeat; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            const SherpaOnnxOnlineStream* s = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(ex);
            SherpaOnnxOnlineStreamAcceptWaveform(s, rate, samples.data(), (int32_t) samples.size());
            SherpaOnnxOnlineStreamInputFinished(s);
            const float* v = nullptr;
            if (SherpaOnnxSpeakerEmbeddingExtractorIsReady(ex, s)) {
                v = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(ex, s);
            }
            total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (!v) {
                std::fprintf(stderr, "stream not ready / no embedding for %s\n", path);
                SherpaOnnxDestroyOnlineStream(s);
                return 4;
            }
            emb.assign(v, v + dim);
            SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(v);
            SherpaOnnxDestroyOnlineStream(s);
        }
        double norm = 0;
        for (float x : emb) {
            norm += (double) x * x;
        }
        norm = std::sqrt(norm);
        std::printf("%s\t%.1f ms/iter\tnorm=%.4f\tfirst8=", path, total_ms / repeat, norm);
        for (int i = 0; i < 8 && i < dim; ++i) {
            std::printf("%s%.4f", i ? "," : "", emb[(size_t) i]);
        }
        std::printf("\n");
        if (dump) {
            std::printf("EMB\t%s", path);
            for (int i = 0; i < dim; ++i) {
                std::printf("\t%.6f", emb[(size_t) i]);
            }
            std::printf("\n");
        }
    }
    SherpaOnnxDestroySpeakerEmbeddingExtractor(ex);
    return 0;
}
