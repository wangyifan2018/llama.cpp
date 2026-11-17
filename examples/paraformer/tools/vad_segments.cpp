// Silero VAD through sherpa-onnx: print the speech segments and the per-frame cost.
//
// The cabin client already cuts at its own VAD, so the server-side VAD is for the
// cases the client cannot cover: untidy or over-long commits, and a continuous
// stream that the server has to segment itself.
//
//   vad_segments MODEL.onnx WAV... [--max-speech S] [--min-silence S]

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
    if (!data || channels != 1 || bits != 16) {
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
        std::fprintf(stderr, "usage: %s MODEL.onnx WAV... [--max-speech S] [--min-silence S]\n", argv[0]);
        return 1;
    }
    float max_speech = 10.0f, min_silence = 0.30f;
    std::vector<const char*> wavs;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--max-speech") == 0 && i + 1 < argc) {
            max_speech = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--min-silence") == 0 && i + 1 < argc) {
            min_silence = (float) std::atof(argv[++i]);
        } else {
            wavs.push_back(argv[i]);
        }
    }

    SherpaOnnxVadModelConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.silero_vad.model = argv[1];
    cfg.silero_vad.threshold = 0.5f;
    cfg.silero_vad.min_silence_duration = min_silence;
    cfg.silero_vad.min_speech_duration = 0.25f;
    cfg.silero_vad.window_size = 512;
    cfg.silero_vad.max_speech_duration = max_speech;
    cfg.sample_rate = 16000;
    cfg.num_threads = 1;
    cfg.provider = "cpu";
    // The detector keeps a cumulative sample offset across files, so recreate it
    // per file instead of clearing (clear does not reset that offset).
    const SherpaOnnxVoiceActivityDetector* vad = nullptr;
    for (const char* path : wavs) {
        if (vad) {
            SherpaOnnxDestroyVoiceActivityDetector(vad);
        }
        vad = SherpaOnnxCreateVoiceActivityDetector(&cfg, 60.0f);
        if (!vad) {
            std::fprintf(stderr, "cannot create VAD from %s\n", argv[1]);
            return 2;
        }
        std::vector<float> samples;
        int rate = 0;
        if (!read_wav(path, samples, rate) || rate != 16000) {
            std::fprintf(stderr, "cannot read %s (need 16 kHz mono 16-bit PCM)\n", path);
            return 3;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < samples.size(); i += 512) {
            const size_t n = std::min<size_t>(512, samples.size() - i);
            SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, samples.data() + i, (int32_t) n);
        }
        SherpaOnnxVoiceActivityDetectorFlush(vad);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double audio_s = (double) samples.size() / 16000.0;
        std::printf("%s\taudio %.2f s\tvad %.1f ms (%.2f%% of real time)\tsegments:", path, audio_s, ms,
                    100.0 * ms / (audio_s * 1000.0));
        while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
            const SherpaOnnxSpeechSegment* seg = SherpaOnnxVoiceActivityDetectorFront(vad);
            std::printf(" [%.2f-%.2f]", (double) seg->start / 16000.0,
                        ((double) seg->start + (double) seg->n) / 16000.0);
            SherpaOnnxDestroySpeechSegment(seg);
            SherpaOnnxVoiceActivityDetectorPop(vad);
        }
        std::printf("\n");
    }
    SherpaOnnxDestroyVoiceActivityDetector(vad);
    return 0;
}
