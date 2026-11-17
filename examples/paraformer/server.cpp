// Minimal whole-utterance ASR server around the Paraformer engine.
//
// Paraformer is non-autoregressive: one VAD segment in, one final transcript
// out. There is no append/commit streaming state, so the wire protocol is a
// single request/response over a length-prefixed little-endian frame:
//
//   request : u32 op, u32 reserved, u32 bytes, payload (float32 mono 16 kHz)
//   response: u32 status, u32 bytes, payload
//
//   op:     0 RECOGNIZE (payload = one whole utterance)
//   status: 1 FINAL (payload = UTF-8 transcript), 2 ERROR (payload = message)
//
// The x-asr client's framing (struct "<III" header + "<II" reply) is reused so
// the same socket helpers work. One context is shared by all connections and
// inference is serialized under a mutex: paraformer_context holds mutable
// scheduler and encoder-graph-cache state, so it is not safe to transcribe
// concurrently. Start with --warmup so the one-time HTP graph registration
// (~1 s on SM6650) is paid before the listener accepts traffic.

#include "paraformer.h"

#include "ggml-backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using pf_clock = std::chrono::steady_clock;

enum { OP_RECOGNIZE = 0 };
enum { ST_FINAL = 1, ST_ERROR = 2 };

static std::mutex g_infer_mutex;
static paraformer_context * g_ctx = nullptr;
static std::atomic<uint64_t> g_requests{0};
static std::atomic<double> g_total_ms{0};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(pf_clock::now().time_since_epoch()).count();
}

static bool read_exact(int fd, void * data, size_t bytes) {
    auto * p = static_cast<uint8_t *>(data);
    while (bytes > 0) {
        const ssize_t n = recv(fd, p, bytes, 0);
        if (n <= 0)
            return false;
        p += n;
        bytes -= (size_t) n;
    }
    return true;
}

static bool write_exact(int fd, const void * data, size_t bytes) {
    auto * p = static_cast<const uint8_t *>(data);
    while (bytes > 0) {
        const ssize_t n = send(fd, p, bytes, MSG_NOSIGNAL);
        if (n <= 0)
            return false;
        p += n;
        bytes -= (size_t) n;
    }
    return true;
}

static bool respond(int fd, uint32_t status, const std::string & payload) {
    uint32_t head[2] = {status, (uint32_t) payload.size()};
    return write_exact(fd, head, sizeof(head)) &&
           (payload.empty() || write_exact(fd, payload.data(), payload.size()));
}

static bool read_wav_bytes(const std::string & path, std::vector<float> & samples) {
    // Accept raw float32 files for warmup to avoid a miniaudio dependency here.
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f)
        return false;
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % 4 != 0) {
        std::fclose(f);
        return false;
    }
    samples.resize((size_t) bytes / 4);
    size_t got = std::fread(samples.data(), 1, (size_t) bytes, f);
    std::fclose(f);
    if (got != (size_t) bytes) {
        samples.clear();
        return false;
    }
    return true;
}

static void handle_connection(int fd) {
    for (;;) {
        uint32_t head[3];
        if (!read_exact(fd, head, sizeof(head)))
            break;
        uint32_t op = head[0], bytes = head[2];
        if (bytes > 64u * 1024u * 1024u) { // 64 MB guard (~33 min audio)
            respond(fd, ST_ERROR, "payload too large");
            break;
        }
        std::vector<float> samples;
        if (op != OP_RECOGNIZE) {
            // Drain any payload so the framing stays aligned, then reject.
            if (bytes) {
                std::vector<uint8_t> sink(bytes);
                if (!read_exact(fd, sink.data(), bytes))
                    break;
            }
            respond(fd, ST_ERROR, "unknown op");
            continue;
        }
        samples.resize(bytes / sizeof(float));
        if (bytes && !read_exact(fd, samples.data(), bytes)) {
            break;
        }
        if (samples.empty()) {
            respond(fd, ST_ERROR, "empty audio");
            continue;
        }

        std::string text;
        const double t0 = now_ms();
        {
            std::lock_guard<std::mutex> lock(g_infer_mutex);
            std::unique_ptr<char, decltype(&std::free)> out(
                paraformer_transcribe(g_ctx, samples.data(), (int) samples.size()), &std::free);
            if (out)
                text = out.get();
        }
        const double dt = now_ms() - t0;
        g_requests.fetch_add(1);
        double cur = g_total_ms.load();
        while (!g_total_ms.compare_exchange_weak(cur, cur + dt)) {}
        if (text.empty()) {
            respond(fd, ST_ERROR, "inference failed");
            continue;
        }
        if (!respond(fd, ST_FINAL, text))
            break;
    }
    close(fd);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [DEVICE] [--port N] [--warmup F32] [--threads N] [--bind ADDR]\n",
                     argv[0]);
        return 1;
    }
    const char * model = argv[1];
    const char * device = "CPU";
    int port = 8766;
    const char * bind_addr = "0.0.0.0";
    const char * warmup = nullptr;
    int threads = 2;
    bool device_set = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (argv[i][0] != '-' && !device_set) {
            device = argv[i];
            device_set = true;
        }
    }
    if (device_set && std::strcmp(device, "CPU") != 0) {
        setenv("PARAFORMER_DEVICE", device, 1);
        setenv("CRISPASR_PARAFORMER_GPU", "1", 1);
    }

    ggml_backend_load_all();
    paraformer_context_params params = paraformer_context_default_params();
    params.use_gpu = false;
    params.flash_attn = false;
    params.n_threads = threads;
    params.verbosity = 0;
    g_ctx = paraformer_init_from_file(model, params);
    if (!g_ctx) {
        std::fprintf(stderr, "pf-server: cannot load %s\n", model);
        return 2;
    }

    if (warmup) {
        std::vector<float> warm;
        if (!read_wav_bytes(warmup, warm)) {
            std::fprintf(stderr, "pf-server: cannot read warmup f32 %s\n", warmup);
            return 3;
        }
        const double t0 = now_ms();
        paraformer_warmup(g_ctx, warm.data(), (int) warm.size());
        std::fprintf(stderr, "pf-server: warmup %.1f ms\n", now_ms() - t0);
    }

    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        std::perror("socket");
        return 4;
    }
    int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    addr.sin_addr.s_addr = inet_addr(bind_addr);
    if (bind(listener, (sockaddr *) &addr, sizeof(addr)) < 0) {
        std::perror("bind");
        return 5;
    }
    listen(listener, 64);
    std::fprintf(stderr, "pf-server: listening on %s:%d (device=%s, threads=%d)\n", bind_addr, port, device, threads);

    for (;;) {
        int fd = accept(listener, nullptr, nullptr);
        if (fd < 0)
            continue;
        int nodelay = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        std::thread(handle_connection, fd).detach();
    }
}
