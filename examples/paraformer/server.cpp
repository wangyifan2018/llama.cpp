// Whole-utterance ASR server around the Paraformer engine.
//
// Wire protocol (x-asr compatible framing):
//   request : u32 op, u32 reserved, u32 bytes, payload (float32 mono 16 kHz)
//   response: u32 status, u32 bytes, payload
//   op:     0 RECOGNIZE (payload = one whole utterance)
//   status: 1 FINAL (UTF-8 transcript), 2 ERROR (message)
//
// Weight sharing + parallel inference:
//   The GGUF is read once into a canonical host-buffer weight set. Each engine
//   clones it into its own backend buffer (HTP tiled device buffer for engine 0,
//   CPU host buffer for engine 1). Engines are independent contexts with their
//   own scheduler/graph cache, so they run in parallel without a global lock.
//   Requests are dispatched to a shared queue; one worker per engine pulls a
//   request, transcribes on its context, and fulfils the caller's promise.
//
//   --engines 2 (default): HTP + CPU, two workers.
//   --engines 1: single context on --device (baseline, comparable to M3).
//
// A device has one HTP session (v73 forces ndev=1), so two HTP engines cannot
// run concurrently; the second engine is always CPU.

#include "paraformer.h"
#include "ggml-backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <algorithm>
#include <cerrno>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using pf_clock = std::chrono::steady_clock;

enum { OP_RECOGNIZE = 0 };
enum { ST_FINAL = 1, ST_ERROR = 2 };

struct Engine {
    paraformer_context* ctx = nullptr;
    int id = 0;
    const char* name = "";
    std::thread worker;
    std::atomic<uint64_t> requests{0};
    std::atomic<double> total_ms{0};
};

struct Task {
    std::vector<float> samples;
    std::promise<std::pair<int, std::string>> result; // status, text or error
    int fd = -1;
    double enqueued_ms = 0;
};

static std::queue<std::unique_ptr<Task>> g_queue;
static std::mutex g_queue_mutex;
static std::condition_variable g_queue_cv;
static bool g_stop = false;
static size_t queue_limit = 32;      // pending requests before OPEN-side reject
static double queue_timeout_ms = 15000; // a request that waited longer is dropped
static double g_queue_ms = 0;        // last request's queue wait, for the log
static double g_infer_ms = 0;

// A client that hung up before its turn must not consume an engine slot.
static bool peer_alive(int fd) {
    if (fd < 0)
        return false;
    char c = 0;
    const ssize_t n = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0)
        return false;
    if (n < 0)
        return errno == EAGAIN || errno == EWOULDBLOCK;
    return true;
}
static std::vector<Engine*> g_engines;
static std::atomic<uint64_t> g_requests{0};
static std::atomic<double> g_total_ms{0};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(pf_clock::now().time_since_epoch()).count();
}

static bool read_exact(int fd, void* data, size_t bytes) {
    auto* p = static_cast<uint8_t*>(data);
    while (bytes > 0) {
        const ssize_t n = recv(fd, p, bytes, 0);
        if (n <= 0)
            return false;
        p += n;
        bytes -= (size_t) n;
    }
    return true;
}

static bool write_exact(int fd, const void* data, size_t bytes) {
    auto* p = static_cast<const uint8_t*>(data);
    while (bytes > 0) {
        const ssize_t n = send(fd, p, bytes, MSG_NOSIGNAL);
        if (n <= 0)
            return false;
        p += n;
        bytes -= (size_t) n;
    }
    return true;
}

static bool respond(int fd, uint32_t status, const std::string& payload) {
    uint32_t head[2] = {status, (uint32_t) payload.size()};
    return write_exact(fd, head, sizeof(head)) &&
           (payload.empty() || write_exact(fd, payload.data(), payload.size()));
}

static bool read_wav_bytes(const std::string& path, std::vector<float>& samples) {
    FILE* f = std::fopen(path.c_str(), "rb");
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

static void worker_loop(Engine* eng) {
    for (;;) {
        std::unique_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            g_queue_cv.wait(lock, [] { return !g_queue.empty() || g_stop; });
            if (g_queue.empty()) {
                if (g_stop)
                    return;
                continue;
            }
            task = std::move(g_queue.front());
            g_queue.pop();
        }
        const double queue_ms = now_ms() - task->enqueued_ms;
        std::pair<int, std::string> reply;
        if (queue_timeout_ms > 0 && queue_ms > queue_timeout_ms) {
            reply = {ST_ERROR, "queue timeout"};
            if (getenv("PF_TIMING"))
                std::fprintf(stderr, "pf-server: dropped request after %.1f ms in queue\n", queue_ms);
        } else if (!peer_alive(task->fd)) {
            reply = {ST_ERROR, "client closed"};
            if (getenv("PF_TIMING"))
                std::fprintf(stderr, "pf-server: dropped request after %.1f ms, client gone\n", queue_ms);
        } else {
            const double t0 = now_ms();
            std::unique_ptr<char, decltype(&std::free)> out(
                paraformer_transcribe(eng->ctx, task->samples.data(), (int) task->samples.size()), &std::free);
            const double dt = now_ms() - t0;
            eng->requests.fetch_add(1);
            double cur = eng->total_ms.load();
            while (!eng->total_ms.compare_exchange_weak(cur, cur + dt)) {}
            g_requests.fetch_add(1);
            cur = g_total_ms.load();
            while (!g_total_ms.compare_exchange_weak(cur, cur + dt)) {}
            if (getenv("PF_TIMING")) {
                static std::mutex log_mutex;
                std::lock_guard<std::mutex> lock(log_mutex);
                std::fprintf(stderr, "pf-server: queue_ms=%.3f infer_ms=%.3f engine=%s\n", queue_ms, dt, eng->name);
            }
            g_queue_ms = queue_ms;
            g_infer_ms = dt;
            reply = out ? std::make_pair((int) ST_FINAL, std::string(out.get())) : std::make_pair((int) ST_ERROR, std::string("inference failed"));
        }
        task->result.set_value(std::move(reply));
    }
}

static void handle_connection(int fd) {
    for (;;) {
        uint32_t head[3];
        if (!read_exact(fd, head, sizeof(head)))
            break;
        uint32_t op = head[0], bytes = head[2];
        if (bytes > 64u * 1024u * 1024u) {
            respond(fd, ST_ERROR, "payload too large");
            break;
        }
        if (op != OP_RECOGNIZE) {
            if (bytes) {
                std::vector<uint8_t> sink(bytes);
                if (!read_exact(fd, sink.data(), bytes))
                    break;
            }
            respond(fd, ST_ERROR, "unknown op");
            continue;
        }
        std::vector<float> samples(bytes / sizeof(float));
        if (bytes && !read_exact(fd, samples.data(), bytes))
            break;
        if (samples.empty()) {
            respond(fd, ST_ERROR, "empty audio");
            continue;
        }

        auto task = std::unique_ptr<Task>(new Task());
        task->samples = std::move(samples);
        task->fd = fd;
        task->enqueued_ms = now_ms();
        auto result = task->result.get_future();
        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            if (g_queue.size() >= queue_limit) {
                respond(fd, ST_ERROR, "busy: queue full");
                continue;
            }
            g_queue.push(std::move(task));
        }
        g_queue_cv.notify_one();

        const std::pair<int, std::string> reply = result.get();
        if (!respond(fd, (uint32_t) reply.first, reply.second))
            break;
    }
    close(fd);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf [DEVICE] [--port N] [--warmup F32] [--threads N] "
                     "[--engines 1|2] [--bind ADDR] [--queue N] [--timeout-ms N] [--no-warmup]\n",
                     argv[0]);
        return 1;
    }
    const char* model = argv[1];
    const char* device = "CPU";
    int port = 8766;
    const char* bind_addr = "0.0.0.0";
    const char* warmup = nullptr;
    int threads = 2;
    int n_engines = 2;
    bool no_warmup = false;
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
        } else if (std::strcmp(argv[i], "--engines") == 0 && i + 1 < argc) {
            n_engines = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--queue") == 0 && i + 1 < argc) {
            queue_limit = (size_t) std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc) {
            queue_timeout_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-warmup") == 0) {
            no_warmup = true;
        } else if (argv[i][0] != '-' && !device_set) {
            device = argv[i];
            device_set = true;
        }
    }
    if (n_engines < 1)
        n_engines = 1;
    if (n_engines > 4)
        n_engines = 4; // one device engine plus up to three CPU engines

    // The HTP backend is selected via a device env read by crispasr_init_gpu_backend.
    if (device_set && std::strcmp(device, "CPU") != 0) {
        setenv("PARAFORMER_DEVICE", device, 1);
    }

    ggml_backend_load_all();

    std::vector<float> warm;
    if (warmup && !read_wav_bytes(warmup, warm)) {
        std::fprintf(stderr, "pf-server: cannot read warmup f32 %s\n", warmup);
        return 3;
    }
    if (warm.empty() && !no_warmup) {
        // The first request otherwise pays the DSP-side graph registration
        // (about 1 s). Shape does not matter for that cost, so synthesize a
        // representative 1.2 s of silence and keep warmup inside startup.
        warm.assign((size_t) 16000 * 12 / 10, 0.0f);
        std::fprintf(stderr, "pf-server: no --warmup file, warming with 1.2 s of silence\n");
    }

    // Load the GGUF once into a canonical host-buffer weight set.
    const double load0 = now_ms();
    paraformer_weights* weights = paraformer_weights_load(model, 0);
    if (!weights) {
        std::fprintf(stderr, "pf-server: cannot load weights from %s\n", model);
        return 2;
    }
    std::fprintf(stderr, "pf-server: shared weights load %.1f ms\n", now_ms() - load0);

    paraformer_context_params params = paraformer_context_default_params();
    params.use_gpu = false;
    params.flash_attn = false;
    params.n_threads = threads;
    params.verbosity = 0;

    // Engine 0: requested device (HTP when a device is given).
    // Engine 1: CPU, from the same canonical weights.
    struct EngineSpec {
        bool use_gpu;
        const char* name;
    };
    std::vector<EngineSpec> specs;
    specs.push_back({device_set && std::strcmp(device, "CPU") != 0, device});
    for (int i = 1; i < n_engines; ++i)
        specs.push_back({false, "CPU"});

    for (size_t i = 0; i < specs.size(); ++i) {
        auto* eng = new Engine();
        eng->id = (int) i;
        eng->name = specs[i].name;
        auto p = params;
        p.use_gpu = specs[i].use_gpu;
        const double t0 = now_ms();
        eng->ctx = paraformer_init_shared(weights, p);
        if (!eng->ctx) {
            std::fprintf(stderr, "pf-server: engine %zu (%s) init failed\n", i, specs[i].name);
            return 2;
        }
        if (!warm.empty())
            paraformer_warmup(eng->ctx, warm.data(), (int) warm.size());
        std::fprintf(stderr, "pf-server: engine %d (%s) ready in %.1f ms\n", eng->id, eng->name, now_ms() - t0);
        g_engines.push_back(eng);
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
    if (bind(listener, (sockaddr*) &addr, sizeof(addr)) < 0) {
        std::perror("bind");
        return 5;
    }
    listen(listener, 64);
    std::fprintf(stderr, "pf-server: listening on %s:%d (%zu engines)\n", bind_addr, port, g_engines.size());

    for (Engine* eng : g_engines)
        eng->worker = std::thread(worker_loop, eng);

    for (;;) {
        int fd = accept(listener, nullptr, nullptr);
        if (fd < 0)
            continue;
        int nodelay = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        std::thread(handle_connection, fd).detach();
    }
}
