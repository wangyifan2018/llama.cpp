// sherpa-onnx service control for the X-ASR benchmark.
//
// Same wire protocol and the same worker structure as llama-x-asr-server: one
// worker, one request queue, COMMIT runs after that session's earlier APPENDs.
// APPEND also drains sherpa's ready chunks, so encoder work happens while audio
// arrives in both engines and the commit only covers the tail. This is a
// measurement control, not part of the deliverable.

#include "c-api.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using xasr_clock = std::chrono::steady_clock;

enum { OP_OPEN = 0, OP_APPEND = 1, OP_COMMIT = 2, OP_CANCEL = 3, OP_CLOSE = 4 };
enum { ST_OK = 0, ST_FINAL = 1, ST_ERROR = 2 };

struct request {
    int fd = -1;
    uint32_t op = 0;
    uint32_t session = 0;
    double enqueued_ms = 0;
    std::vector<float> audio;
};

static std::mutex queue_mutex;
static std::condition_variable queue_ready;
static std::deque<request> queue;
static size_t queue_limit = 256;
static std::atomic<bool> stopping{false};

static const SherpaOnnxOnlineRecognizer * recognizer = nullptr;
static std::map<uint32_t, const SherpaOnnxOnlineStream *> sessions;
static std::mutex sessions_mutex;
static uint32_t next_session = 1;
static int max_sessions = 1;
static int tail_pad_ms = 1200;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(xasr_clock::now().time_since_epoch()).count();
}

static bool read_exact(int fd, void * data, size_t bytes) {
    auto * p = static_cast<uint8_t *>(data);
    while (bytes > 0) {
        const ssize_t n = recv(fd, p, bytes, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        bytes -= (size_t) n;
    }
    return true;
}

static bool write_exact(int fd, const void * data, size_t bytes) {
    auto * p = static_cast<const uint8_t *>(data);
    while (bytes > 0) {
        const ssize_t n = send(fd, p, bytes, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        p += n;
        bytes -= (size_t) n;
    }
    return true;
}

static bool respond(int fd, uint32_t status, const std::string & payload) {
    if (fd < 0) {
        return true;
    }
    uint32_t head[2] = {status, (uint32_t) payload.size()};
    return write_exact(fd, head, sizeof(head)) &&
           (payload.empty() || write_exact(fd, payload.data(), payload.size()));
}

// Decode every chunk sherpa has ready for this stream.
static std::string drain(const SherpaOnnxOnlineStream * stream) {
    std::string text;
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
        SherpaOnnxDecodeOnlineStream(recognizer, stream);
        const SherpaOnnxOnlineRecognizerResult * r = SherpaOnnxGetOnlineStreamResult(recognizer, stream);
        if (r) {
            text = r->text;
            SherpaOnnxDestroyOnlineRecognizerResult(r);
        }
    }
    return text;
}

static void reader_thread(int fd) {
    for (;;) {
        uint32_t head[3];
        if (!read_exact(fd, head, sizeof(head))) {
            break;
        }
        request req;
        req.fd = fd;
        req.op = head[0];
        req.session = head[1];
        req.enqueued_ms = now_ms();
        const uint32_t bytes = head[2];
        if (req.op == OP_APPEND) {
            if (bytes % sizeof(float) != 0 || bytes > 16000u * 4u * 60u) {
                respond(fd, ST_ERROR, "append: bad payload length");
                break;
            }
            req.audio.resize(bytes / sizeof(float));
            if (bytes && !read_exact(fd, req.audio.data(), bytes)) {
                break;
            }
        } else if (bytes > 0) {
            std::vector<uint8_t> discard(bytes);
            if (!read_exact(fd, discard.data(), bytes)) {
                break;
            }
        }
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_ready.wait(lock, [] { return queue.size() < queue_limit || stopping; });
            if (stopping) {
                break;
            }
            queue.push_back(std::move(req));
        }
        queue_ready.notify_all();
    }
    close(fd);
}

static void handle(request & req) {
    switch (req.op) {
        case OP_OPEN: {
            {
                std::lock_guard<std::mutex> lock(sessions_mutex);
                if ((int) sessions.size() >= max_sessions) {
                    respond(req.fd, ST_ERROR, "open: all streams busy");
                    return;
                }
            }
            const SherpaOnnxOnlineStream * stream = SherpaOnnxCreateOnlineStream(recognizer);
            if (!stream) {
                respond(req.fd, ST_ERROR, "open: stream init failed");
                return;
            }
            uint32_t id = 0;
            {
                std::lock_guard<std::mutex> lock(sessions_mutex);
                id = next_session++;
                sessions[id] = stream;
            }
            respond(req.fd, ST_OK, std::string(reinterpret_cast<const char *>(&id), sizeof(id)));
            return;
        }
        case OP_APPEND: {
            const SherpaOnnxOnlineStream * stream = nullptr;
            {
                std::lock_guard<std::mutex> lock(sessions_mutex);
                auto it = sessions.find(req.session);
                if (it != sessions.end()) {
                    stream = it->second;
                }
            }
            if (!stream) {
                fprintf(stderr, "onnx-server: append for unknown session\n");
                close(req.fd);
                return;
            }
            if (!req.audio.empty()) {
                SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, req.audio.data(), (int) req.audio.size());
            }
            drain(stream);
            return;
        }
        case OP_COMMIT: {
            const SherpaOnnxOnlineStream * stream = nullptr;
            {
                std::lock_guard<std::mutex> lock(sessions_mutex);
                auto it = sessions.find(req.session);
                if (it != sessions.end()) {
                    stream = it->second;
                }
            }
            if (!stream) {
                respond(req.fd, ST_ERROR, "commit: unknown session");
                return;
            }
            const double queue_ms = now_ms() - req.enqueued_ms;
            const auto begin = xasr_clock::now();
            std::vector<float> pad((size_t) 16000 * tail_pad_ms / 1000, 0.0f);
            SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, pad.data(), (int) pad.size());
            SherpaOnnxOnlineStreamInputFinished(stream);
            const std::string text = drain(stream);
            if (getenv("ONNX_TIMING")) {
                fprintf(stderr, "onnx-server: commit session %u queue_ms=%.3f commit_ms=%.3f\n", req.session, queue_ms,
                        std::chrono::duration<double, std::milli>(xasr_clock::now() - begin).count());
            }
            respond(req.fd, ST_FINAL, text);
            return;
        }
        case OP_CANCEL:
        case OP_CLOSE: {
            const SherpaOnnxOnlineStream * stream = nullptr;
            {
                std::lock_guard<std::mutex> lock(sessions_mutex);
                auto it = sessions.find(req.session);
                if (it != sessions.end()) {
                    stream = it->second;
                    sessions.erase(it);
                }
            }
            if (stream) {
                SherpaOnnxDestroyOnlineStream(stream);
            }
            respond(req.fd, ST_OK, std::string());
            return;
        }
        default:
            respond(req.fd, ST_ERROR, "unknown op");
            return;
    }
}

static void worker_loop() {
    std::vector<request> batch;
    for (;;) {
        batch.clear();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_ready.wait(lock, [] { return !queue.empty() || stopping; });
            if (queue.empty() && stopping) {
                return;
            }
            size_t commit_at = queue.size();
            for (size_t i = 0; i < queue.size(); ++i) {
                if (queue[i].op == OP_COMMIT) {
                    commit_at = i;
                    break;
                }
            }
            if (commit_at == queue.size()) {
                batch.push_back(std::move(queue.front()));
                queue.pop_front();
            } else {
                const uint32_t sid = queue[commit_at].session;
                for (size_t i = 0; i < commit_at;) {
                    if (queue[i].op == OP_APPEND && queue[i].session == sid) {
                        batch.push_back(std::move(queue[i]));
                        queue.erase(queue.begin() + (long) i);
                        --commit_at;
                    } else {
                        ++i;
                    }
                }
                batch.push_back(std::move(queue[commit_at]));
                queue.erase(queue.begin() + (long) commit_at);
            }
        }
        queue_ready.notify_all();
        for (auto & req : batch) {
            handle(req);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL_DIR THREADS [--port N] [--streams N] [--bind ADDR]\n", argv[0]);
        return 1;
    }
    const std::string root = argv[1];
    const int threads = std::atoi(argv[2]);
    int port = 8766;
    const char * bind_addr = "0.0.0.0";
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
            max_sessions = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--tail-ms") == 0 && i + 1 < argc) {
            tail_pad_ms = std::atoi(argv[++i]);
        }
    }
    const std::string tokens = root + "/tokens.txt";
    const std::string encoder = root + "/encoder.int8.onnx";
    const std::string decoder = root + "/decoder.onnx";
    const std::string joiner = root + "/joiner.int8.onnx";
    SherpaOnnxOnlineRecognizerConfig config;
    memset(&config, 0, sizeof(config));
    config.feat_config = {16000, 80};
    config.model_config.tokens = tokens.c_str();
    config.model_config.num_threads = threads;
    config.model_config.provider = "cpu";
    config.model_config.transducer = {encoder.c_str(), decoder.c_str(), joiner.c_str()};
    config.model_config.model_type = "zipformer2";
    config.decoding_method = "greedy_search";
    config.enable_endpoint = 0;
    recognizer = SherpaOnnxCreateOnlineRecognizer(&config);
    if (!recognizer) {
        fprintf(stderr, "onnx-server: cannot create recognizer\n");
        return 2;
    }
    std::thread worker(worker_loop);
    (void) worker;

    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        fprintf(stderr, "onnx-server: socket failed\n");
        return 3;
    }
    int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, bind_addr, &addr.sin_addr) != 1) {
        fprintf(stderr, "onnx-server: bad bind address\n");
        return 4;
    }
    if (bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || listen(listener, 8) != 0) {
        fprintf(stderr, "onnx-server: cannot listen on port %d\n", port);
        _exit(5); // the worker thread is already running
    }
    fprintf(stderr, "onnx-server: %d threads, %d streams, tail %d ms, listening on %s:%d\n", threads, max_sessions,
            tail_pad_ms, bind_addr, port);
    for (;;) {
        const int fd = accept(listener, nullptr, nullptr);
        if (fd < 0) {
            continue;
        }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        timeval tv{};
        tv.tv_sec = 60;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        std::thread(reader_thread, fd).detach();
    }
    return 0;
}
