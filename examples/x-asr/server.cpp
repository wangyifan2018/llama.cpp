// Minimal streaming ASR server around the X-ASR engine.
//
// tools/server is scoped to model-agnostic LLM features, and X-ASR is a
// model-specific streaming engine, so this daemon lives with the example. The
// wire protocol is length-prefixed little-endian frames:
//
//   request : u32 op, u32 session, u32 bytes, payload
//   response: u32 status, u32 bytes, payload
//
// ops: 0 OPEN, 1 APPEND (payload float32 mono 16 kHz), 2 COMMIT, 3 CANCEL, 4 CLOSE
// status: 0 OK, 1 FINAL (payload text), 2 ERROR (payload message)
//
// APPEND has no reply: audio is one way, and backpressure comes from the request
// queue filling and the reader blocking on the socket. Every other op answers
// exactly once. An APPEND that cannot be accepted closes the connection, so the
// client can never resynchronise on a stale reply.
//
// One worker thread owns the engine and all sessions; connection threads only
// move bytes. The worker answers every request, so a client may pipeline
// APPENDs and read the answers later.

#include "xasr.h"

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
    double enqueued_ms = 0; // steady clock, for queue-wait accounting
    std::vector<float> audio;
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(xasr_clock::now().time_since_epoch()).count();
}

static std::mutex queue_mutex;
static std::condition_variable queue_ready;
static std::deque<request> queue;
static size_t queue_limit = 64;
static std::atomic<bool> stopping{false};

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
    uint32_t head[2] = {status, (uint32_t) payload.size()};
    return write_exact(fd, head, sizeof(head)) &&
           (payload.empty() || write_exact(fd, payload.data(), payload.size()));
}

// Reads one request frame and queues it. Audio is capped so a malformed length
// cannot allocate without bound.
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

struct worker {
    xasr_context * model = nullptr;
    std::map<uint32_t, xasr_stream *> sessions;
    uint32_t next_id = 1;
    int max_sessions = 1;

    xasr_stream * find(uint32_t id) const {
        auto it = sessions.find(id);
        return it == sessions.end() ? nullptr : it->second;
    }

    // COMMIT has priority: when one is queued, the audio that arrived before it
    // for the same session is drained and run first, then the commit. That keeps
    // per-session order while stopping another session's audio from delaying a
    // finished utterance.
    void run() {
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

    void handle(request & req) {
        switch (req.op) {
            case OP_OPEN: {
                if ((int) sessions.size() >= max_sessions) {
                    respond(req.fd, ST_ERROR, "open: all streams busy");
                    return;
                }
                xasr_stream * s = xasr_stream_init(model);
                if (!s) {
                    respond(req.fd, ST_ERROR, "open: stream init failed");
                    return;
                }
                const uint32_t id = next_id++;
                sessions[id] = s;
                respond(req.fd, ST_OK, std::string(reinterpret_cast<const char *>(&id), sizeof(id)));
                return;
            }
            case OP_APPEND: {
                xasr_stream * s = find(req.session);
                if (!s) {
                    fprintf(stderr, "xasr-server: append for unknown session, closing connection\n");
                    close(req.fd);
                    return;
                }
                if (!xasr_append(s, req.audio.data(), (int) req.audio.size())) {
                    fprintf(stderr, "xasr-server: append rejected, closing connection\n");
                    close(req.fd);
                    return;
                }
                return;
            }
            case OP_COMMIT: {
                xasr_stream * s = find(req.session);
                if (!s) {
                    respond(req.fd, ST_ERROR, "commit: unknown session");
                    return;
                }
                const auto begin = xasr_clock::now();
                const double queue_ms = now_ms() - req.enqueued_ms;
                std::string text;
                const bool ok = xasr_commit(s, text);
                const double ms = std::chrono::duration<double, std::milli>(xasr_clock::now() - begin).count();
                if (getenv("XASR_TIMING")) {
                    fprintf(stderr, "xasr-server: commit session %u queue_ms=%.3f commit_ms=%.3f\n", req.session, queue_ms, ms);
                }
                if (!ok) {
                    respond(req.fd, ST_ERROR, "commit: failed");
                    return;
                }
                respond(req.fd, ST_FINAL, text);
                return;
            }
            case OP_CANCEL: {
                auto it = sessions.find(req.session);
                if (it == sessions.end()) {
                    respond(req.fd, ST_ERROR, "cancel: unknown session");
                    return;
                }
                xasr_stream_free(it->second);
                sessions.erase(it);
                respond(req.fd, ST_OK, std::string());
                return;
            }
            case OP_CLOSE: {
                auto it = sessions.find(req.session);
                if (it != sessions.end()) {
                    xasr_stream_free(it->second);
                    sessions.erase(it);
                }
                respond(req.fd, ST_OK, std::string());
                return;
            }
            default:
                respond(req.fd, ST_ERROR, "unknown op");
                return;
        }
    }
};

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf DEVICE [--port N] [--streams N] [--queue N] [--bind ADDR]\n", argv[0]);
        return 1;
    }
    xasr_context_params params;
    params.device = argv[2];
    int port = 8765;
    const char * bind_addr = "0.0.0.0";
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
            params.n_streams = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--queue") == 0 && i + 1 < argc) {
            queue_limit = (size_t) std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            params.n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--joiner-host") == 0) {
            params.joiner_on_device = false;
        }
    }
    ggml_backend_load_all();
    xasr_context * model = xasr_init_from_file(argv[1], params);
    if (!model) {
        fprintf(stderr, "xasr-server: cannot load %s on %s\n", argv[1], params.device);
        return 2;
    }
    worker w;
    w.model = model;
    w.max_sessions = params.n_streams > 0 ? params.n_streams : 1;
    std::thread worker_thread([&w] { w.run(); });

    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        fprintf(stderr, "xasr-server: socket failed\n");
        return 3;
    }
    int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, bind_addr, &addr.sin_addr) != 1) {
        fprintf(stderr, "xasr-server: bad bind address %s\n", bind_addr);
        return 4;
    }
    if (bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || listen(listener, 8) != 0) {
        fprintf(stderr, "xasr-server: cannot listen on %s:%d\n", bind_addr, port);
        return 5;
    }
    fprintf(stderr, "xasr-server: listening on %s:%d, %d streams\n", bind_addr, port, w.max_sessions);
    for (;;) {
        const int fd = accept(listener, nullptr, nullptr);
        if (fd < 0) {
            if (stopping) {
                break;
            }
            continue;
        }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::thread(reader_thread, fd).detach();
    }
    return 0;
}
