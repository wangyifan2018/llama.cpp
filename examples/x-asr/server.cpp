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
// Each backend engine (one xasr_context) owns a worker thread and a request
// queue, so a device engine and a CPU engine run at the same time instead of
// serialising behind one worker. Sessions are pinned to an engine at OPEN.

#include "xasr.h"

#include "ggml-backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using xasr_clock = std::chrono::steady_clock;

enum { OP_OPEN = 0, OP_APPEND = 1, OP_COMMIT = 2, OP_CANCEL = 3, OP_CLOSE = 4 };
enum { ST_OK = 0, ST_FINAL = 1, ST_ERROR = 2 };

struct engine;

using socket_ref = std::shared_ptr<const int>;

struct request {
    int fd = -1;
    uint32_t op = 0;
    uint32_t session = 0;
    double enqueued_ms = 0;
    engine * eng = nullptr;
    std::vector<float> audio;
    socket_ref connection;
    bool started = false;
    xasr_stream * ready_stream = nullptr;
    double queue_ms = 0;
};

struct session_entry {
    engine * eng = nullptr;
    xasr_stream * stream = nullptr;
    std::weak_ptr<const int> connection;
    double last_ms = 0;
    bool expiring = false;
};

struct engine {
    xasr_context * ctx = nullptr;
    const char * device = "";
    int max_sessions = 0;
    int max_batch = 1;
    int threads = 1;
    std::atomic<double> oldest_active_commit_ms{0};
    std::atomic<bool> active_work{false};
    int live = 0;
    std::deque<request> queue;
    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable space;
    std::thread worker;
    bool stopping = false;

    size_t pending() {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
    bool has_room() {
        std::lock_guard<std::mutex> lock(mutex);
        return live < max_sessions;
    }
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(xasr_clock::now().time_since_epoch()).count();
}

static std::mutex sessions_mutex;
static std::map<uint32_t, session_entry> sessions;
static uint32_t next_session = 1;
static double idle_limit_ms = 30000;

// Reaps sessions that stopped sending. Expiry is queued to the owning engine so
// the stream is freed on the worker thread, never concurrently with a commit.
static void janitor_loop(std::vector<engine *> engines) {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        const double now = now_ms();
        std::vector<std::pair<engine *, uint32_t>> expired;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex);
            for (auto & [id, entry] : sessions) {
                if (!entry.expiring && now - entry.last_ms > idle_limit_ms) {
                    entry.expiring = true;
                    expired.emplace_back(entry.eng, id);
                }
            }
        }
        for (auto & [eng, id] : expired) {
            request req;
            req.fd = -1;
            req.op = OP_CLOSE;
            req.session = id;
            req.eng = eng;
            req.enqueued_ms = now_ms();
            fprintf(stderr, "xasr-server: session %u idle, releasing\n", id);
            std::lock_guard<std::mutex> lock(eng->mutex);
            eng->queue.push_back(std::move(req));
            eng->ready.notify_one();
        }
    }
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
        return true; // internal request, nobody to answer
    }
    uint32_t head[2] = {status, (uint32_t) payload.size()};
    return write_exact(fd, head, sizeof(head)) &&
           (payload.empty() || write_exact(fd, payload.data(), payload.size()));
}

static void close_session(uint32_t id) {
    std::lock_guard<std::mutex> lock(sessions_mutex);
    auto it = sessions.find(id);
    if (it == sessions.end()) {
        return;
    }
    xasr_stream_free(it->second.stream);
    {
        std::lock_guard<std::mutex> elock(it->second.eng->mutex);
        it->second.eng->live -= 1;
    }
    sessions.erase(it);
}

// Bounded queue: a reader that outruns its engine blocks here, which is the
// backpressure the client sees as a stalled socket.
static size_t queue_limit = 256;
static bool chunk_scheduler = false;
static bool adaptive_cpu = false;
static std::atomic<int> idle_cpus{-1};

#ifdef __linux__
struct cpu_ticks {
    uint64_t total;
    uint64_t idle;
};

static std::map<int, cpu_ticks> read_cpu_ticks() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return {};
    }
    FILE * file = fopen("/proc/stat", "r");
    if (!file) {
        return {};
    }
    std::map<int, cpu_ticks> ticks;
    char line[256];
    while (fgets(line, sizeof(line), file) && std::strncmp(line, "cpu", 3) == 0) {
        int core;
        unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
        if (sscanf(line, "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu",
                   &core, &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) != 9 ||
            core < 0 || core >= CPU_SETSIZE || !CPU_ISSET(core, &allowed)) {
            continue;
        }
        ticks.emplace(core, cpu_ticks{user + nice + system + idle + iowait + irq + softirq + steal, idle});
    }
    fclose(file);
    return ticks;
}

[[noreturn]] static void sample_cpu_headroom() {
    auto previous = read_cpu_ticks();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto current = read_cpu_ticks();
        int idle = 0;
        int sampled = 0;
        for (const auto & [core, ticks] : current) {
            const auto old = previous.find(core);
            if (old == previous.end() || ticks.total <= old->second.total || ticks.idle < old->second.idle) {
                continue;
            }
            ++sampled;
            if (double(ticks.idle - old->second.idle) / double(ticks.total - old->second.total) >= 0.8) {
                ++idle;
            }
        }
        idle_cpus.store(sampled ? idle : -1);
        previous = std::move(current);
    }
}
#endif

static double oldest_commit(double oldest, const request & req) {
    if (req.op == OP_COMMIT && (oldest == 0 || req.enqueued_ms < oldest)) {
        return req.enqueued_ms;
    }
    return oldest;
}

// Estimated completion order: sessions already running, plus a penalty when the
// engine's oldest commit is already late. The engine with the smallest score is
// expected to finish a new session first. Ties go to the earlier engine, which
// is the device engine, so model compute stays on HTP when both are free.
static bool eta_router = false;

static engine * reserve_eta(const std::vector<engine *> & engines) {
    engine * best = nullptr;
    double best_score = 0;
    for (auto * eng : engines) {
        std::lock_guard<std::mutex> lock(eng->mutex);
        if (eng->live >= eng->max_sessions) {
            continue;
        }
        const double age = eng->oldest_active_commit_ms.load();
        const double score = (double) eng->live + age / 300.0;
        if (!best || score < best_score - 1e-9) {
            best = eng;
            best_score = score;
        }
    }
    if (best) {
        ++best->live; // reserve the slot; the worker creates the stream
    }
    if (getenv("XASR_TIMING")) {
        fprintf(stderr, "xasr-route mode=eta device=%s score=%.2f live=%d age_ms=%.1f\n",
                best ? best->device : "busy", best_score, best ? best->live : 0,
                best ? best->oldest_active_commit_ms.load() : 0.0);
    }
    return best;
}

static engine * reserve_adaptive(const std::vector<engine *> & engines) {
    auto * htp = engines[0];
    auto * cpu = engines[1];
    std::scoped_lock locked(htp->mutex, cpu->mutex);
    double oldest = htp->oldest_active_commit_ms.load();
    for (const auto & req : htp->queue) {
        oldest = oldest_commit(oldest, req);
    }
    const double age_ms = oldest == 0 ? 0 : now_ms() - oldest;
    const int idle = idle_cpus.load();
    const bool htp_room = htp->live < htp->max_sessions;
    const bool cpu_room = cpu->live < cpu->max_sessions;
    const bool cpu_draining = cpu->active_work.load() || !cpu->queue.empty();
    auto * chosen = htp_room ? htp : nullptr;
    if (cpu_room && !cpu_draining && idle >= cpu->threads && (!htp_room || age_ms >= 80.0)) {
        chosen = cpu;
    }
    if (chosen) {
        ++chosen->live;
    }
    if (getenv("XASR_TIMING")) {
        fprintf(stderr, "xasr-route device=%s idle_cpus=%d htp_commit_age_ms=%.1f htp_room=%d\n",
                chosen ? chosen->device : "busy", idle, age_ms, htp_room);
    }
    return chosen;
}


static void push(engine * eng, request && req) {
    {
        std::unique_lock<std::mutex> lock(eng->mutex);
        eng->space.wait(lock, [eng] { return eng->queue.size() < queue_limit || eng->stopping; });
        eng->queue.push_back(std::move(req));
    }
    eng->ready.notify_one();
}

// One worker per engine: drain that engine's queue, preferring a queued COMMIT
// together with the session's earlier APPENDs so audio from another session
// cannot delay a finished utterance.
static void worker_loop(engine * eng) {
    std::vector<request> batch;
    std::deque<request> pending;
    bool resume_next = false;
    int admissions = 0;
    for (;;) {
        batch.clear();
        bool dispatch = false;
        {
            std::unique_lock<std::mutex> lock(eng->mutex);
            if (pending.empty()) {
                eng->ready.wait(lock, [eng] { return !eng->queue.empty() || eng->stopping; });
                if (eng->queue.empty() && eng->stopping) {
                    return;
                }
            }
            if (chunk_scheduler) {
                auto eligible = [&](const request & item) {
                    return std::none_of(pending.begin(), pending.end(), [&](const request & active) {
                        return (item.session && item.session == active.session) ||
                               (item.connection && item.connection == active.connection);
                    });
                };
                auto next = eng->queue.end();
                const bool admit = eng->max_batch == 1 ? (!resume_next || pending.empty()) :
                    (pending.empty() || (pending.size() < size_t(eng->max_batch) && admissions < eng->max_batch));
                if (admit) {
                    // A commit may pass unrelated clients, but not its own earlier frames.
                    const auto commit = std::find_if(eng->queue.begin(), eng->queue.end(), [&](const request & item) {
                        return item.op == OP_COMMIT && eligible(item);
                    });
                    if (commit != eng->queue.end()) {
                        next = std::find_if(eng->queue.begin(), std::next(commit), [&](const request & item) {
                            return item.connection == commit->connection && eligible(item);
                        });
                    } else {
                        next = std::find_if(eng->queue.begin(), eng->queue.end(), eligible);
                    }
                }
                if (next != eng->queue.end()) {
                    batch.push_back(std::move(*next));
                    eng->queue.erase(next);
                    resume_next = !pending.empty();
                    if (eng->max_batch > 1 && !pending.empty()) {
                        ++admissions;
                    }
                } else {
                    const size_t count = pending.size() >= 4 && eng->max_batch == 4 ? 4 : pending.size() >= 2 && eng->max_batch > 1 ? 2 : 1;
                    for (size_t i = 0; i < count; ++i) {
                        batch.push_back(std::move(pending.front()));
                        pending.pop_front();
                    }
                    dispatch = eng->max_batch > 1;
                    resume_next = false;
                    admissions = 0;
                }
            } else {
                size_t commit_at = eng->queue.size();
                for (size_t i = 0; i < eng->queue.size(); ++i) {
                    if (eng->queue[i].op == OP_COMMIT) {
                        commit_at = i;
                        break;
                    }
                }
                if (commit_at == eng->queue.size()) {
                    batch.push_back(std::move(eng->queue.front()));
                    eng->queue.pop_front();
                } else {
                    const uint32_t sid = eng->queue[commit_at].session;
                    for (size_t i = 0; i < commit_at;) {
                        if (eng->queue[i].op == OP_APPEND && eng->queue[i].session == sid) {
                            batch.push_back(std::move(eng->queue[i]));
                            eng->queue.erase(eng->queue.begin() + (long) i);
                            --commit_at;
                        } else {
                            ++i;
                        }
                    }
                    batch.push_back(std::move(eng->queue[commit_at]));
                    eng->queue.erase(eng->queue.begin() + (long) commit_at);
                }
            }
            if (adaptive_cpu) {
                double oldest = 0;
                for (const auto & req : pending) {
                    oldest = oldest_commit(oldest, req);
                }
                for (const auto & req : batch) {
                    oldest = oldest_commit(oldest, req);
                }
                eng->oldest_active_commit_ms.store(oldest);
                eng->active_work.store(!pending.empty() || !batch.empty());
            }
        }
        eng->space.notify_all();
        std::array<std::string, 4> batch_text;
        std::array<xasr_step_result, 4> batch_result;
        if (dispatch && batch.size() > 1) {
            std::array<xasr_stream *, 4> streams;
            for (size_t lane = 0; lane < batch.size(); ++lane) {
                streams[lane] = batch[lane].ready_stream;
            }
            xasr_step_batch(streams.data(), static_cast<int>(batch.size()), batch_result.data(), batch_text.data());
        }
        for (size_t lane = 0; lane < batch.size(); ++lane) {
            auto & req = batch[lane];
            xasr_stream * s = nullptr;
            {
                std::lock_guard<std::mutex> lock(sessions_mutex);
                auto it = sessions.find(req.session);
                if (it != sessions.end() && (!req.connection || it->second.connection.lock() == req.connection)) {
                    s = it->second.stream;
                }
            }
            const bool known = req.op == OP_OPEN || s != nullptr;
            switch (req.op) {
                case OP_OPEN: {
                    // Stream creation touches the engine, so it runs here and
                    // never on a connection thread.
                    xasr_stream * created = xasr_stream_init(eng->ctx);
                    if (!created) {
                        std::lock_guard<std::mutex> lock(eng->mutex);
                        eng->live -= 1;
                        respond(req.fd, ST_ERROR, "open: stream init failed");
                        break;
                    }
                    uint32_t id = 0;
                    {
                        std::lock_guard<std::mutex> lock(sessions_mutex);
                        id = next_session++;
                        sessions[id] = session_entry{eng, created, req.connection, now_ms(), false};
                    }
                    respond(req.fd, ST_OK, std::string(reinterpret_cast<const char *>(&id), sizeof(id)));
                    break;
                }
                case OP_APPEND:
                case OP_COMMIT: {
                    const bool commit = req.op == OP_COMMIT;
                    const auto begin = xasr_clock::now();
                    if (!req.started) {
                        req.queue_ms = now_ms() - req.enqueued_ms;
                    }
                    std::string text;
                    bool ok = known;
                    if (ok && chunk_scheduler) {
                        if (!req.started) {
                            ok = xasr_begin_step(s, commit ? nullptr : req.audio.data(), (int) req.audio.size(), commit);
                            req.started = true;
                            std::vector<float>().swap(req.audio);
                        }
                        if (ok && eng->max_batch > 1 && !dispatch && xasr_step_ready(s)) {
                            req.ready_stream = s;
                            if (pending.empty()) {
                                admissions = 0;
                            }
                            pending.push_back(std::move(req));
                            break;
                        }
                        auto result = xasr_step_result::error;
                        if (ok && dispatch && batch.size() > 1) {
                            result = batch_result[lane];
                            text = std::move(batch_text[lane]);
                        } else if (ok) {
                            result = xasr_step(s, text);
                        }
                        if (result == xasr_step_result::pending) {
                            pending.push_back(std::move(req));
                            break;
                        }
                        ok = result == xasr_step_result::done;
                    } else if (ok) {
                        ok = commit ? xasr_commit(s, text) : xasr_append(s, req.audio.data(), (int) req.audio.size());
                    }
                    if (!commit && !ok) {
                        fprintf(stderr, "xasr-server: append rejected, closing connection\n");
                        shutdown(req.fd, SHUT_RDWR);
                    } else if (commit) {
                        if (ok && getenv("XASR_TIMING")) {
                            const double commit_ms = chunk_scheduler ? xasr_stream_stats(s)->commit_wall_ms :
                                std::chrono::duration<double, std::milli>(xasr_clock::now() - begin).count();
                            const double elapsed_ms = now_ms() - req.enqueued_ms;
                            fprintf(stderr, "xasr-server: commit session %u on %s queue_ms=%.3f commit_ms=%.3f elapsed_ms=%.3f\n",
                                    req.session, eng->device, req.queue_ms, commit_ms, elapsed_ms);
                        }
                        respond(req.fd, ok ? ST_FINAL : ST_ERROR, ok ? text : std::string("commit: failed or unknown session"));
                    }
                    break;
                }
                case OP_CANCEL:
                case OP_CLOSE:
                    if (known) {
                        close_session(req.session);
                    }
                    respond(req.fd, ST_OK, std::string());
                    break;
                default:
                    respond(req.fd, ST_ERROR, "unknown op");
                    break;
            }
        }
        if (adaptive_cpu) {
            // Publish pending commits between iterations, including before a wait.
            double oldest = 0;
            for (const auto & req : pending) {
                oldest = oldest_commit(oldest, req);
            }
            eng->oldest_active_commit_ms.store(oldest);
            eng->active_work.store(!pending.empty());
        }
    }
}

// Reads one request frame, resolves the engine and queues it.
static void reader_thread(int fd, std::vector<engine *> engines) {
    const socket_ref connection(new int(fd), [](const int * socket) {
        close(*socket);
        delete socket;
    });
    for (;;) {
        uint32_t head[3];
        if (!read_exact(fd, head, sizeof(head))) {
            break;
        }
        request req;
        req.fd = fd;
        req.connection = connection;
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
            respond(fd, ST_ERROR, "control frame: unexpected payload");
            break;
        }
        if (req.op == OP_OPEN) {
            if (eta_router) {
                req.eng = reserve_eta(engines);
                if (!req.eng) {
                    respond(fd, ST_ERROR, "open: no available engine capacity");
                } else {
                    push(req.eng, std::move(req));
                }
                continue;
            }
            if (adaptive_cpu) {
                req.eng = reserve_adaptive(engines);
                if (!req.eng) {
                    respond(fd, ST_ERROR, "open: no available engine capacity");
                } else {
                    push(req.eng, std::move(req));
                }
                continue;
            }
            // Pin the session to the least loaded engine that still has a slot.
            engine * best = nullptr;
            size_t best_pending = 0;
            for (auto * eng : engines) {
                std::lock_guard<std::mutex> lock(eng->mutex);
                if (eng->live >= eng->max_sessions) {
                    continue;
                }
                if (!best || eng->queue.size() < best_pending) {
                    best = eng;
                    best_pending = eng->queue.size();
                }
            }
            if (!best) {
                respond(fd, ST_ERROR, "open: all streams busy");
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(best->mutex);
                if (best->live < best->max_sessions) {
                    ++best->live;
                    req.eng = best;
                }
            }
            if (!req.eng) {
                respond(fd, ST_ERROR, "open: all streams busy");
                continue;
            }
            push(best, std::move(req));
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(sessions_mutex);
            auto it = sessions.find(req.session);
            if (it == sessions.end() || it->second.connection.lock() != connection || it->second.expiring) {
                if (req.op == OP_APPEND) {
                    fprintf(stderr, "xasr-server: append for unknown session, closing connection\n");
                    shutdown(fd, SHUT_RDWR);
                    return;
                }
                respond(fd, ST_ERROR, "unknown session");
                continue;
            }
            req.eng = it->second.eng;
            it->second.last_ms = now_ms();
        }
        push(req.eng, std::move(req));
    }
    shutdown(fd, SHUT_RDWR);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf DEVICE [--hotwords FILE] [--port N] [--streams N] [--cpu-streams N] [--queue N] [--bind ADDR] [--chunk-scheduler] [--encoder-batch 1|2|4] [--adaptive-cpu] [--eta-router]\n", argv[0]);
        return 1;
    }
    xasr_context_params params;
    params.device = argv[2];
    int port = 8765;
    int cpu_streams = 0;
    const char * bind_addr = "0.0.0.0";
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
            params.n_streams = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cpu-streams") == 0 && i + 1 < argc) {
            cpu_streams = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--idle-ms") == 0 && i + 1 < argc) {
            idle_limit_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--queue") == 0 && i + 1 < argc) {
            queue_limit = (size_t) std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            params.n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--encoder-batch") == 0) {
            if (i + 1 >= argc || (std::strcmp(argv[i + 1], "1") && std::strcmp(argv[i + 1], "2") && std::strcmp(argv[i + 1], "4"))) {
                fprintf(stderr, "xasr-server: --encoder-batch expects 1, 2 or 4\n");
                return 1;
            }
            params.encoder_batch = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--adaptive-cpu") == 0) {
            adaptive_cpu = true;
        } else if (std::strcmp(argv[i], "--eta-router") == 0) {
            eta_router = true;
        } else if (std::strcmp(argv[i], "--chunk-scheduler") == 0) {
            chunk_scheduler = true;
        } else if (std::strcmp(argv[i], "--hotwords") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "xasr: --hotwords requires a file\n");
                return 1;
            }
            params.hotwords = argv[++i];
        } else if (std::strcmp(argv[i], "--joiner-host") == 0) {
            params.joiner_on_device = false;
        }
    }
    if (adaptive_cpu) {
#ifdef __linux__
        if (std::strncmp(params.device, "HTP", 3) != 0 || cpu_streams <= 0 || params.n_streams <= 0) {
            fprintf(stderr, "xasr-server: --adaptive-cpu requires HTP and positive --cpu-streams\n");
            return 1;
        }
#else
        fprintf(stderr, "xasr-server: --adaptive-cpu requires Linux CPU statistics\n");
        return 1;
#endif
    }
    chunk_scheduler = chunk_scheduler || params.encoder_batch > 1;
    ggml_backend_load_all();

    std::vector<std::unique_ptr<engine>> engines;
    auto start_engine = [&](const char * device, int streams) {
        if (streams <= 0) {
            return true;
        }
        xasr_context_params local = params;
        local.device = device;
        local.n_streams = streams;
        if (std::strcmp(device, "CPU") == 0 && std::strcmp(params.device, "CPU") != 0) {
            local.encoder_batch = 1;
        }
        xasr_context * ctx = xasr_init_from_file(argv[1], local);
        if (!ctx) {
            fprintf(stderr, "xasr-server: cannot load %s on %s\n", argv[1], device);
            return false;
        }
        auto eng = std::make_unique<engine>();
        eng->ctx = ctx;
        eng->device = device;
        eng->max_sessions = streams;
        eng->max_batch = local.encoder_batch;
        eng->threads = local.n_threads;
        engines.push_back(std::move(eng));
        return true;
    };
    if (!start_engine(params.device, params.n_streams)) {
        return 2;
    }
    if (cpu_streams > 0 && std::strcmp(params.device, "CPU") != 0) {
        if (!start_engine("CPU", cpu_streams)) {
            return 2;
        }
    }
    std::vector<engine *> raw;
    for (auto & eng : engines) {
        raw.push_back(eng.get());
        eng->worker = std::thread(worker_loop, eng.get());
    }

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
    for (auto * eng : raw) {
        fprintf(stderr, "xasr-server: engine %s with %d streams\n", eng->device, eng->max_sessions);
    }
    fprintf(stderr, "xasr-server: idle timeout %.0f ms, listening on %s:%d\n", idle_limit_ms, bind_addr, port);
    std::thread(janitor_loop, raw).detach();
#ifdef __linux__
    if (adaptive_cpu) {
        std::thread(sample_cpu_headroom).detach();
    }
#endif
    for (;;) {
        const int fd = accept(listener, nullptr, nullptr);
        if (fd < 0) {
            continue;
        }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        timeval tv{};
        tv.tv_sec = 60;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        tv.tv_sec = 1;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        std::thread(reader_thread, fd, raw).detach();
    }
    return 0;
}
