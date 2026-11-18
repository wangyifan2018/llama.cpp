// OpenAI-shaped HTTP and Realtime WebSocket front end for the Paraformer engine.
//
// REST: POST /v1/audio/transcriptions (multipart WAV), /asr/segment (raw f32),
//       /v1/models, /health, /metrics.
// WS:   /v1/realtime with the subset a VAD front end needs: session.update,
//       input_audio_buffer.append/commit/clear, transcription delta + completed.
//
// Commit is the VAD endpoint: the client sends a whole segment and gets the
// final text. Partials are opportunistic full re-decodes of the audio received
// so far, throttled and always outranked by a commit.
//
// See docs/backend/snapdragon/api-reference.md for the supported formats.

#include "paraformer.h"
#include "core/engine_start.h"
#include "core/segment_limits.h"

#include "httplib.h"
#include "json.hpp"

#ifdef XASR_SPEAKER
#include "c-api.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using pf_clock = std::chrono::steady_clock;

class HtpWatchdog {
public:
    explicit HtpWatchdog(std::chrono::milliseconds limit) : limit(limit), thread([this] { monitor(); }) {}

    ~HtpWatchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_one();
        thread.join();
    }

    struct Guard {
        Guard(HtpWatchdog* watchdog, const char* stage) : Guard(watchdog, stage, std::chrono::milliseconds(0)) {}
        Guard(HtpWatchdog* watchdog, const char* stage, std::chrono::milliseconds limit) : watchdog(watchdog) {
            if (watchdog) {
                const auto span = limit.count() > 0 ? limit : watchdog->limit;
                watchdog->set(pf_clock::now() + span, stage, span);
            }
        }
        ~Guard() {
            if (watchdog) watchdog->set(pf_clock::time_point::max(), nullptr, std::chrono::milliseconds(0));
        }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

    private:
        HtpWatchdog* watchdog;
    };

private:
    void set(pf_clock::time_point when, const char* stage, std::chrono::milliseconds span) {
        std::lock_guard<std::mutex> lock(mutex);
        deadline = when;
        phase = stage;
        active_limit = span;
        changed.notify_one();
    }

    void monitor() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!stopping) {
            changed.wait(lock, [this] { return stopping || phase; });
            if (stopping) break;
            const auto when = deadline;
            if (!changed.wait_until(lock, when, [this, when] { return stopping || deadline != when; })) {
                std::fprintf(stderr, "pf-api: HTP watchdog expired during %s after %lld ms\n", phase,
                             (long long) active_limit.count());
                // The backend may be waiting on FastRPC. Skip its teardown.
                std::_Exit(75);
            }
        }
    }

    const std::chrono::milliseconds limit;
    std::mutex mutex;
    std::condition_variable changed;
    pf_clock::time_point deadline = pf_clock::time_point::max();
    const char* phase = nullptr;
    std::chrono::milliseconds active_limit{0};
    bool stopping = false;
    std::thread thread;
};

struct Engine {
    paraformer_context_ptr ctx{nullptr, &paraformer_free};
    std::thread worker;
    std::string requested;
    std::string name;
    bool flash_attn = true;
    HtpWatchdog* watchdog = nullptr;
    paraformer_placement placement = {};
    std::atomic<uint64_t> requests{0};
#ifdef XASR_SPEAKER
    std::unique_ptr<const SherpaOnnxVoiceActivityDetector, decltype(&SherpaOnnxDestroyVoiceActivityDetector)> vad{
        nullptr, &SherpaOnnxDestroyVoiceActivityDetector};
#endif
};

struct TokensDeleter {
    void operator()(paraformer_tokens* t) const {
        paraformer_tokens_free(t);
    }
};
struct WordSpan {
    std::string text;
    double start_ms = 0;
    double end_ms = 0;
};

struct SpeakerTicket {
    std::atomic<bool> ready{false};
    std::optional<float> score;
};

struct SpeakerResult {
    const char* status = "disabled";
    const char* reason = nullptr;
    std::optional<float> score;
};

struct SegmentReply {
    int status = 200;
    std::string text;
    double queue_ms = 0;
    double infer_ms = 0;
    double vad_ms = 0;
    std::string backend;
    size_t pieces = 0;
    bool silence = false;
    std::vector<WordSpan> words;
    SpeakerResult speaker;
};

using pool_done = std::function<void(SegmentReply)>;
struct WsSession;

struct PoolTask {
    std::vector<float> samples;
    pool_done done;
    double enqueued_ms = 0;
    pf_clock::time_point deadline = pf_clock::time_point::max();
    int priority = 0;    // 1 = commit, 0 = partial
    bool want_tokens = false;
    const WsSession* ws_owner = nullptr;
    std::function<bool()> cancelled;
    std::shared_ptr<SpeakerTicket> speaker;
    const char* speaker_skip_reason = "busy";
};

static std::deque<std::unique_ptr<PoolTask>> g_queue;
static std::mutex g_mutex;
static std::condition_variable g_cv;
static bool g_stop = false;
static std::vector<Engine*> g_engines;
static Engine* g_preferred_engine = nullptr;
static bool g_preferred_busy = false;
static size_t g_queue_limit = 64;
static size_t g_queue_bytes_limit = 8 * 1024 * 1024;
static size_t g_queue_bytes = 0;
static double g_timeout_ms = 30000;
static std::chrono::milliseconds g_deadline{30000};
static std::string g_model_id = "paraformer-zh";

// Optional bearer token. Empty means the service is open, which is the
// behaviour every earlier measurement was taken under - the gate stays exact
// by not imposing a header that existing clients do not send.
static std::string g_api_key;
static std::atomic<uint64_t> g_auth_rejected{0};

// Shutdown. SIGTERM means "finish what you accepted and exit", which is what a
// service manager expects: dropping in-flight requests turns a rolling restart
// into a burst of client-visible failures on every deploy.
static std::atomic<bool> g_draining{false};
static std::chrono::milliseconds g_shutdown_grace{5000};

// Requests accepted but not yet answered, across their whole life: queued,
// being decoded, and being written back. `g_inflight` above counts only the
// middle stage - it is incremented when a worker pops a task - so it reads zero
// while a request is still sitting in the queue or already has its answer and
// is being serialised. Draining on g_inflight would therefore declare success
// with clients still waiting, which is exactly the failure this exists to
// prevent.
static std::atomic<int> g_accepted{0};
static std::atomic<uint64_t> g_requests{0};
static std::atomic<int> g_inflight{0}; // tasks a worker is currently running
static std::atomic<uint64_t> g_rejected{0};
static std::atomic<uint64_t> g_failed{0};
static std::atomic<uint64_t> g_deadline_expired{0};
static std::atomic<uint64_t> g_cancelled_queued{0};
static std::atomic<size_t> g_ws_sessions{0};
static constexpr size_t max_ws_sessions = 4;
static constexpr size_t max_ws_pending = 4;
static bool g_degraded = false;
static std::atomic<uint64_t> g_event_seq{0};
static double g_seg_gap_ms = 600;       // silence that starts a new subtitle/segment
static double g_punct_gap_ms = 600;     // silence that earns a clause separator
static double g_partial_min_ms = 800;   // new audio needed before a partial
static double g_partial_interval = 600; // minimum time between partials
static double g_partial_max_s = 2.0;    // stop partials past this much audio
static bool g_partials = true;
static std::atomic<uint64_t> g_partials_run{0};
static std::atomic<uint64_t> g_partials_skipped{0};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(pf_clock::now().time_since_epoch()).count();
}

static size_t queue_depth() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_queue.size();
}

static SegmentReply deadline_reply() {
    SegmentReply reply;
    reply.status = 504;
    reply.text = "transcription deadline exceeded";
    return reply;
}

static json error_body(const std::string& message, const std::string& type, const std::string& param = "",
                       const std::string& code = "") {
    json e = {{"message", message}, {"type", type}, {"param", param.empty() ? nullptr : json(param)},
              {"code", code.empty() ? nullptr : json(code)}};
    return json{{"error", e}};
}

static void reply_error(httplib::Response& res, int status, const std::string& message, const std::string& type,
                        const std::string& param = "", const std::string& code = "") {
    res.status = status;
    res.set_content(error_body(message, type, param, code).dump(), "application/json");
}

// ---- Auth ----

// Constant-time compare. A plain == leaks the length of the matching prefix
// through timing, which over a network is enough to recover a token byte by
// byte. The length itself is not secret, so comparing it up front is fine.
static bool token_matches(const std::string& got, const std::string& want) {
    if (got.size() != want.size()) return false;
    unsigned char diff = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        diff |= (unsigned char) (got[i] ^ want[i]);
    }
    return diff == 0;
}

// Pull the credential out of `Authorization: Bearer <token>`. The scheme is
// case-insensitive per RFC 7235, so `bearer` and `BEARER` have to work; the
// comparison is spelled out here because strncasecmp does not exist on Windows
// under that name and this file builds there too.
static std::string bearer_token(const httplib::Request& req) {
    static const char scheme[] = "bearer ";
    const std::string& h = req.get_header_value("Authorization");
    if (h.size() > sizeof(scheme) - 1) {
        bool match = true;
        for (size_t i = 0; i < sizeof(scheme) - 1; ++i) {
            char c = h[i];
            if (c >= 'A' && c <= 'Z') c = (char) (c - 'A' + 'a');
            if (c != scheme[i]) { match = false; break; }
        }
        if (match) {
            std::string t = h.substr(sizeof(scheme) - 1);
            // Trailing whitespace from a sloppy client is not a credential.
            while (!t.empty() && (t.back() == ' ' || t.back() == '\r' || t.back() == '\n')) t.pop_back();
            while (!t.empty() && t.front() == ' ') t.erase(t.begin());
            return t;
        }
    }
    return std::string();
}

// Endpoints a probe needs before it has a credential: liveness, readiness and
// the model list. Everything that accepts audio, or that exposes queue state,
// is behind the token.
static bool path_is_public(const std::string& path) {
    return path == "/health" || path == "/v1/models" || path == "/metrics" ||
           path == "/metrics.json" || path == "/" || path.rfind("/ui", 0) == 0;
}

static bool authorized(const httplib::Request& req, httplib::Response& res) {
    if (g_api_key.empty()) return true;
    if (token_matches(bearer_token(req), g_api_key)) return true;

    // Same envelope shape as every other error, so a client has one parser.
    res.set_header("WWW-Authenticate", "Bearer realm=\"pf-api\"");
    reply_error(res, 401, "missing or invalid bearer token", "authentication_error", "", "invalid_api_key");
    g_auth_rejected.fetch_add(1);
    return false;
}

// ---- Shutdown ----

// The handler only records the request: everything it could do here (close the
// socket, wake the accept loop) is not async-signal-safe. The main thread does
// the work once listen_after_bind() returns.
//
// Declared before the definition because -Wmissing-prototypes wants a visible
// prototype, and it has to keep C linkage to be a valid sa_handler.
extern "C" void pf_on_term(int);
extern "C" void pf_on_term(int) {
    g_draining.store(true);
}

static void install_shutdown_handlers() {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = pf_on_term;
    sigemptyset(&sa.sa_mask);
    // No SA_RESTART: the accept loop must return so main() can begin draining.
    sa.sa_flags = 0;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    // A client that vanishes mid-write must not kill the process.
    std::signal(SIGPIPE, SIG_IGN);
}

// ---- WAV ----

static std::vector<float> decode_pcm16(const char* data, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int16_t sample;
        std::memcpy(&sample, data + i * sizeof(sample), sizeof(sample));
        out[i] = (float) sample / 32768.0f;
    }
    return out;
}

static bool parse_wav(const std::string& bytes, std::vector<float>& out, std::string& err) {
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        err = "file is not a RIFF/WAVE container";
        return false;
    }
    size_t pos = 12;
    int channels = 0, bits = 0, rate = 0, format = 0;
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
            uint16_t ch = 0, bps = 0, tag = 0;
            std::memcpy(&tag, bytes.data() + body, 2);
            std::memcpy(&ch, bytes.data() + body + 2, 2);
            std::memcpy(&rate, bytes.data() + body + 4, 4);
            std::memcpy(&bps, bytes.data() + body + 14, 2);
            channels = ch;
            bits = bps;
            format = tag;
        } else if (std::strcmp(id, "data") == 0) {
            data = bytes.data() + body;
            data_bytes = size;
        }
        pos = body + size + (size & 1);
    }
    if (format != 1 || rate != 16000 || channels != 1 || bits != 16 || !data) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "expected 16 kHz mono 16-bit PCM, got %d Hz, %d channels, %d bits", rate, channels,
                      bits);
        err = buf;
        return false;
    }
    const size_t n = data_bytes / 2;
    if (data_bytes % 2 != 0 || n > pf_segment::max_samples) {
        err = "expected whole PCM16 samples and at most 30 s of audio";
        return false;
    }
    out = decode_pcm16(data, n);
    return true;
}

static std::string srt_time(double seconds) {
    int ms = (int) std::lround(seconds * 1000.0);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d", ms / 3600000, (ms / 60000) % 60, (ms / 1000) % 60, ms % 1000);
    return buf;
}

// ---- engine pool ----

// Commit tasks jump the queue so a partial decode never delays an endpoint.
static std::unique_ptr<PoolTask> pop_task() {
    auto pos = std::find_if(g_queue.begin(), g_queue.end(), [](const auto& task) { return task->priority > 0; });
    if (pos == g_queue.end()) pos = g_queue.begin();
    auto t = std::move(*pos);
    g_queue_bytes -= t->samples.capacity() * sizeof(float);
    g_queue.erase(pos);
    return t;
}

static SegmentReply decode_task(Engine& eng, const PoolTask& task);
static void try_schedule_speaker(PoolTask& task, const std::vector<float>& samples);
static SpeakerResult speaker_result(const PoolTask& task, bool silence);

static void finish_worker_task(Engine* eng) {
    if (eng == g_preferred_engine) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_preferred_busy = false;
        g_inflight.fetch_sub(1);
    } else {
        g_inflight.fetch_sub(1);
    }
}

static void worker_loop(Engine* eng) {
    for (;;) {
        std::unique_ptr<PoolTask> task;
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            g_cv.wait(lock, [eng] {
                const bool may_take = !g_preferred_engine || eng == g_preferred_engine || g_preferred_busy;
                return (!g_queue.empty() && may_take) || g_stop;
            });
            if (g_queue.empty()) {
                if (g_stop) {
                    return;
                }
                continue;
            }
            task = pop_task();
            if (eng == g_preferred_engine) g_preferred_busy = true;
            g_inflight.fetch_add(1);
        }
        g_cv.notify_all();
        const double queue_ms = now_ms() - task->enqueued_ms;
        if (pf_clock::now() >= task->deadline) {
            finish_worker_task(eng);
            task->done(deadline_reply());
            continue;
        }
        if (task->cancelled && task->cancelled()) {
            finish_worker_task(eng);
            SegmentReply reply;
            reply.status = 499;
            reply.text = "request cancelled";
            reply.queue_ms = queue_ms;
            task->done(std::move(reply));
            continue;
        }
        if (g_timeout_ms > 0 && queue_ms > g_timeout_ms) {
            finish_worker_task(eng);
            SegmentReply reply;
            reply.status = 503;
            reply.text = "queue timeout";
            reply.queue_ms = queue_ms;
            task->done(std::move(reply));
            continue;
        }
        SegmentReply reply;
        try {
            reply = decode_task(*eng, *task);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "pf-api: inference on %s failed: %s\n", eng->name.c_str(), e.what());
            reply.status = 500;
            reply.text = "inference failed";
        }
        finish_worker_task(eng);
        if (pf_clock::now() >= task->deadline) {
            task->done(deadline_reply());
            continue;
        }
        reply.queue_ms = queue_ms;
        reply.backend = eng->name;
        if (reply.status == 500) {
            g_failed.fetch_add(1);
        } else if (reply.status == 200) {
            eng->requests.fetch_add(1);
            if (task->priority > 0) reply.speaker = speaker_result(*task, reply.silence);
        }
        task->done(std::move(reply));
    }
}

// Returns false when the queue is full (the caller maps that to 429 / an error event).
static bool submit(std::vector<float>&& samples, int priority, pool_done done, bool want_tokens = false,
                   std::function<bool()> cancelled = {}, pf_clock::time_point deadline = pf_clock::time_point::max(),
                   const WsSession* ws_owner = nullptr) {
    auto task = std::make_unique<PoolTask>();
    task->done = std::move(done);
    task->priority = priority;
    task->want_tokens = want_tokens;
    task->ws_owner = ws_owner;
    task->cancelled = std::move(cancelled);
    task->enqueued_ms = now_ms();
    task->deadline = deadline;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const size_t bytes = samples.capacity() * sizeof(float);
        if (g_queue.size() >= g_queue_limit || bytes > g_queue_bytes_limit - g_queue_bytes) {
            g_rejected.fetch_add(1);
            return false;
        }
        if (priority > 0 && g_queue.empty() && g_inflight.load() == 0) {
            try_schedule_speaker(*task, samples);
        }
        task->samples = std::move(samples);
        g_queue.push_back(std::move(task));
        g_queue_bytes += bytes;
    }
    if (g_preferred_engine) g_cv.notify_all();
    else g_cv.notify_one();
    g_requests.fetch_add(1);
    return true;
}

// Synchronous helper for the REST endpoints.
static SegmentReply transcribe_sync(std::vector<float>&& samples, bool want_tokens = false) {
    const auto deadline = pf_clock::now() + g_deadline;
    auto promise = std::make_shared<std::promise<SegmentReply>>();
    auto fut = promise->get_future();
    const bool ok = submit(std::move(samples), 1, [promise](SegmentReply reply) {
        promise->set_value(std::move(reply));
    }, want_tokens, {}, deadline);
    if (!ok) {
        SegmentReply r;
        r.status = 429;
        r.text = "server is busy, retry later";
        return r;
    }
    if (fut.wait_until(deadline) != std::future_status::ready) {
        g_deadline_expired.fetch_add(1);
        return deadline_reply();
    }
    auto reply = fut.get();
    if (reply.status == 504) g_deadline_expired.fetch_add(1);
    return reply;
}

// Chinese/CJK shows up as multi-byte UTF-8; English transcripts are ASCII.
static bool any_multibyte(const std::string& s) {
    for (unsigned char c : s) {
        if (c >= 0x80) {
            return true;
        }
    }
    return false;
}

// ---- word and clause structure from token times ----

// A word is one or more BPE pieces; a piece ending in @@ continues the word.
static void append_words(std::vector<WordSpan>& out, const paraformer_tokens* t, double offset_ms) {
    if (!t || t->n_tokens <= 0) {
        return;
    }
    WordSpan cur;
    bool open = false;
    for (int i = 0; i < t->n_tokens; ++i) {
        const char* p = t->tokens[i].text ? t->tokens[i].text : "";
        std::string s(p);
        const bool cont = s.size() >= 2 && s.compare(s.size() - 2, 2, "@@") == 0;
        if (!open) {
            cur = WordSpan();
            // Use the start of the original CIF span for the pause test.
            cur.start_ms = offset_ms + t->tokens[i].start_ms;
            open = true;
        }
        cur.text += cont ? s.substr(0, s.size() - 2) : s;
        cur.end_ms = offset_ms + t->tokens[i].t_ms;
        if (!cont) {
            out.push_back(cur);
            open = false;
        }
    }
    if (open) {
        out.push_back(cur);
    }
}

// Insert a separator where consecutive words are separated by a pause longer
// than gap_ms. Acoustic evidence, unlike the keyword restorer that split inside
// single commands.
static std::string punctuate_words(const std::vector<WordSpan>& words, bool zh, double gap_ms) {
    std::string out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            const double gap = words[i].start_ms - words[i - 1].end_ms;
            // Spacing is per pair, not per utterance: a mixed zh+en transcript
            // needs spaces inside its English run and none inside its Chinese one.
            const bool latin_pair = !any_multibyte(words[i].text) && !any_multibyte(words[i - 1].text);
            if (gap >= gap_ms) {
                out += (!any_multibyte(words[i].text) && !any_multibyte(words[i - 1].text)) ? "," : "，";
            } else if (latin_pair) {
                out += " ";
            }
        }
        out += words[i].text;
    }
    out += zh ? "。" : ".";
    return out;
}

// Whole-utterance label from the word scripts: mixed when both carry weight.
// Decoder-side LID: no extra audio pass. On a 4.97 s mixed zh+en clip the first
// Latin token landed 40 ms after the true English onset.
static std::string utterance_language(const std::vector<WordSpan>& words) {
    size_t zh = 0, en = 0;
    for (const auto& w : words) {
        (any_multibyte(w.text) ? zh : en)++;
    }
    if (zh + en == 0) {
        return "zh";
    }
    const double zf = (double) zh / (double) (zh + en);
    if (zf > 0.9) {
        return "zh";
    }
    if (zf < 0.1) {
        return "en";
    }
    return "mixed";
}

static json segments_json(const std::vector<WordSpan>& words, bool zh, double gap_ms, const std::string& fallback) {
    json segs = json::array();
    if (words.empty()) {
        segs.push_back({{"id", 0}, {"start", 0.0}, {"end", 0.0}, {"text", fallback}, {"language", "zh"}});
        return segs;
    }
    size_t idx = 0;
    std::string cur;
    double start = words[0].start_ms;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            const double gap = words[i].start_ms - words[i - 1].end_ms;
            const bool latin_pair = !any_multibyte(words[i].text) && !any_multibyte(words[i - 1].text);
            if (gap >= gap_ms) {
                segs.push_back({{"id", (int) idx++},
                                {"start", start / 1000.0},
                                {"end", words[i - 1].end_ms / 1000.0},
                                {"text", cur},
                                {"language", any_multibyte(cur) ? "zh" : "en"}});
                cur.clear();
                start = words[i].start_ms;
            } else if (latin_pair) {
                cur += " ";
            }
        }
        cur += words[i].text;
    }
    segs.push_back({{"id", (int) idx},
                    {"start", start / 1000.0},
                    {"end", words.back().end_ms / 1000.0},
                    {"text", cur},
                    {"language", any_multibyte(cur) ? "zh" : "en"}});
    return segs;
}

// ---- speaker gate (optional) ----
//
// The CPU scorer only publishes results that are ready when ASR completes.

static std::string g_speaker_model;
static std::string g_enroll_dir;
static float g_speaker_threshold = 0.55f;
static bool g_speaker_enabled = false;

// Server-side VAD. Conservative by design: a client that already cut at its own
// VAD endpoint sends almost no tail silence, so the gate must never require a
// closed segment. The audio is only reshaped when the commit is longer than
// --vad-split-s; everything else is transcribed untouched.
static std::string g_vad_model;
static float g_vad_split_s = 18.0f;   // only split commits longer than this
static float g_vad_piece_s = 15.0f;    // target length when merging speech spans
static float g_silence_rms = 0.004f;   // below this, "no speech" means empty text
static float g_speaker_max_s = 1.5f;    // only the first N seconds are embedded
static std::atomic<uint64_t> g_speaker_deferred{0};
static std::atomic<uint64_t> g_speaker_started{0};
static std::atomic<uint64_t> g_speaker_assessed{0};
static float g_vad_threshold = 0.5f;
static float g_vad_min_silence = 0.15f;
static float g_vad_min_speech = 0.20f;
static bool g_vad_enabled = false;

#ifdef XASR_SPEAKER
static std::unique_ptr<const SherpaOnnxSpeakerEmbeddingExtractor, decltype(&SherpaOnnxDestroySpeakerEmbeddingExtractor)>
    g_extractor{nullptr, &SherpaOnnxDestroySpeakerEmbeddingExtractor};
static std::vector<float> g_voiceprint;

static bool spk_read_wav(const std::string& path, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    if (n < 44 || n > 2 * 1024 * 1024) {
        std::fclose(f);
        return false;
    }
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> bytes((size_t) n);
    const bool ok = std::fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
    std::fclose(f);
    if (!ok) {
        return false;
    }
    std::string err;
    return parse_wav(std::string(bytes.data(), bytes.size()), out, err);
}

static std::vector<float> spk_embed(const std::vector<float>& samples) {
    if (!g_extractor) {
        return {};
    }
    const auto* extractor = g_extractor.get();
    std::unique_ptr<const SherpaOnnxOnlineStream, decltype(&SherpaOnnxDestroyOnlineStream)> s(
        SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor), &SherpaOnnxDestroyOnlineStream);
    if (!s) return {};
    SherpaOnnxOnlineStreamAcceptWaveform(s.get(), 16000, samples.data(), (int32_t) samples.size());
    SherpaOnnxOnlineStreamInputFinished(s.get());
    if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(extractor, s.get())) return {};
    std::unique_ptr<const float, decltype(&SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding)> v(
        SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor, s.get()),
        &SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding);
    if (!v) return {};
    const int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor);
    if (dim <= 0) return {};
    return {v.get(), v.get() + dim};
}

static std::optional<float> spk_cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return std::nullopt;
    }
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    const double norm = std::sqrt(na * nb);
    if (!std::isfinite(norm) || norm <= 0.0 || !std::isfinite(dot)) return std::nullopt;
    return (float) std::clamp(dot / norm, -1.0, 1.0);
}

static bool spk_init() {
    if (g_speaker_model.empty() && g_enroll_dir.empty()) return true;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(g_speaker_model, ec) || !std::filesystem::is_directory(g_enroll_dir, ec)) {
        std::fprintf(stderr, "pf-api: speaker scoring needs a model file and enrollment directory\n");
        return false;
    }
    SherpaOnnxSpeakerEmbeddingExtractorConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.model = g_speaker_model.c_str();
    cfg.num_threads = 1;
    cfg.debug = 0;
    cfg.provider = "cpu";
    g_extractor.reset(SherpaOnnxCreateSpeakerEmbeddingExtractor(&cfg));
    if (!g_extractor) {
        std::fprintf(stderr, "pf-api: cannot load speaker model %s\n", g_speaker_model.c_str());
        return false;
    }
    const int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(g_extractor.get());
    std::fprintf(stderr, "pf-api: speaker model %s, dim=%d\n", g_speaker_model.c_str(), dim);
    if (dim <= 0) return false;
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(g_enroll_dir)) {
        const std::string p = entry.path().string();
        if (p.size() > 4 && p.compare(p.size() - 4, 4, ".wav") == 0) {
            files.push_back(p);
        }
    }
    std::sort(files.begin(), files.end());
    std::vector<float> acc((size_t) dim, 0.0f);
    int used = 0;
    double self_cos = 0;
    std::vector<std::vector<float>> embs;
    for (const auto& p : files) {
        std::vector<float> samples;
        if (!spk_read_wav(p, samples) || samples.empty()) {
            continue;
        }
        std::vector<float> e = spk_embed(samples);
        if (e.size() != (size_t) dim) {
            continue;
        }
        double nrm = 0;
        for (float v : e) nrm += (double) v * v;
        nrm = std::sqrt(nrm);
        if (!std::isfinite(nrm) || nrm <= 0.0) continue;
        for (size_t i = 0; i < e.size(); ++i) acc[i] += (float) (e[i] / nrm);
        embs.push_back(e);
        ++used;
    }
    if (used == 0) {
        std::fprintf(stderr, "pf-api: no usable enrollment wavs in %s\n", g_enroll_dir.c_str());
        return false;
    }
    double nrm = 0;
    for (float v : acc) nrm += (double) v * v;
    nrm = std::sqrt(nrm);
    if (!std::isfinite(nrm) || nrm <= 0.0) return false;
    for (float& v : acc) v = (float) (v / nrm);
    g_voiceprint = acc;
    for (const auto& e : embs) self_cos += spk_cosine(e, g_voiceprint).value_or(0.0f);
    std::fprintf(stderr, "pf-api: speaker gate on, enrolled %d clips, mean self-cosine %.3f, threshold %.2f\n", used,
                 self_cos / used, g_speaker_threshold);
    g_speaker_enabled = true;
    return true;
}

class SpeakerWorker {
public:
    SpeakerWorker() : worker(&SpeakerWorker::run, this) {}
    ~SpeakerWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        cv.notify_one();
        worker.join();
    }

    void schedule(PoolTask& task, const std::vector<float>& samples) {
        const size_t n = std::min(samples.size(), (size_t) (g_speaker_max_s * 16000.0f));
        std::lock_guard<std::mutex> lock(mutex);
        if (job) return;
        if (std::none_of(samples.begin(), samples.begin() + n, [](float v) { return v != 0.0f; })) {
            task.speaker_skip_reason = "no_audio_in_prefix";
            return;
        }
        head.assign(samples.begin(), samples.begin() + n);
        job = std::make_shared<SpeakerTicket>();
        task.speaker = job;
        g_speaker_started.fetch_add(1);
        cv.notify_one();
    }

private:
    void run() {
        for (;;) {
            std::vector<float> samples;
            std::shared_ptr<SpeakerTicket> ticket;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stopping || job; });
                if (stopping) return;
                ticket = job;
                samples.swap(head);
            }
            try {
                ticket->score = spk_cosine(spk_embed(samples), g_voiceprint);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "pf-api: speaker scoring failed: %s\n", e.what());
            }
            ticket->ready.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(mutex);
                job.reset();
            }
        }
    }

    std::mutex mutex;
    std::condition_variable cv;
    bool stopping = false;
    std::vector<float> head;
    std::shared_ptr<SpeakerTicket> job;
    std::thread worker;
};

static std::unique_ptr<SpeakerWorker> g_speaker_worker;

// ---- server-side VAD ----

static bool vad_init(Engine& eng) {
    if (g_vad_model.empty()) {
        return true;
    }
    SherpaOnnxVadModelConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.silero_vad.model = g_vad_model.c_str();
    cfg.silero_vad.threshold = g_vad_threshold;
    cfg.silero_vad.min_silence_duration = g_vad_min_silence;
    cfg.silero_vad.min_speech_duration = g_vad_min_speech;
    cfg.silero_vad.window_size = 512;
    cfg.silero_vad.max_speech_duration = g_vad_split_s;
    cfg.sample_rate = 16000;
    cfg.num_threads = 1;
    cfg.provider = "cpu";
    eng.vad.reset(SherpaOnnxCreateVoiceActivityDetector(&cfg, 30.0f));
    return static_cast<bool>(eng.vad);
}

static std::vector<std::pair<int, int>> vad_spans(Engine& eng, const std::vector<float>& samples) {
    std::vector<std::pair<int, int>> spans;
    const auto* vad = eng.vad.get();
    SherpaOnnxVoiceActivityDetectorReset(vad);
    for (size_t i = 0; i < samples.size(); i += 512) {
        const size_t n = std::min<size_t>(512, samples.size() - i);
        SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, samples.data() + i, (int32_t) n);
    }
    SherpaOnnxVoiceActivityDetectorFlush(vad);
    while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
        const SherpaOnnxSpeechSegment* seg = SherpaOnnxVoiceActivityDetectorFront(vad);
        if (seg) {
            spans.emplace_back(seg->start, seg->start + seg->n);
            SherpaOnnxDestroySpeechSegment(seg);
        }
        SherpaOnnxVoiceActivityDetectorPop(vad);
    }
    std::sort(spans.begin(), spans.end());
    return spans;
}
#else
static bool spk_init() {
    return g_speaker_model.empty() && g_enroll_dir.empty();
}
static bool vad_init(Engine&) {
    return g_vad_model.empty();
}
static std::vector<std::pair<int, int>> vad_spans(Engine&, const std::vector<float>&) {
    return {};
}
#endif

static void try_schedule_speaker(PoolTask& task, const std::vector<float>& samples) {
#ifdef XASR_SPEAKER
    if (g_speaker_worker) g_speaker_worker->schedule(task, samples);
#else
    (void) task;
    (void) samples;
#endif
}

static SpeakerResult speaker_result(const PoolTask& task, bool silence) {
    if (!g_speaker_enabled) return {};
    if (silence) return {"not_assessed", "silence", std::nullopt};
    const char* reason = task.speaker_skip_reason;
    if (task.speaker) {
        if (!task.speaker->ready.load(std::memory_order_acquire)) {
            reason = "in_progress";
        } else if (task.speaker->score) {
            g_speaker_assessed.fetch_add(1);
            return {"assessed", nullptr, task.speaker->score};
        } else {
            reason = "no_embedding";
        }
    }
    g_speaker_deferred.fetch_add(1);
    return {"not_assessed", reason, std::nullopt};
}

static void set_speaker_json(json& body, const SpeakerResult& speaker) {
    body["speaker_status"] = speaker.status;
    if (speaker.reason) body["speaker_reason"] = speaker.reason;
    if (speaker.score) {
        body["speaker_score"] = *speaker.score;
        body["speaker_ok"] = *speaker.score >= g_speaker_threshold;
    }
}

// Pieces to transcribe for one committed segment. Normal cabin commits (already
// cut at the client's VAD) stay a single piece; only over-long audio is split, so
// a client with its own VAD and little tail silence is never re-cut.
static std::vector<std::pair<size_t, size_t>> plan_pieces(Engine& eng, const std::vector<float>& samples) {
    std::vector<std::pair<size_t, size_t>> pieces;
    double energy = 0.0;
    for (float sample : samples) energy += (double) sample * sample;
    if (energy == 0.0) {
        return pieces;
    }
    if (!g_vad_enabled) return {{0, samples.size()}};
    const bool long_segment = samples.size() > (size_t) (g_vad_split_s * 16000.0f);
    const bool quiet = energy < (double) g_silence_rms * g_silence_rms * samples.size();
    if (!long_segment && !quiet) {
        return {{0, samples.size()}};
    }

    const auto spans = vad_spans(eng, samples);
    if (quiet && spans.empty()) return pieces;
    if (!long_segment || spans.empty()) return {{0, samples.size()}};

    const size_t piece_len = (size_t) (g_vad_piece_s * 16000.0f);
    for (const auto& sp : spans) {
        const size_t a = (size_t) std::max(0, sp.first);
        const size_t b = std::min(samples.size(), (size_t) std::max(0, sp.second));
        if (b <= a + 1600) continue;
        if (!pieces.empty() && (a <= pieces.back().second || b - pieces.back().first <= piece_len)) {
            pieces.back().second = std::max(b, pieces.back().second);
        } else {
            pieces.emplace_back(a, b);
        }
    }
    if (pieces.empty()) return {{0, samples.size()}};

    // Keep up to 100 ms of surrounding audio without duplicating neighboring pieces.
    size_t left_limit = 0;
    for (size_t i = 0; i < pieces.size(); ++i) {
        auto& p = pieces[i];
        const size_t right_limit = i + 1 < pieces.size() ? (p.second + pieces[i + 1].first) / 2 : samples.size();
        p.first = std::max(left_limit, p.first > 1600 ? p.first - 1600 : 0);
        p.second = std::min(right_limit, p.second + 1600);
        left_limit = right_limit;
    }
    return pieces;
}

// Join piece transcripts: a space only between two ASCII ends/ starts.
static void append_text(std::string& out, const std::string& add) {
    if (add.empty()) {
        return;
    }
    if (!out.empty()) {
        const unsigned char a = (unsigned char) out.back();
        const unsigned char b = (unsigned char) add.front();
        const bool latin = a < 0x80 && b < 0x80;
        if (latin && a != ' ') {
            out += ' ';
        }
    }
    out += add;
}

static SegmentReply decode_task(Engine& eng, const PoolTask& task) {
    SegmentReply reply;
    const double t0 = now_ms();
    const auto& samples = task.samples;
    const auto pieces = task.priority > 0 ? plan_pieces(eng, samples)
                                         : std::vector<std::pair<size_t, size_t>>{{0, samples.size()}};
    reply.vad_ms = now_ms() - t0;
    reply.pieces = pieces.size();
    reply.silence = pieces.empty();
    for (const auto& piece : pieces) {
        if (pf_clock::now() >= task.deadline) return deadline_reply();
        if (task.cancelled && task.cancelled()) {
            reply.status = 499;
            reply.text = "request cancelled";
            break;
        }
        const float* pcm = samples.data() + piece.first;
        const int count = (int) (piece.second - piece.first);
        const double begin_ms = now_ms();
        HtpWatchdog::Guard guard(eng.watchdog, "inference");
        if (task.want_tokens) {
            std::unique_ptr<paraformer_tokens, TokensDeleter> tokens(paraformer_transcribe_tokens(eng.ctx.get(), pcm, count));
            if (tokens) {
                append_text(reply.text, tokens->text);
                append_words(reply.words, tokens.get(), (double) piece.first / 16.0);
            } else {
                reply.status = 500;
            }
        } else {
            std::unique_ptr<char, decltype(&std::free)> text(paraformer_transcribe(eng.ctx.get(), pcm, count), &std::free);
            if (text) append_text(reply.text, text.get());
            else reply.status = 500;
        }
        reply.infer_ms += now_ms() - begin_ms;
        if (reply.status != 200) {
            reply.text = "inference failed";
            break;
        }
    }
    return reply;
}

// ---- base64 ----

static bool base64_decode(const std::string& in, std::string& out) {
    static const auto table = [] {
        std::array<int8_t, 256> alphabet;
        alphabet.fill(-1);
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) {
            alphabet[(uint8_t) a[i]] = (int8_t) i;
        }
        return alphabet;
    }();
    out.clear();
    out.reserve(in.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=' ) {
            break;
        }
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            continue;
        }
        const int8_t v = table[(uint8_t) c];
        if (v < 0) {
            return false;
        }
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char) ((acc >> bits) & 0xFF));
        }
    }
    return true;
}

// ---- Realtime session ----

struct WsCommit {
    std::string item;
    pf_clock::time_point deadline;
};

struct WsSession {
    std::mutex m;
    std::deque<std::string> outbox;
    std::deque<WsCommit> pending;
    std::vector<float> pcm;       // 16 kHz mono
    std::string language = "zh";
    std::string partial_text;
    uint64_t gen = 0;             // invalidates previews on commit or clear
    size_t partial_mark = 0;      // pcm size when the last partial was scheduled
    double last_partial_ms = 0;
    bool partial_inflight = false;
    bool closed = false;
    uint64_t commits = 0;
};

static std::string next_event_id() {
    return "event_" + std::to_string(++g_event_seq);
}

static void ws_queue_event(WsSession& s, const char* type, json body) {
    body["type"] = type;
    body["event_id"] = next_event_id();
    s.outbox.push_back(body.dump());
}

static void ws_send_event(WsSession& s, const char* type, json body) {
    std::lock_guard<std::mutex> lock(s.m);
    ws_queue_event(s, type, std::move(body));
}

static void ws_send_error(WsSession& s, const std::string& message, const std::string& type,
                          const std::string& param = "", const std::string& code = "") {
    ws_send_event(s, "error", error_body(message, type, param, code));
}

static void ws_final_error(WsSession& s, const std::string& item, const SegmentReply& reply) {
    const bool expired = reply.status == 504;
    if (expired) g_deadline_expired.fetch_add(1);
    auto body = error_body(reply.text, "server_error", "", expired ? "deadline_exceeded" : "");
    body["item_id"] = item;
    ws_queue_event(s, "error", std::move(body));
}

static json session_object(const WsSession& s) {
    return json{{"id", "sess_" + std::to_string((uintptr_t) &s % 100000)},
                {"object", "realtime.session"},
                {"model", g_model_id},
                {"audio", {{"input", {{"format", {{"type", "audio/pcm"}, {"rate", 16000}}}}}}},
                {"input_audio_transcription", {{"model", g_model_id}, {"language", s.language}}}};
}

// Schedule a partial decode of the audio received so far.
static void maybe_schedule_partial(const std::shared_ptr<WsSession>& s) {
    if (!g_partials) {
        return;
    }
    // A partial costs a full decode. Never let one run while a commit (or any
    // other request) is waiting, and stop partials once the segment is long
    // enough that the final matters more than the preview.
    if (queue_depth() > 0) {
        g_partials_skipped.fetch_add(1);
        return;
    }
    std::lock_guard<std::mutex> lock(s->m);
    if (g_partial_max_s > 0 && s->pcm.size() > (size_t) (g_partial_max_s * 16000.0)) {
        g_partials_skipped.fetch_add(1);
        return;
    }
    const size_t new_samples = s->pcm.size() - s->partial_mark;
    if (s->partial_inflight || new_samples < (size_t) (g_partial_min_ms * 16.0)) {
        return;
    }
    const double t = now_ms();
    if (t - s->last_partial_ms < g_partial_interval) {
        return;
    }
    s->partial_inflight = true;
    s->partial_mark = s->pcm.size();
    s->last_partial_ms = t;
    std::vector<float> snapshot = s->pcm;
    const uint64_t gen = s->gen;
    const std::string item = "item_" + std::to_string(s->commits + 1);
    const bool ok = submit(std::move(snapshot), 0, [s, gen, item](SegmentReply reply) {
        std::lock_guard<std::mutex> lock(s->m);
        s->partial_inflight = false;
        auto& text = reply.text;
        if (reply.status != 200 || s->closed || gen != s->gen || text.size() <= s->partial_text.size() ||
            text.compare(0, s->partial_text.size(), s->partial_text) != 0) {
            return;
        }
        auto delta = text.substr(s->partial_text.size());
        s->partial_text = std::move(text);
        ws_queue_event(*s, "conversation.item.input_audio_transcription.delta",
                       {{"item_id", item}, {"content_index", 0}, {"delta", delta}});
    }, false, [s, gen] {
        std::lock_guard<std::mutex> lock(s->m);
        return s->closed || gen != s->gen;
    }, pf_clock::time_point::max(), s.get());
    if (!ok) {
        s->partial_inflight = false;
        return;
    }
    g_partials_run.fetch_add(1);
}

static void ws_handle_text(const std::shared_ptr<WsSession>& s, const std::string& msg) {
    if (msg.size() > 4 * ((pf_segment::max_samples * sizeof(int16_t) + 2) / 3) + 4096) {
        ws_send_error(*s, "event exceeds maximum size", "invalid_request_error");
        return;
    }
    json j;
    try {
        j = json::parse(msg);
    } catch (const std::exception&) {
        ws_send_error(*s, "client event is not valid JSON", "invalid_request_error");
        return;
    }
    if (!j.is_object() || !j.contains("type") || !j["type"].is_string()) {
        ws_send_error(*s, "event needs a string 'type' field", "invalid_request_error", "type");
        return;
    }
    const std::string type = j["type"].get<std::string>();
    if (type == "session.update") {
        const json& sess = j.contains("session") ? j["session"] : json::object();
        if (!sess.is_object() || (sess.contains("input_audio_transcription") && !sess["input_audio_transcription"].is_object())) {
            ws_send_error(*s, "session and input_audio_transcription must be objects", "invalid_request_error", "session");
            return;
        }
        if (sess.contains("input_audio_transcription") && sess["input_audio_transcription"].contains("language")) {
            const auto& language = sess["input_audio_transcription"]["language"];
            if (!language.is_string() || language.get_ref<const std::string&>().size() > 32) {
                ws_send_error(*s, "language must be a string of at most 32 bytes", "invalid_request_error", "language");
                return;
            }
            s->language = language.get<std::string>();
        }
        ws_send_event(*s, "session.updated", {{"session", session_object(*s)}});
        return;
    }
    if (type == "input_audio_buffer.append") {
        std::string raw;
        if (!j.contains("audio") || !j["audio"].is_string() || !base64_decode(j["audio"].get_ref<const std::string&>(), raw)) {
            ws_send_error(*s, "append needs a base64 'audio' field", "invalid_request_error", "audio");
            return;
        }
        if (raw.size() % 2 != 0) {
            ws_send_error(*s, "audio must be 16-bit PCM", "invalid_request_error", "audio");
            return;
        }
        const size_t n = raw.size() / 2;
        std::vector<float> f = decode_pcm16(raw.data(), n);
        {
            std::lock_guard<std::mutex> lock(s->m);
            if (s->pcm.size() + f.size() > pf_segment::max_samples) {
                ws_queue_event(*s, "error", error_body("segment exceeds 30 s, commit or clear first", "invalid_request_error"));
                return;
            }
            s->pcm.insert(s->pcm.end(), f.begin(), f.end());
        }
        maybe_schedule_partial(s);
        return;
    }
    if (type == "input_audio_buffer.commit") {
        std::lock_guard<std::mutex> lock(s->m);
        if (s->pcm.empty()) {
            ws_queue_event(*s, "error", error_body("buffer is empty", "invalid_request_error"));
            return;
        }
        if (s->pending.size() >= max_ws_pending) {
            g_rejected.fetch_add(1);
            ws_queue_event(*s, "error", error_body("too many pending commits, retry later", "rate_limit_error"));
            return;
        }
        const std::string item = "item_" + std::to_string(s->commits + 1);
        const auto deadline = pf_clock::now() + g_deadline;
        const double duration = (double) s->pcm.size() / 16000.0;
        const bool ok = submit(std::move(s->pcm), 1, [s, item, duration, deadline](SegmentReply reply) {
            std::lock_guard<std::mutex> lock(s->m);
            if (s->closed) return;
            const auto pending = std::find_if(s->pending.begin(), s->pending.end(), [&](const auto& commit) {
                return commit.item == item;
            });
            if (pending == s->pending.end()) return;
            s->pending.erase(pending);
            if (pf_clock::now() >= deadline) reply = deadline_reply();
            if (reply.status != 200) {
                ws_final_error(*s, item, reply);
                return;
            }
            json body = {{"item_id", item},
                         {"content_index", 0},
                         {"transcript", reply.text},
                         {"vad_pieces", reply.pieces},
                         {"usage", {{"type", "duration"}, {"seconds", duration}}}};
            if (reply.silence) body["empty"] = "silence";
            set_speaker_json(body, reply.speaker);
            ws_queue_event(*s, "conversation.item.input_audio_transcription.completed", std::move(body));
            ws_queue_event(*s, "x.pf.metrics", {{"item_id", item}, {"queue_ms", reply.queue_ms},
                                               {"infer_ms", reply.infer_ms}, {"vad_ms", reply.vad_ms},
                                               {"backend", reply.backend}});
        }, false, [s] {
            std::lock_guard<std::mutex> lock(s->m);
            return s->closed;
        }, deadline, s.get());
        if (!ok) {
            ws_queue_event(*s, "error", error_body("server is busy, retry later", "rate_limit_error"));
            return;
        }
        s->pending.push_back({item, deadline});
        json prev = s->commits > 0 ? json("item_" + std::to_string(s->commits)) : json(nullptr);
        ++s->commits;
        ++s->gen;
        s->pcm.clear();
        s->partial_text.clear();
        s->partial_mark = 0;
        ws_queue_event(*s, "input_audio_buffer.committed", {{"item_id", item}, {"previous_item_id", std::move(prev)}});
        return;
    }
    if (type == "input_audio_buffer.clear") {
        {
            std::lock_guard<std::mutex> lock(s->m);
            std::vector<float>().swap(s->pcm);
            s->partial_text.clear();
            s->partial_mark = 0;
            ++s->gen;
            ws_queue_event(*s, "input_audio_buffer.cleared", json::object());
        }
        return;
    }
    ws_send_error(*s, "unsupported event type", "invalid_request_error", "type", "unsupported_event");
}

// ---- WS loop ----

static void ws_close_session(WsSession& s) {
    std::lock_guard<std::mutex> session_lock(s.m);
    s.closed = true;
    std::vector<float>().swap(s.pcm);
    s.outbox.clear();
    s.pending.clear();
    std::lock_guard<std::mutex> queue_lock(g_mutex);
    for (auto it = g_queue.begin(); it != g_queue.end();) {
        if ((*it)->ws_owner != &s) {
            ++it;
            continue;
        }
        g_queue_bytes -= (*it)->samples.capacity() * sizeof(float);
        it = g_queue.erase(it);
        g_cancelled_queued.fetch_add(1);
    }
}

static void ws_serve(httplib::ws::WebSocket& ws) {
    if (g_ws_sessions.fetch_add(1) >= max_ws_sessions) {
        g_ws_sessions.fetch_sub(1);
        g_rejected.fetch_add(1);
        ws.close(httplib::ws::CloseStatus::PolicyViolation, "too many realtime sessions");
        return;
    }
    struct SessionSlot {
        ~SessionSlot() { g_ws_sessions.fetch_sub(1); }
    } slot;
    auto s = std::make_shared<WsSession>();
    ws.set_read_timeout(std::chrono::milliseconds(40));
    ws_send_event(*s, "session.created", {{"session", session_object(*s)}});
    std::string msg;
    while (ws.is_open()) {
        const auto r = ws.read(msg);
        if (r == httplib::ws::ReadResult::Text || r == httplib::ws::ReadResult::Binary) {
            ws_handle_text(s, msg);
        } else if (r == httplib::ws::ReadResult::Fail) {
            break;
        }
        std::deque<std::string> pending;
        {
            std::lock_guard<std::mutex> lock(s->m);
            const auto now = pf_clock::now();
            while (!s->pending.empty() && s->pending.front().deadline <= now) {
                ws_final_error(*s, s->pending.front().item, deadline_reply());
                s->pending.pop_front();
            }
            pending.swap(s->outbox);
        }
        for (const auto& e : pending) {
            if (!ws.send(e)) {
                ws_close_session(*s);
                return;
            }
        }
    }
    ws_close_session(*s);
}

// ---- HTTP ----

static void set_segment_headers(httplib::Response& res, const SegmentReply& reply) {
    res.set_header("X-Queue-Ms", std::to_string((int) std::lround(reply.queue_ms)));
    res.set_header("X-Infer-Ms", std::to_string((int) std::lround(reply.infer_ms)));
    res.set_header("X-Vad-Ms", std::to_string(reply.vad_ms));
    res.set_header("X-Vad-Pieces", std::to_string(reply.pieces));
    if (!reply.backend.empty()) res.set_header("X-ASR-Backend", reply.backend);
    if (reply.silence) res.set_header("X-Empty", "silence");
}

static void set_speaker_headers(httplib::Response& res, const SpeakerResult& speaker) {
    res.set_header("X-Speaker-Status", speaker.status);
    if (speaker.reason) res.set_header("X-Speaker-Reason", speaker.reason);
    if (speaker.score) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", *speaker.score);
        res.set_header("X-Speaker-Score", buf);
        res.set_header("X-Speaker-Ok", *speaker.score >= g_speaker_threshold ? "1" : "0");
    } else if (std::strcmp(speaker.status, "not_assessed") == 0 && std::strcmp(speaker.reason, "silence") != 0) {
        res.set_header("X-Speaker-Deferred", "1");
    }
}

// RAII so that an added early return cannot forget to decrement. A counter
// that leaks upward would make every future shutdown wait the full grace.
struct AcceptedGuard {
    AcceptedGuard() { g_accepted.fetch_add(1); }
    ~AcceptedGuard() { g_accepted.fetch_sub(1); }
    AcceptedGuard(const AcceptedGuard&) = delete;
    AcceptedGuard& operator=(const AcceptedGuard&) = delete;
};

static void handle_transcription(const httplib::Request& req, httplib::Response& res) {
    AcceptedGuard accepted;
    if (!req.form.has_file("file")) {
        reply_error(res, 400, "missing form field 'file'", "invalid_request_error", "file");
        return;
    }
    const auto& file = req.form.get_file("file");
    std::vector<float> samples;
    std::string err;
    if (!parse_wav(file.content, samples, err)) {
        reply_error(res, 400, err, "invalid_request_error", "file");
        return;
    }
    if (samples.empty()) {
        reply_error(res, 400, "audio is empty", "invalid_request_error", "file");
        return;
    }
    const std::string fmt = req.form.has_field("response_format") ? req.form.get_field("response_format") : "json";
    if (fmt != "json" && fmt != "text" && fmt != "verbose_json" && fmt != "srt") {
        reply_error(res, 400, "unsupported response_format '" + fmt + "'", "invalid_request_error", "response_format");
        return;
    }
    const std::string punct = req.form.has_field("punctuate") ? req.form.get_field("punctuate") : "false";
    const bool want_punct = punct == "true" || punct == "1";
    const bool want_tokens = fmt == "verbose_json" || fmt == "srt" || want_punct;
    const double duration = (double) samples.size() / 16000.0;
    const SegmentReply r = transcribe_sync(std::move(samples), want_tokens);
    set_segment_headers(res, r);
    if (r.status != 200) {
        reply_error(res, r.status, r.text, r.status == 429 ? "rate_limit_error" : "server_error", "",
                    r.status == 504 ? "deadline_exceeded" : "");
        return;
    }
    const auto& words = r.words;
    const bool zh = any_multibyte(r.text);
    const std::string detected = r.text.empty() ? "und" :
        (words.empty() ? (zh ? "zh" : "en") : utterance_language(words));
    const std::string text = want_punct && !words.empty() ? punctuate_words(words, zh, g_punct_gap_ms) : r.text;
    res.set_header("X-Language", detected);
    set_speaker_headers(res, r.speaker);
    if (fmt == "text") {
        res.set_content(text, "text/plain; charset=utf-8");
        return;
    }
    if (fmt == "verbose_json") {
        json jw = json::array();
        for (const auto& w : words) {
            jw.push_back({{"word", w.text}, {"start", w.start_ms / 1000.0}, {"end", w.end_ms / 1000.0}});
        }
        json j = {{"task", "transcribe"},
                  {"language", detected},
                  {"duration", duration},
                  {"text", text},
                  {"words", jw},
                  {"segments", r.silence ? json::array() : segments_json(words, zh, g_seg_gap_ms, text)}};
        set_speaker_json(j, r.speaker);
        res.set_content(j.dump(2), "application/json");
        return;
    }
    if (fmt == "srt") {
        if (r.silence) {
            res.set_content("", "text/plain; charset=utf-8");
            return;
        }
        const json segs = segments_json(words, zh, g_seg_gap_ms, text);
        std::string body;
        for (size_t i = 0; i < segs.size(); ++i) {
            body += std::to_string(i + 1) + "\n" + srt_time(segs[i]["start"].get<double>()) + " --> " +
                    srt_time(segs[i]["end"].get<double>()) + "\n" + segs[i]["text"].get<std::string>() + "\n\n";
        }
        res.set_content(body.empty() ? text + "\n" : body, "text/plain; charset=utf-8");
        return;
    }
    res.set_content(json{{"text", text}}.dump(), "application/json");
}

static void handle_segment(const httplib::Request& req, httplib::Response& res) {
    AcceptedGuard accepted;
    if (req.body.size() % sizeof(float) != 0 || req.body.empty()) {
        reply_error(res, 400, "body must be non-empty float32 PCM", "invalid_request_error", "body");
        return;
    }
    if (req.body.size() / sizeof(float) > pf_segment::max_samples) {
        reply_error(res, 413, "segment exceeds 30 s", "invalid_request_error", "body");
        return;
    }
    std::vector<float> samples(req.body.size() / sizeof(float));
    std::memcpy(samples.data(), req.body.data(), req.body.size());
    if (!pf_segment::finite(samples)) {
        reply_error(res, 400, "audio contains non-finite samples", "invalid_request_error", "body");
        return;
    }
    const std::string fmt = req.has_param("format") ? req.get_param_value("format") : "json";
    if (fmt != "json" && fmt != "text") {
        reply_error(res, 400, "unsupported format '" + fmt + "'", "invalid_request_error", "format");
        return;
    }
    const double duration = (double) samples.size() / 16000.0;
    const SegmentReply r = transcribe_sync(std::move(samples));
    set_segment_headers(res, r);
    if (r.status != 200) {
        reply_error(res, r.status, r.text, r.status == 429 ? "rate_limit_error" : "server_error", "",
                    r.status == 504 ? "deadline_exceeded" : "");
        return;
    }
    set_speaker_headers(res, r.speaker);
    if (fmt == "text") {
        res.set_content(r.text, "text/plain; charset=utf-8");
        return;
    }
    json body = {{"text", r.text}, {"duration", duration}};
    set_speaker_json(body, r.speaker);
    res.set_content(body.dump(), "application/json");
}

// ---- Metrics ----

// One snapshot read once, rendered in two formats. Two separate readers is how
// /metrics and /metrics.json start disagreeing: the counters move between the
// two requests and a dashboard silently plots a different number than the
// script that gates the deploy.
struct Metrics {
    uint64_t requests;
    uint64_t rejected;
    uint64_t failed;
    uint64_t deadline_expired;
    uint64_t cancelled_queued;
    uint64_t auth_rejected;
    bool cpu_fallback;
    uint64_t partials;
    uint64_t partials_skipped;
    uint64_t speaker_deferred;
    uint64_t speaker_started;
    uint64_t speaker_assessed;
    size_t queue;
    size_t queue_bytes;
    size_t queue_bytes_limit;
    int inflight;
    int accepted;
    int ws_sessions;
    size_t engines;
};

static Metrics collect_metrics() {
    Metrics m;
    m.requests = g_requests.load();
    m.rejected = g_rejected.load();
    m.failed = g_failed.load();
    m.deadline_expired = g_deadline_expired.load();
    m.cancelled_queued = g_cancelled_queued.load();
    m.auth_rejected = g_auth_rejected.load();
    m.cpu_fallback = g_degraded;
    m.partials = g_partials_run.load();
    m.partials_skipped = g_partials_skipped.load();
    m.speaker_deferred = g_speaker_deferred.load();
    m.speaker_started = g_speaker_started.load();
    m.speaker_assessed = g_speaker_assessed.load();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        m.queue = g_queue.size();
        m.queue_bytes = g_queue_bytes;
    }
    m.queue_bytes_limit = g_queue_bytes_limit;
    m.inflight = g_inflight.load();
    m.accepted = g_accepted.load();
    m.ws_sessions = g_ws_sessions.load();
    m.engines = g_engines.size();
    return m;
}

static json metrics_json(const Metrics& m) {
    return json{{"requests", m.requests},
                {"rejected", m.rejected},
                {"failed", m.failed},
                {"deadline_expired", m.deadline_expired},
                {"cancelled_queued", m.cancelled_queued},
                {"auth_rejected", m.auth_rejected},
                {"cpu_fallback", m.cpu_fallback},
                {"partials", m.partials},
                {"partials_skipped", m.partials_skipped},
                {"speaker_deferred", m.speaker_deferred},
                {"speaker_started", m.speaker_started},
                {"speaker_assessed", m.speaker_assessed},
                {"queue", m.queue},
                {"queue_bytes", m.queue_bytes},
                {"queue_bytes_limit", m.queue_bytes_limit},
                {"inflight", m.inflight},
                {"accepted", m.accepted},
                {"ws_sessions", m.ws_sessions},
                {"engines", m.engines}};
}

// Prometheus exposition, text format 0.0.4.
//
// Deliberately no "_total" on any name. A name ending in _total is, by
// convention, a monotonically increasing counter, and Prometheus converts it on
// scrape to a series with the suffix stripped. Naming a gauge with it - queue,
// inflight, accepted, engines all go up and down - makes rate() and
// increase() produce nonsense, which is worse than the name being slightly
// less idiomatic. Same reason there is no HELP on the gauges beyond one line:
// what each one counts is the thing a reader actually needs.
static std::string metrics_prometheus(const Metrics& m) {
    std::ostringstream o;
    auto counter = [&o](const char* name, const char* help, uint64_t v) {
        o << "# HELP " << name << ' ' << help << '\n';
        o << "# TYPE " << name << " counter\n";
        o << name << ' ' << v << '\n';
    };
    auto gauge = [&o](const char* name, const char* help, auto v) {
        o << "# HELP " << name << ' ' << help << '\n';
        o << "# TYPE " << name << " gauge\n";
        o << name << ' ' << v << '\n';
    };

    counter("pf_requests", "Transcription and segment requests started.", m.requests);
    counter("pf_rejected", "Requests refused by admission control.", m.rejected);
    counter("pf_failed", "Requests that ended in an error response.", m.failed);
    counter("pf_deadline_expired", "Requests that reached --deadline-ms.", m.deadline_expired);
    counter("pf_cancelled_queued", "Queued requests dropped before a worker took them.", m.cancelled_queued);
    counter("pf_auth_rejected", "Requests refused for a missing or wrong bearer token.", m.auth_rejected);
    counter("pf_partials", "Partial decodes run for realtime sessions.", m.partials);
    counter("pf_partials_skipped", "Partial decodes skipped to keep a commit first.", m.partials_skipped);
    counter("pf_speaker_started", "Speaker scoring jobs started.", m.speaker_started);
    counter("pf_speaker_assessed", "Speaker scoring jobs that produced a score.", m.speaker_assessed);
    counter("pf_speaker_deferred", "Speaker scoring jobs left without a result.", m.speaker_deferred);

    // 1/0 rather than a label so an alert reads `pf_cpu_fallback == 1` without
    // having to know which label value means degraded.
    //
    // Gauges are emitted through the integer overload on purpose. Streaming a
    // size_t through double prints 8.38861e+06 for 8388608 - Prometheus parses
    // it, so nothing breaks loudly, but the value in the series no longer
    // equals the configured cap and an alert built on it is off by 8 bytes.
    gauge("pf_cpu_fallback", "1 when engines are running on CPU after a fallback.", (int) (m.cpu_fallback ? 1 : 0));
    gauge("pf_queue_depth", "Tasks waiting for a worker.", m.queue);
    gauge("pf_queue_bytes", "Audio bytes held by queued tasks.", m.queue_bytes);
    gauge("pf_queue_bytes_limit", "Configured cap on pf_queue_bytes.", m.queue_bytes_limit);
    gauge("pf_inflight", "Tasks a worker is currently running.", m.inflight);
    gauge("pf_accepted", "Requests accepted and not yet answered, queued or running.", m.accepted);
    gauge("pf_ws_sessions", "Open realtime sessions.", m.ws_sessions);
    gauge("pf_engines", "Engines in this process.", m.engines);
    return o.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf [--device HTP0:0] [--port N] [--engines N] [--threads N] "
                     "[--htp-flash-attn | --no-htp-flash-attn] [--prefer-idle-htp] [--htp-watchdog-ms N] [--htp-warmup-ms N] "
                     "[--bind ADDR] [--queue N] [--queue-mib N] [--timeout-ms N] [--deadline-ms N] "
                     "[--model-id ID] [--warmup WAV] [--allow-cpu-fallback] [--recover-on-cpu] "
                     "[--partial-ms N] [--partial-interval-ms N] [--partial-max-s N] [--no-partials] "
                     "[--seg-gap-ms N] [--punct-gap-ms N] "
                     "[--web-dir DIR] [--speaker-model FILE] [--enroll-dir DIR] [--speaker-threshold F] [--speaker-max-s S] "
                     "[--vad-model FILE] [--vad-split-s S] [--vad-threshold F] [--vad-min-silence S] "
                     "[--vad-min-speech S] [--api-key TOKEN | PF_API_KEY] [--shutdown-grace-ms N]\n",
                     argv[0]);
        return 1;
    }
    const char* model = argv[1];
    const char* device = "CPU";
    // Loopback by default. A speech service that accepts unauthenticated audio
    // should not appear on every interface because of a default nobody chose;
    // putting it on the LAN has to be an explicit --bind. Every recorded
    // measurement used an explicit address, so no number here moves.
    const char* bind_addr = "127.0.0.1";
    const char* warmup = nullptr;
    const char* web_dir = "web";
    int port = 8080;
    int n_engines = 2;
    int threads = 2;
    bool device_set = false;
    bool allow_cpu_fallback = false;
    bool recover_on_cpu = false;
    bool htp_flash_attn = false;
    bool prefer_idle_htp = false;
    std::chrono::milliseconds htp_watchdog_limit{0};
    std::chrono::milliseconds htp_warmup_limit{0};
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = argv[++i];
            device_set = true;
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--engines") == 0 && i + 1 < argc) {
            n_engines = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
            g_api_key = argv[++i];
        } else if (std::strcmp(argv[i], "--shutdown-grace-ms") == 0 && i + 1 < argc) {
            g_shutdown_grace = std::chrono::milliseconds(std::max(0, std::atoi(argv[++i])));
        } else if (std::strcmp(argv[i], "--queue") == 0 && i + 1 < argc) {
            g_queue_limit = (size_t) std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--queue-mib") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const long mib = std::strtol(argv[++i], &end, 10);
            if (*end || mib < 1 || mib > 128) {
                std::fprintf(stderr, "pf-api: --queue-mib must be an integer from 1 to 128\n");
                return 1;
            }
            g_queue_bytes_limit = (size_t) mib * 1024 * 1024;
        } else if (std::strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc) {
            g_timeout_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--deadline-ms") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const long ms = std::strtol(argv[++i], &end, 10);
            if (*end || ms < 1 || ms > 120000) {
                std::fprintf(stderr, "pf-api: --deadline-ms must be an integer from 1 to 120000\n");
                return 1;
            }
            g_deadline = std::chrono::milliseconds(ms);
        } else if (std::strcmp(argv[i], "--model-id") == 0 && i + 1 < argc) {
            g_model_id = argv[++i];
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = argv[++i];
        } else if (std::strcmp(argv[i], "--allow-cpu-fallback") == 0) {
            allow_cpu_fallback = true;
        } else if (std::strcmp(argv[i], "--recover-on-cpu") == 0) {
            recover_on_cpu = true;
        } else if (std::strcmp(argv[i], "--no-htp-flash-attn") == 0) {
            htp_flash_attn = false;
        } else if (std::strcmp(argv[i], "--htp-flash-attn") == 0) {
            htp_flash_attn = true;
        } else if (std::strcmp(argv[i], "--prefer-idle-htp") == 0) {
            prefer_idle_htp = true;
        } else if (std::strcmp(argv[i], "--htp-watchdog-ms") == 0) {
            char* end = nullptr;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "pf-api: --htp-watchdog-ms requires a value\n");
                return 1;
            }
            const long ms = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end || (ms != 0 && (ms < 100 || ms > 300000))) {
                std::fprintf(stderr, "pf-api: --htp-watchdog-ms must be 0 or an integer from 100 to 300000\n");
                return 1;
            }
            htp_watchdog_limit = std::chrono::milliseconds(ms);
        } else if (std::strcmp(argv[i], "--htp-warmup-ms") == 0) {
            char* end = nullptr;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "pf-api: --htp-warmup-ms requires a value\n");
                return 1;
            }
            const long ms = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end || ms < 100 || ms > 600000) {
                std::fprintf(stderr, "pf-api: --htp-warmup-ms must be an integer from 100 to 600000\n");
                return 1;
            }
            htp_warmup_limit = std::chrono::milliseconds(ms);
        } else if (std::strcmp(argv[i], "--web-dir") == 0 && i + 1 < argc) {
            web_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--partial-ms") == 0 && i + 1 < argc) {
            g_partial_min_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--partial-interval-ms") == 0 && i + 1 < argc) {
            g_partial_interval = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-partials") == 0) {
            g_partials = false;
        } else if (std::strcmp(argv[i], "--partial-max-s") == 0 && i + 1 < argc) {
            g_partial_max_s = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--speaker-model") == 0 && i + 1 < argc) {
            g_speaker_model = argv[++i];
        } else if (std::strcmp(argv[i], "--enroll-dir") == 0 && i + 1 < argc) {
            g_enroll_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--speaker-threshold") == 0 && i + 1 < argc) {
            g_speaker_threshold = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--vad-model") == 0 && i + 1 < argc) {
            g_vad_model = argv[++i];
        } else if (std::strcmp(argv[i], "--vad-split-s") == 0 && i + 1 < argc) {
            g_vad_split_s = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--vad-piece-s") == 0 && i + 1 < argc) {
            g_vad_piece_s = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--silence-rms") == 0 && i + 1 < argc) {
            g_silence_rms = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--speaker-max-s") == 0 && i + 1 < argc) {
            g_speaker_max_s = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--vad-threshold") == 0 && i + 1 < argc) {
            g_vad_threshold = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--vad-min-silence") == 0 && i + 1 < argc) {
            g_vad_min_silence = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--vad-min-speech") == 0 && i + 1 < argc) {
            g_vad_min_speech = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--seg-gap-ms") == 0 && i + 1 < argc) {
            g_seg_gap_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--punct-gap-ms") == 0 && i + 1 < argc) {
            g_punct_gap_ms = std::atof(argv[++i]);
        }
    }
    if (recover_on_cpu && (!device_set || std::strncmp(device, "HTP", 3) != 0)) {
        std::fprintf(stderr, "pf-api: --recover-on-cpu requires an HTP --device\n");
        return 1;
    }

    // A token in argv is visible in /proc/<pid>/cmdline to every local process,
    // so the environment is the better default for anything that is not a test.
    // The flag wins if both are set, because an explicit argument is a
    // deliberate act and silently preferring the environment would be worse.
    if (g_api_key.empty()) {
        if (const char* env = std::getenv("PF_API_KEY")) g_api_key = env;
    }

    install_shutdown_handlers();
    g_degraded = recover_on_cpu;
    if (device_set && std::strcmp(device, "CPU") != 0) {
        setenv("PARAFORMER_DEVICE", device, 1);
    }
    if (!g_vad_model.empty()) {
        std::error_code ec;
        const auto valid_seconds = [](float value) { return value > 0.0f && value <= 30.0f; };
        if (!std::filesystem::is_regular_file(g_vad_model, ec) ||
            !valid_seconds(g_vad_split_s) || !valid_seconds(g_vad_piece_s) ||
            !valid_seconds(g_vad_min_silence) || !valid_seconds(g_vad_min_speech) ||
            !(g_vad_threshold > 0.0f && g_vad_threshold <= 1.0f) ||
            !(g_silence_rms >= 0.0f && g_silence_rms <= 1.0f)) {
            std::fprintf(stderr, "pf-api: invalid VAD model or parameters\n");
            return 3;
        }
        g_vad_enabled = true;
    }
    std::unique_ptr<HtpWatchdog> watchdog;
    if (!recover_on_cpu) {
        if (std::strncmp(device, "HTP", 3) == 0 && htp_watchdog_limit.count() > 0) {
            watchdog = std::make_unique<HtpWatchdog>(htp_watchdog_limit);
        }
        HtpWatchdog::Guard guard(watchdog.get(), "backend discovery",
                                 htp_warmup_limit.count() > 0 ? htp_warmup_limit : htp_watchdog_limit * 20);
        ggml_backend_load_all();
    }

    std::unique_ptr<paraformer_weights, decltype(&paraformer_weights_free)> w(paraformer_weights_load(model, 1),
                                                                           &paraformer_weights_free);
    if (!w) {
        std::fprintf(stderr, "pf-api: cannot load %s\n", model);
        return 2;
    }
    std::vector<float> warm;
    if (warmup) {
        std::string err;
        std::vector<char> bytes;
        FILE* f = std::fopen(warmup, "rb");
        if (f) {
            std::fseek(f, 0, SEEK_END);
            long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            bytes.resize((size_t) std::max(0L, n));
            if (std::fread(bytes.data(), 1, bytes.size(), f) == bytes.size()) {
                parse_wav(std::string(bytes.data(), bytes.size()), warm, err);
            }
            std::fclose(f);
        }
        if (warm.empty()) {
            std::fprintf(stderr, "pf-api: invalid warmup WAV %s: %s\n", warmup, err.c_str());
            return 3;
        }
    }
    if (warm.empty()) {
        warm.assign((size_t) 16000 * 12 / 10, 0.0f);
    }

    std::vector<std::unique_ptr<Engine>> engines;
    for (int i = 0; i < n_engines; ++i) {
        paraformer_context_params params = paraformer_context_default_params();
        params.n_threads = threads;
        const bool wants_accelerator = i == 0 && device_set && std::strcmp(device, "CPU") != 0;
        params.use_gpu = wants_accelerator && !recover_on_cpu;
        params.verbosity = 0;
        if (params.use_gpu && std::strncmp(device, "HTP", 3) == 0) params.flash_attn = htp_flash_attn;
        auto eng = std::make_unique<Engine>();
        eng->requested = wants_accelerator ? device : "CPU";
        const double t0 = now_ms();
        {
            // Warmup covers engine init, weight materialisation and the first
            // graph allocation, which is 1.2 s on SM6650 against 0.3-1.5 s for
            // one inference. Reusing the inference limit here kills the process
            // during start-up whenever the limit is tighter than a warmup, so
            // warmup gets its own, wider bound.
            HtpWatchdog::Guard guard(params.use_gpu ? watchdog.get() : nullptr, "engine warmup",
                                     htp_warmup_limit.count() > 0 ? htp_warmup_limit : htp_watchdog_limit * 20);
            eng->ctx = paraformer_start_engine(w.get(), params, warm);
        }
        if (!eng->ctx && params.use_gpu && allow_cpu_fallback) {
            std::fprintf(stderr, "pf-api: %s init or warmup failed, retrying on CPU\n", device);
            params.use_gpu = false;
            params.flash_attn = paraformer_context_default_params().flash_attn;
            eng->ctx = paraformer_start_engine(w.get(), params, warm);
            g_degraded = true;
        }
        if (!eng->ctx) {
            std::fprintf(stderr, "pf-api: engine %d (%s) init or warmup failed\n", i, eng->requested.c_str());
            return 3;
        }
        if (!vad_init(*eng)) {
            std::fprintf(stderr, "pf-api: cannot initialize server VAD for engine %d\n", i);
            return 3;
        }
        eng->name = paraformer_backend_name(eng->ctx.get());
        if (i == 0 && eng->name.rfind("HTP", 0) == 0) {
            eng->watchdog = watchdog.get();
            if (prefer_idle_htp) g_preferred_engine = eng.get();
        }
        eng->flash_attn = params.flash_attn;
        eng->placement = paraformer_warmup_placement(eng->ctx.get());
        const auto& p = eng->placement;
        std::fprintf(stderr, "pf-api: engine %d requested=%s actual=%s flash_attn=%d encoder=%d/%d decoder=%d/%d ready in %.0f ms\n",
                     i, eng->requested.c_str(), eng->name.c_str(), eng->flash_attn, p.encoder_device_ops, p.encoder_total_ops,
                     p.decoder_device_ops, p.decoder_total_ops, now_ms() - t0);
        // Ops that do not land on the accelerator turn into FastRPC round trips
        // for every request, so measure the placement instead of trusting the
        // flash-attn default. On a v73 part (2 HVX, no HMX) FLASH_ATTN_EXT is
        // unsupported and turning it on pushes 50 encoder and 16 decoder ops
        // off the DSP: 447 -> 355 ms p50 and 1004 -> 710 ms p95 at 4 lanes.
        // Flash is also the less accurate graph there -- it reads light_en02 as
        // "reading leg", where the decomposed graph gets "reading light" and
        // matches the 6.87% en WER release figure. Reporting the gap makes the
        // next part that behaves differently visible without a new experiment.
        if (i == 0 && eng->name.rfind("HTP", 0) == 0) {
            const int off = (p.encoder_total_ops - p.encoder_device_ops) + (p.decoder_total_ops - p.decoder_device_ops);
            if (off > 0) {
                std::fprintf(stderr,
                             "pf-api: %d op(s) in this graph do not run on %s; each one costs a FastRPC round trip per "
                             "request. The default graph avoids this; flash is only worth it if the count is 0 on your part.\n",
                             off, eng->name.c_str());
            }
        }
        g_engines.push_back(eng.get());
        engines.push_back(std::move(eng));
    }
    w.reset();

    httplib::Server srv;
    srv.new_task_queue = [] { return new httplib::ThreadPool(12, 12, 8); };
    srv.set_read_timeout(10, 0);
    srv.set_write_timeout(5, 0);
    srv.set_payload_max_length(2u * 1024u * 1024u);

    if (!(g_speaker_threshold >= -1.0f && g_speaker_threshold <= 1.0f) ||
        !(g_speaker_max_s >= 0.025f && g_speaker_max_s <= 30.0f) || !spk_init()) {
        std::fprintf(stderr, "pf-api: invalid or unsupported speaker scoring configuration\n");
        return 3;
    }

    srv.Get("/health", [htp_watchdog_limit](const httplib::Request&, httplib::Response& res) {
        json backends = json::array();
        for (const auto* eng : g_engines) {
            const auto& p = eng->placement;
            backends.push_back({{"requested", eng->requested}, {"actual", eng->name},
                                {"flash_attn", eng->flash_attn}, {"watchdog_ms", eng->watchdog ? htp_watchdog_limit.count() : 0},
                                {"encoder_device_ops", p.encoder_device_ops}, {"encoder_total_ops", p.encoder_total_ops},
                                {"decoder_device_ops", p.decoder_device_ops}, {"decoder_total_ops", p.decoder_total_ops},
                                // Per-engine request count. Without it there is no way to tell
                                // whether a latency change came from routing or from the engine
                                // itself, and the two look identical from the client side.
                                {"requests", eng->requests.load()}});
        }
        json j = {{"status", g_degraded ? "degraded" : "ok"},
                  {"engines", g_engines.size()},
                  {"backends", std::move(backends)},
                  {"routing", g_preferred_engine ? "prefer_idle_htp" : "shared_queue"},
                  {"queue", queue_depth()},
                  {"speaker_gate", g_speaker_enabled},
                  {"server_vad", g_vad_enabled},
                  {"partials", g_partials},
                  {"model", g_model_id}};
        res.set_content(j.dump(), "application/json");
    });

    srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json j = {{"object", "list"},
                  {"data", {{{"id", g_model_id},
                             {"object", "model"},
                             {"created", 0},
                             {"owned_by", "local"},
                             {"capabilities", {{"streaming", true}, {"timestamps", "char"}, {"languages", {"zh", "en"}},
                                               {"punctuation", false}, {"diarization", false}}}}}}};
        res.set_content(j.dump(), "application/json");
    });

    srv.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        const Metrics m = collect_metrics();
        res.set_content(metrics_prometheus(m), "text/plain; version=0.0.4; charset=utf-8");
    });

    // Kept as the machine-readable twin of /metrics. Every in-tree consumer
    // (smoke_production.py, check_admission.py, bench_ws_realtime.py) parses
    // this JSON, so /metrics can serve the exposition format without breaking
    // them; external scrapers get the format their name implies.
    srv.Get("/metrics.json", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics_json(collect_metrics()).dump(), "application/json");
    });

    srv.Post("/v1/audio/transcriptions", handle_transcription);
    srv.Post("/asr/segment", handle_segment);

    // Test console: static page at /, served from --web-dir when present.
    if (!srv.set_mount_point("/", web_dir)) {
        std::fprintf(stderr, "pf-api: web dir '%s' not found, / serves a short hint\n", web_dir);
        srv.Get("/", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pf-api: no web dir mounted. Use --web-dir examples/paraformer/web\n",
                            "text/plain; charset=utf-8");
        });
    }
    srv.WebSocket("/v1/realtime", [](const httplib::Request&, httplib::ws::WebSocket& ws) { ws_serve(ws); });

    // One gate in front of every route, rather than a check inside each
    // handler. A per-handler check is how an endpoint added later ships
    // unauthenticated: the WebSocket upgrade is the obvious one to forget, and
    // it carries the most audio.
    srv.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        if (path_is_public(req.path)) return httplib::Server::HandlerResponse::Unhandled;
        if (authorized(req, res)) return httplib::Server::HandlerResponse::Unhandled;
        return httplib::Server::HandlerResponse::Handled;
    });

    srv.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string what = "internal error";
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            what = e.what();
        } catch (...) {
        }
        reply_error(res, 500, what, "server_error");
    });

    std::fprintf(stderr, "pf-api: model '%s', %zu engines, queue %zu, partials %s, listening on %s:%d\n",
                 g_model_id.c_str(), g_engines.size(), g_queue_limit, g_partials ? "on" : "off", bind_addr, port);
    if (g_api_key.empty()) {
        std::fprintf(stderr, "pf-api: no --api-key / PF_API_KEY set; every endpoint accepts unauthenticated audio\n");
        // Binding to a wildcard or a concrete non-loopback address without a
        // token publishes the microphone to the network. Say so loudly: the
        // operator may have meant it, but they should have to mean it.
        const bool exposed = std::strcmp(bind_addr, "0.0.0.0") == 0 || std::strcmp(bind_addr, "::") == 0 ||
                             std::strcmp(bind_addr, "[::]") == 0 ||
                             (std::strcmp(bind_addr, "127.0.0.1") != 0 && std::strcmp(bind_addr, "::1") != 0 &&
                              std::strcmp(bind_addr, "localhost") != 0);
        if (exposed) {
            std::fprintf(stderr,
                         "pf-api: WARNING %s is not a loopback address and no token is set, so any host that can\n"
                         "pf-api: WARNING reach this port can submit audio and read the queue counters. Set PF_API_KEY\n"
                         "pf-api: WARNING or bind 127.0.0.1 unless this is deliberate.\n",
                         bind_addr);
        }
    } else {
        std::fprintf(stderr, "pf-api: bearer token required on audio endpoints; /health, /v1/models and /metrics stay open\n");
    }
    if (!srv.bind_to_port(bind_addr, port)) {
        std::fprintf(stderr, "pf-api: cannot listen on %s:%d\n", bind_addr, port);
        return 4;
    }
#ifdef XASR_SPEAKER
    if (g_speaker_enabled) g_speaker_worker = std::make_unique<SpeakerWorker>();
#endif
    for (auto* eng : g_engines) eng->worker = std::thread(worker_loop, eng);

    // A signal can arrive at any time, including while the accept loop is
    // blocked in poll(). This thread only observes the flag; it does NOT stop
    // the server yet. Calling srv.stop() here would close every active
    // connection, which tears down the very handlers the drain is waiting for:
    // measured, the guard dropped to zero before the drain loop started and
    // only the already-buffered response reached the client. So the sequence is
    // drain first, stop second, and both live on this thread so the accept loop
    // keeps running until nothing is left to finish.
    std::thread drain_watch([&srv] {
        while (!g_draining.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        std::fprintf(stderr, "pf-api: shutdown requested; draining before closing the listener\n");

        // Wait for requests that have been accepted to be answered. New
        // requests can still arrive during this window - that is deliberate:
        // the alternative is refusing connections while the operator watches a
        // rolling restart, and the grace bound keeps it from lasting forever.
        const auto drain_start = pf_clock::now();
        const int initial = g_accepted.load();
        std::fprintf(stderr, "pf-api: drain start, accepted=%d inflight=%d, grace %lld ms\n", initial,
                     g_inflight.load(), (long long) g_shutdown_grace.count());
        while (g_accepted.load() > 0 && pf_clock::now() - drain_start < g_shutdown_grace) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        const int left = g_accepted.load();
        if (left > 0) {
            // Deliberately not "their clients will see a reset": stopping the
            // listener does not itself tear down an in-flight handler, so a
            // request that finishes microseconds later still gets a full 200.
            // Measured with grace=120ms and three queued requests: the warning
            // fired and all three clients still received complete bodies, only
            // after the listener had closed. Say what is true - the grace ran
            // out, the outcome is no longer guaranteed - instead of predicting
            // a failure that may not happen and sending a postmortem the wrong
            // way.
            std::fprintf(stderr,
                         "pf-api: grace expired with %d request(s) still accepted; closing the listener, "
                         "their responses are no longer guaranteed\n",
                         left);
        }
        std::fprintf(stderr, "pf-api: closing the listener\n");
        srv.stop();
    });

    const bool ok = srv.listen_after_bind();

    g_draining.store(true);
    drain_watch.join();

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_stop = true;
    }
    g_cv.notify_all();
    for (auto* eng : g_engines) eng->worker.join();
#ifdef XASR_SPEAKER
    g_speaker_worker.reset();
#endif
    if (g_accepted.load() == 0 && g_inflight.load() == 0) {
        std::fprintf(stderr, "pf-api: shut down cleanly\n");
    }
    return ok ? 0 : 4;
}
