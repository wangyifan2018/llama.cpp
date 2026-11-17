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
// Limits: one segment per commit (no server-side VAD), no diarisation, no word
// timestamps yet. See docs/backend/snapdragon/asr-server-api-design.md.

#include "paraformer.h"

#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using pf_clock = std::chrono::steady_clock;

struct Engine {
    paraformer_context* ctx = nullptr;
    std::thread worker;
    std::string name;
    std::atomic<uint64_t> requests{0};
};

struct TokensDeleter {
    void operator()(paraformer_tokens* t) const {
        paraformer_tokens_free(t);
    }
};
using TokensPtr = std::shared_ptr<paraformer_tokens>;

// status, text, queue_ms, infer_ms, tokens (null unless requested)
using pool_done = std::function<void(int, std::string, double, double, TokensPtr)>;

struct PoolTask {
    std::vector<float> samples;
    pool_done done;
    double enqueued_ms = 0;
    int priority = 0;    // 1 = commit, 0 = partial
    bool want_tokens = false;
};

static std::deque<std::unique_ptr<PoolTask>> g_queue;
static std::mutex g_mutex;
static std::condition_variable g_cv;
static bool g_stop = false;
static std::vector<Engine*> g_engines;
static size_t g_queue_limit = 64;
static double g_timeout_ms = 30000;
static std::string g_model_id = "paraformer-zh";
static std::atomic<uint64_t> g_requests{0};
static std::atomic<uint64_t> g_rejected{0};
static std::atomic<uint64_t> g_failed{0};
static std::atomic<uint64_t> g_event_seq{0};
static double g_seg_gap_ms = 600;       // silence that starts a new subtitle/segment
static double g_punct_gap_ms = 600;     // silence that earns a clause separator
static double g_partial_min_ms = 800;   // new audio needed before a partial
static double g_partial_interval = 600; // minimum time between partials
static bool g_partials = true;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(pf_clock::now().time_since_epoch()).count();
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

// ---- WAV ----

static bool parse_wav(const std::string& bytes, std::vector<float>& out, std::string& err) {
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        err = "file is not a RIFF/WAVE container";
        return false;
    }
    size_t pos = 12;
    int channels = 0, bits = 0, rate = 0;
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
    if (rate != 16000 || channels != 1 || bits != 16 || !data) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "expected 16 kHz mono 16-bit PCM, got %d Hz, %d channels, %d bits", rate, channels,
                      bits);
        err = buf;
        return false;
    }
    const size_t n = data_bytes / 2;
    out.resize(n);
    const int16_t* p = reinterpret_cast<const int16_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        out[i] = (float) p[i] / 32768.0f;
    }
    return true;
}

// Linear resampling to 16 kHz. Fine for the Realtime convenience path; the native
// endpoints take 16 kHz directly and are what the product uses.
static std::vector<float> resample_to_16k(const int16_t* pcm, size_t n, int rate) {
    std::vector<float> out;
    if (rate == 16000) {
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            out[i] = (float) pcm[i] / 32768.0f;
        }
        return out;
    }
    const double step = (double) rate / 16000.0;
    const size_t out_n = (size_t) (n / step);
    out.resize(out_n);
    for (size_t i = 0; i < out_n; ++i) {
        const double pos = i * step;
        const size_t i0 = (size_t) pos;
        const size_t i1 = std::min(i0 + 1, n - 1);
        const double frac = pos - (double) i0;
        const double a = (double) pcm[i0] / 32768.0, b = (double) pcm[i1] / 32768.0;
        out[i] = (float) (a + (b - a) * frac);
    }
    return out;
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
    for (size_t i = 0; i < g_queue.size(); ++i) {
        if (g_queue[i]->priority > 0) {
            auto t = std::move(g_queue[i]);
            g_queue.erase(g_queue.begin() + (long) i);
            return t;
        }
    }
    auto t = std::move(g_queue.front());
    g_queue.pop_front();
    return t;
}

static void worker_loop(Engine* eng) {
    for (;;) {
        std::unique_ptr<PoolTask> task;
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            g_cv.wait(lock, [] { return !g_queue.empty() || g_stop; });
            if (g_queue.empty()) {
                if (g_stop) {
                    return;
                }
                continue;
            }
            task = pop_task();
        }
        g_cv.notify_all();
        const double queue_ms = now_ms() - task->enqueued_ms;
        if (g_timeout_ms > 0 && queue_ms > g_timeout_ms) {
            task->done(503, "queue timeout", queue_ms, 0.0, nullptr);
            continue;
        }
        const double t0 = now_ms();
        TokensPtr tokens;
        std::string text;
        if (task->want_tokens) {
            tokens.reset(paraformer_transcribe_tokens(eng->ctx, task->samples.data(), (int) task->samples.size()),
                         TokensDeleter{});
            if (tokens && tokens->text) {
                text = tokens->text;
            }
        } else {
            std::unique_ptr<char, decltype(&std::free)> t(
                paraformer_transcribe(eng->ctx, task->samples.data(), (int) task->samples.size()), &std::free);
            if (t) {
                text = t.get();
            }
        }
        const double infer_ms = now_ms() - t0;
        if (!tokens && text.empty()) {
            g_failed.fetch_add(1);
            task->done(500, "inference failed", queue_ms, infer_ms, nullptr);
            continue;
        }
        eng->requests.fetch_add(1);
        task->done(200, std::move(text), queue_ms, infer_ms, std::move(tokens));
    }
}

// Returns false when the queue is full (the caller maps that to 429 / an error event).
static bool submit(std::vector<float> samples, int priority, pool_done done, bool want_tokens = false) {
    auto task = std::make_unique<PoolTask>();
    task->samples = std::move(samples);
    task->done = std::move(done);
    task->priority = priority;
    task->want_tokens = want_tokens;
    task->enqueued_ms = now_ms();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_queue.size() >= g_queue_limit) {
            g_rejected.fetch_add(1);
            return false;
        }
        g_queue.push_back(std::move(task));
    }
    g_cv.notify_one();
    g_requests.fetch_add(1);
    return true;
}

struct SyncReply {
    int status = 200;
    std::string text;
    double queue_ms = 0;
    double infer_ms = 0;
    TokensPtr tokens;
};

// Synchronous helper for the REST endpoints.
static SyncReply transcribe_sync(const std::vector<float>& samples, bool want_tokens = false) {
    struct Result {
        int status;
        std::string text;
        double queue_ms;
        double infer_ms;
        TokensPtr tokens;
    };
    auto promise = std::make_shared<std::promise<Result>>();
    auto fut = promise->get_future();
    const bool ok = submit(
        samples, 1,
        [promise](int status, std::string text, double queue_ms, double infer_ms, TokensPtr tokens) {
            promise->set_value({status, std::move(text), queue_ms, infer_ms, std::move(tokens)});
        },
        want_tokens);
    if (!ok) {
        SyncReply r;
        r.status = 429;
        r.text = "server is busy, retry later";
        return r;
    }
    const Result v = fut.get();
    SyncReply r;
    r.status = v.status;
    r.text = v.text;
    r.queue_ms = v.queue_ms;
    r.infer_ms = v.infer_ms;
    r.tokens = v.tokens;
    return r;
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
struct WordSpan {
    std::string text;
    double start_ms = 0;
    double end_ms = 0;
};

static std::vector<WordSpan> build_words(const paraformer_tokens* t) {
    std::vector<WordSpan> out;
    if (!t || t->n_tokens <= 0) {
        return out;
    }
    WordSpan cur;
    bool open = false;
    for (int i = 0; i < t->n_tokens; ++i) {
        const char* p = t->tokens[i].text ? t->tokens[i].text : "";
        std::string s(p);
        const bool cont = s.size() >= 2 && s.compare(s.size() - 2, 2, "@@") == 0;
        if (!open) {
            cur = WordSpan();
            // Start at this word's first fire time: the gap to the previous word
            // is then (start_next - end_prev), which is what the pause test needs.
            cur.start_ms = (double) t->tokens[i].t_ms;
            open = true;
        }
        cur.text += cont ? s.substr(0, s.size() - 2) : s;
        cur.end_ms = (double) t->tokens[i].t_ms;
        if (!cont) {
            out.push_back(cur);
            open = false;
        }
    }
    if (open) {
        out.push_back(cur);
    }
    return out;
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

// ---- base64 ----

static bool base64_decode(const std::string& in, std::string& out) {
    static int8_t table[256];
    static bool init = false;
    if (!init) {
        std::fill(std::begin(table), std::end(table), (int8_t) -1);
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) {
            table[(uint8_t) a[i]] = (int8_t) i;
        }
        init = true;
    }
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

struct WsSession {
    std::mutex m;
    std::deque<std::string> outbox;
    std::vector<float> pcm;       // 16 kHz mono
    std::string language = "zh";
    std::string item_id;
    std::string partial_text;
    uint64_t gen = 0;             // bumped by commit/clear to invalidate in-flight work
    size_t partial_mark = 0;      // pcm size when the last partial was scheduled
    double last_partial_ms = 0;
    bool partial_inflight = false;
    int commits = 0;
};

static std::string next_event_id() {
    return "event_" + std::to_string(++g_event_seq);
}

static void ws_send_event(WsSession& s, const char* type, json body) {
    body["type"] = type;
    body["event_id"] = next_event_id();
    std::lock_guard<std::mutex> lock(s.m);
    s.outbox.push_back(body.dump());
}

static void ws_send_error(WsSession& s, const std::string& message, const std::string& type,
                          const std::string& param = "", const std::string& code = "") {
    json e = {{"type", "error"}, {"event_id", next_event_id()}, {"error", error_body(message, type, param, code)["error"]}};
    std::lock_guard<std::mutex> lock(s.m);
    s.outbox.push_back(e.dump());
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
    const bool ok = submit(std::move(snapshot), 0, [s, gen](int status, std::string text, double, double, TokensPtr) {
        {
            std::lock_guard<std::mutex> lock(s->m);
            s->partial_inflight = false;
        }
        if (status != 200 || gen != s->gen) {
            return; // stale or failed: a commit or clear superseded it
        }
        std::string delta;
        {
            std::lock_guard<std::mutex> lock(s->m);
            size_t i = 0;
            while (i < text.size() && i < s->partial_text.size() && text[i] == s->partial_text[i]) {
                ++i;
            }
            delta = text.substr(i);
            s->partial_text = text;
        }
        if (!delta.empty()) {
            ws_send_event(*s, "conversation.item.input_audio_transcription.delta",
                          {{"item_id", s->item_id}, {"content_index", 0}, {"delta", delta}});
        }
    });
    if (!ok) {
        std::lock_guard<std::mutex> lock(s->m);
        s->partial_inflight = false;
    }
}

static void ws_handle_text(const std::shared_ptr<WsSession>& s, const std::string& msg) {
    json j;
    try {
        j = json::parse(msg);
    } catch (const std::exception&) {
        ws_send_error(*s, "client event is not valid JSON", "invalid_request_error");
        return;
    }
    const std::string type = j.value("type", "");
    if (type == "session.update") {
        const json& sess = j.contains("session") ? j["session"] : json::object();
        if (sess.contains("input_audio_transcription") && sess["input_audio_transcription"].contains("language")) {
            s->language = sess["input_audio_transcription"]["language"].get<std::string>();
        }
        ws_send_event(*s, "session.updated", {{"session", session_object(*s)}});
        return;
    }
    if (type == "input_audio_buffer.append") {
        std::string raw;
        if (!j.contains("audio") || !base64_decode(j["audio"].get<std::string>(), raw)) {
            ws_send_error(*s, "append needs a base64 'audio' field", "invalid_request_error", "audio");
            return;
        }
        if (raw.size() % 2 != 0) {
            ws_send_error(*s, "audio must be 16-bit PCM", "invalid_request_error", "audio");
            return;
        }
        const size_t n = raw.size() / 2;
        std::vector<float> f = resample_to_16k(reinterpret_cast<const int16_t*>(raw.data()), n, 16000);
        {
            std::lock_guard<std::mutex> lock(s->m);
            if (s->pcm.size() + f.size() > (size_t) 16000 * 30) {
                ws_send_error(*s, "segment exceeds 30 s, commit or clear first", "invalid_request_error");
                return;
            }
            s->pcm.insert(s->pcm.end(), f.begin(), f.end());
        }
        maybe_schedule_partial(s);
        return;
    }
    if (type == "input_audio_buffer.commit") {
        std::vector<float> snapshot;
        uint64_t gen;
        {
            std::lock_guard<std::mutex> lock(s->m);
            if (s->pcm.empty()) {
                ws_send_error(*s, "buffer is empty", "invalid_request_error");
                return;
            }
            snapshot.swap(s->pcm);
            s->pcm.clear();
            s->partial_text.clear();
            s->partial_mark = 0;
            s->item_id = "item_" + std::to_string(++s->commits);
            gen = ++s->gen;
        }
        ws_send_event(*s, "input_audio_buffer.committed", {{"item_id", s->item_id}, {"previous_item_id", nullptr}});
        const std::string item = s->item_id;
        const bool ok = submit(std::move(snapshot), 1, [s, gen, item](int status, std::string text, double queue_ms,
                                                                     double infer_ms, TokensPtr) {
            if (gen != s->gen) {
                return; // cleared or superseded
            }
            if (status != 200) {
                ws_send_error(*s, text, status == 503 ? "server_error" : "server_error");
                return;
            }
            json body = {{"item_id", item},
                         {"content_index", 0},
                         {"transcript", text},
                         {"usage", {{"type", "duration"}, {"seconds", (double) 0}}}};
            ws_send_event(*s, "conversation.item.input_audio_transcription.completed", body);
            json meta = {{"item_id", item}, {"queue_ms", queue_ms}, {"infer_ms", infer_ms}};
            ws_send_event(*s, "x.pf.metrics", meta);
        });
        if (!ok) {
            ws_send_error(*s, "server is busy, retry later", "rate_limit_error");
        }
        return;
    }
    if (type == "input_audio_buffer.clear") {
        {
            std::lock_guard<std::mutex> lock(s->m);
            s->pcm.clear();
            s->partial_text.clear();
            s->partial_mark = 0;
            ++s->gen;
        }
        ws_send_event(*s, "input_audio_buffer.cleared", json::object());
        return;
    }
    ws_send_error(*s, "unsupported event type '" + type + "'", "invalid_request_error", "type", "unsupported_event");
}

// ---- WS loop ----

static void ws_serve(httplib::ws::WebSocket& ws) {
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
            pending.swap(s->outbox);
        }
        for (const auto& e : pending) {
            if (!ws.send(e)) {
                return;
            }
        }
    }
    // Invalidate anything still in flight so its callback does not touch a
    // session the connection no longer reads from.
    std::lock_guard<std::mutex> lock(s->m);
    ++s->gen;
}

// ---- HTTP ----

static void handle_transcription(const httplib::Request& req, httplib::Response& res) {
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
    const std::string punct = req.form.has_field("punctuate") ? req.form.get_field("punctuate") : "false";
    const bool want_punct = punct == "true" || punct == "1";
    const bool want_tokens = fmt == "verbose_json" || fmt == "srt" || want_punct;
    const SyncReply r = transcribe_sync(samples, want_tokens);
    res.set_header("X-Queue-Ms", std::to_string((int) std::lround(r.queue_ms)));
    res.set_header("X-Infer-Ms", std::to_string((int) std::lround(r.infer_ms)));
    if (r.status != 200) {
        reply_error(res, r.status, r.text, r.status == 429 ? "rate_limit_error" : "server_error");
        return;
    }
    const std::vector<WordSpan> words = build_words(r.tokens.get());
    const bool zh = any_multibyte(r.text);
    const std::string detected =
        words.empty() ? (any_multibyte(r.text) ? std::string("zh") : std::string("en")) : utterance_language(words);
    const std::string text = want_punct && !words.empty() ? punctuate_words(words, zh, g_punct_gap_ms) : r.text;
    res.set_header("X-Language", detected);
    if (fmt == "text") {
        res.set_content(text, "text/plain; charset=utf-8");
        return;
    }
    if (fmt == "verbose_json") {
        const double duration = (double) samples.size() / 16000.0;
        json jw = json::array();
        for (const auto& w : words) {
            jw.push_back({{"word", w.text}, {"start", w.start_ms / 1000.0}, {"end", w.end_ms / 1000.0}});
        }
        json j = {{"task", "transcribe"},
                  {"language", detected},
                  {"duration", duration},
                  {"text", text},
                  {"words", jw},
                  {"segments", segments_json(words, zh, g_seg_gap_ms, text)}};
        res.set_content(j.dump(2), "application/json");
        return;
    }
    if (fmt == "srt") {
        const json segs = segments_json(words, zh, g_seg_gap_ms, text);
        std::string body;
        for (size_t i = 0; i < segs.size(); ++i) {
            body += std::to_string(i + 1) + "\n" + srt_time(segs[i]["start"].get<double>()) + " --> " +
                    srt_time(segs[i]["end"].get<double>()) + "\n" + segs[i]["text"].get<std::string>() + "\n\n";
        }
        res.set_content(body.empty() ? text + "\n" : body, "text/plain; charset=utf-8");
        return;
    }
    if (fmt != "json") {
        reply_error(res, 400, "unsupported response_format '" + fmt + "'", "invalid_request_error", "response_format");
        return;
    }
    res.set_content(json{{"text", text}}.dump(), "application/json");
}

static void handle_segment(const httplib::Request& req, httplib::Response& res) {
    if (req.body.size() % sizeof(float) != 0 || req.body.empty()) {
        reply_error(res, 400, "body must be non-empty float32 PCM", "invalid_request_error", "body");
        return;
    }
    std::vector<float> samples(req.body.size() / sizeof(float));
    std::memcpy(samples.data(), req.body.data(), req.body.size());
    const std::string fmt = req.has_param("format") ? req.get_param_value("format") : "json";
    const SyncReply r = transcribe_sync(samples);
    res.set_header("X-Queue-Ms", std::to_string((int) std::lround(r.queue_ms)));
    res.set_header("X-Infer-Ms", std::to_string((int) std::lround(r.infer_ms)));
    if (r.status != 200) {
        reply_error(res, r.status, r.text, r.status == 429 ? "rate_limit_error" : "server_error");
        return;
    }
    if (fmt == "text") {
        res.set_content(r.text, "text/plain; charset=utf-8");
        return;
    }
    res.set_content(json{{"text", r.text}, {"duration", (double) samples.size() / 16000.0}}.dump(), "application/json");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf [--device HTP0:0] [--port N] [--engines N] [--threads N] "
                     "[--bind ADDR] [--queue N] [--timeout-ms N] [--model-id ID] [--warmup WAV] "
                     "[--partial-ms N] [--partial-interval-ms N] [--no-partials] [--seg-gap-ms N] [--punct-gap-ms N] "
                     "[--web-dir DIR]\n",
                     argv[0]);
        return 1;
    }
    const char* model = argv[1];
    const char* device = "CPU";
    const char* bind_addr = "0.0.0.0";
    const char* warmup = nullptr;
    const char* web_dir = "web";
    int port = 8080;
    int n_engines = 2;
    int threads = 2;
    bool device_set = false;
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
        } else if (std::strcmp(argv[i], "--queue") == 0 && i + 1 < argc) {
            g_queue_limit = (size_t) std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc) {
            g_timeout_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--model-id") == 0 && i + 1 < argc) {
            g_model_id = argv[++i];
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = argv[++i];
        } else if (std::strcmp(argv[i], "--web-dir") == 0 && i + 1 < argc) {
            web_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--partial-ms") == 0 && i + 1 < argc) {
            g_partial_min_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--partial-interval-ms") == 0 && i + 1 < argc) {
            g_partial_interval = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-partials") == 0) {
            g_partials = false;
        } else if (std::strcmp(argv[i], "--seg-gap-ms") == 0 && i + 1 < argc) {
            g_seg_gap_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--punct-gap-ms") == 0 && i + 1 < argc) {
            g_punct_gap_ms = std::atof(argv[++i]);
        }
    }
    if (device_set && std::strcmp(device, "CPU") != 0) {
        setenv("PARAFORMER_DEVICE", device, 1);
    }
    ggml_backend_load_all();

    paraformer_weights* w = paraformer_weights_load(model, 1);
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
            std::fprintf(stderr, "pf-api: cannot read warmup %s, using 1.2 s of silence\n", warmup);
        }
    }
    if (warm.empty()) {
        warm.assign((size_t) 16000 * 12 / 10, 0.0f);
    }

    for (int i = 0; i < n_engines; ++i) {
        paraformer_context_params params = paraformer_context_default_params();
        params.n_threads = threads;
        params.use_gpu = (i == 0) && device_set && std::strcmp(device, "CPU") != 0;
        params.verbosity = 0;
        auto* eng = new Engine();
        eng->ctx = paraformer_init_shared(w, params);
        if (!eng->ctx) {
            std::fprintf(stderr, "pf-api: engine %d init failed\n", i);
            return 3;
        }
        eng->name = params.use_gpu ? device : "CPU";
        const double t0 = now_ms();
        paraformer_warmup(eng->ctx, warm.data(), (int) warm.size());
        eng->worker = std::thread(worker_loop, eng);
        g_engines.push_back(eng);
        std::fprintf(stderr, "pf-api: engine %d (%s) ready in %.0f ms\n", i, eng->name.c_str(), now_ms() - t0);
    }

    httplib::Server srv;
    srv.set_read_timeout(120, 0);
    srv.set_write_timeout(120, 0);
    srv.set_payload_max_length(64u * 1024u * 1024u);

    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}, {"engines", g_engines.size()}, {"queue", g_queue.size()}}.dump(),
                        "application/json");
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
        json j = {{"requests", g_requests.load()},
                  {"rejected", g_rejected.load()},
                  {"failed", g_failed.load()},
                  {"queue", g_queue.size()},
                  {"engines", g_engines.size()}};
        res.set_content(j.dump(), "application/json");
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
    if (!srv.listen(bind_addr, port)) {
        std::fprintf(stderr, "pf-api: cannot listen on %s:%d\n", bind_addr, port);
        return 4;
    }
    return 0;
}
