// OpenAI-compatible transcription endpoint for the Paraformer engine.
//
// The cabin front end already cuts audio at the VAD endpoint and sends one
// segment, so a whole-utterance HTTP endpoint maps directly onto the engine:
// POST /v1/audio/transcriptions with a WAV file and get the final text.
// /asr/segment exists for the device client, which sends float32 PCM and avoids
// WAV parsing.
//
// Deliberate limits: one segment per request (no session state), no speaker
// diarisation, no word timestamps yet. See docs/backend/snapdragon/
// asr-server-api-design.md for the surface this is meant to grow into.

#include "paraformer.h"

#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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

struct Reply {
    int status = 200;
    std::string text; // transcript on success, message on failure
    double queue_ms = 0;
    double infer_ms = 0;
};

struct Task {
    std::vector<float> samples;
    std::promise<Reply> result;
    double enqueued_ms = 0;
};

static std::deque<std::unique_ptr<Task>> g_queue;
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

static double now_ms() {
    return std::chrono::duration<double, std::milli>(pf_clock::now().time_since_epoch()).count();
}

// OpenAI error envelope: every 4xx/5xx body is {"error": {...}}.
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

// 16-bit mono PCM WAV at 16 kHz. Anything else is rejected instead of resampled:
// the product front end already produces this format, and silent resampling
// would hide a client bug.
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

static std::string srt_time(double seconds) {
    int ms = (int) std::lround(seconds * 1000.0);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d", ms / 3600000, (ms / 60000) % 60, (ms / 1000) % 60, ms % 1000);
    return buf;
}

static void worker_loop(Engine* eng) {
    for (;;) {
        std::unique_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            g_cv.wait(lock, [] { return !g_queue.empty() || g_stop; });
            if (g_queue.empty()) {
                if (g_stop) {
                    return;
                }
                continue;
            }
            task = std::move(g_queue.front());
            g_queue.pop_front();
        }
        g_cv.notify_all();
        const double queue_ms = now_ms() - task->enqueued_ms;
        if (g_timeout_ms > 0 && queue_ms > g_timeout_ms) {
            Reply r;
            r.status = 503;
            r.text = "queue timeout";
            r.queue_ms = queue_ms;
            task->result.set_value(std::move(r));
            continue;
        }
        const double t0 = now_ms();
        std::unique_ptr<char, decltype(&std::free)> text(
            paraformer_transcribe(eng->ctx, task->samples.data(), (int) task->samples.size()), &std::free);
        const double infer_ms = now_ms() - t0;
        Reply r;
        r.queue_ms = queue_ms;
        r.infer_ms = infer_ms;
        if (!text) {
            r.status = 500;
            r.text = "inference failed";
            task->result.set_value(std::move(r));
            continue;
        }
        eng->requests.fetch_add(1);
        r.status = 200;
        r.text = std::string(text.get());
        task->result.set_value(std::move(r));
    }
}

// Submit one segment and wait. Returns the HTTP status and the text or error.
static Reply transcribe(const std::vector<float>& samples) {
    auto task = std::make_unique<Task>();
    task->samples = samples;
    task->enqueued_ms = now_ms();
    auto fut = task->result.get_future();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_queue.size() >= g_queue_limit) {
            g_rejected.fetch_add(1);
            Reply r;
            r.status = 429;
            r.text = "server is busy, retry later";
            return r;
        }
        g_queue.push_back(std::move(task));
    }
    g_cv.notify_one();
    g_requests.fetch_add(1);
    return fut.get();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf [--device HTP0:0] [--port N] [--engines N] [--threads N] "
                     "[--bind ADDR] [--queue N] [--timeout-ms N] [--model-id ID] [--warmup F32]\n",
                     argv[0]);
        return 1;
    }
    const char* model = argv[1];
    const char* device = "CPU";
    const char* bind_addr = "0.0.0.0";
    const char* warmup = nullptr;
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
                if (parse_wav(std::string(bytes.data(), bytes.size()), warm, err)) {
                    // ok
                }
            }
            std::fclose(f);
        }
        if (warm.empty()) {
            std::fprintf(stderr, "pf-api: cannot read warmup %s (%s), warming with 1.2 s of silence\n", warmup,
                         err.c_str());
            warm.assign((size_t) 16000 * 12 / 10, 0.0f);
        }
    } else {
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
        json j = {{"status", "ok"}, {"engines", g_engines.size()}, {"queue", g_queue.size()}};
        res.set_content(j.dump(), "application/json");
    });

    srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json j = {{"object", "list"},
                  {"data", {{{"id", g_model_id}, {"object", "model"}, {"created", 0}, {"owned_by", "local"}}}}};
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

    // POST /v1/audio/transcriptions: OpenAI-shaped multipart request.
    srv.Post("/v1/audio/transcriptions", [](const httplib::Request& req, httplib::Response& res) {
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
        const Reply r = transcribe(samples);
        res.set_header("X-Queue-Ms", std::to_string((int) std::lround(r.queue_ms)));
        res.set_header("X-Infer-Ms", std::to_string((int) std::lround(r.infer_ms)));
        if (r.status != 200) {
            reply_error(res, r.status, r.text, r.status == 429 ? "rate_limit_error" : "server_error");
            return;
        }
        const std::string& text = r.text;
        if (fmt == "text") {
            res.set_content(text, "text/plain; charset=utf-8");
            return;
        }
        if (fmt == "verbose_json") {
            const double duration = (double) samples.size() / 16000.0;
            json j = {{"task", "transcribe"},
                      {"language", req.form.has_field("language") ? req.form.get_field("language") : std::string("zh")},
                      {"duration", duration},
                      {"text", text},
                      {"segments", {{{"id", 0}, {"start", 0.0}, {"end", duration}, {"text", text}}}}};
            res.set_content(j.dump(2), "application/json");
            return;
        }
        if (fmt == "srt") {
            const double duration = (double) samples.size() / 16000.0;
            res.set_content("1\n" + srt_time(0.0) + " --> " + srt_time(duration) + "\n" + text + "\n",
                            "text/plain; charset=utf-8");
            return;
        }
        if (fmt != "json") {
            reply_error(res, 400, "unsupported response_format '" + fmt + "'", "invalid_request_error", "response_format");
            return;
        }
        json j = {{"text", text}};
        res.set_content(j.dump(), "application/json");
    });

    // POST /asr/segment: raw float32 little-endian PCM at 16 kHz, one VAD segment.
    // Query: format=json|text, language=zh|en.
    srv.Post("/asr/segment", [](const httplib::Request& req, httplib::Response& res) {
        if (req.body.size() % sizeof(float) != 0 || req.body.empty()) {
            reply_error(res, 400, "body must be non-empty float32 PCM", "invalid_request_error", "body");
            return;
        }
        std::vector<float> samples(req.body.size() / sizeof(float));
        std::memcpy(samples.data(), req.body.data(), req.body.size());
        const std::string fmt = req.has_param("format") ? req.get_param_value("format") : "json";
        const Reply r = transcribe(samples);
        res.set_header("X-Queue-Ms", std::to_string((int) std::lround(r.queue_ms)));
        res.set_header("X-Infer-Ms", std::to_string((int) std::lround(r.infer_ms)));
        if (r.status != 200) {
            reply_error(res, r.status, r.text, r.status == 429 ? "rate_limit_error" : "server_error");
            return;
        }
        const std::string& text = r.text;
        if (fmt == "text") {
            res.set_content(text, "text/plain; charset=utf-8");
            return;
        }
        json j = {{"text", text}, {"duration", (double) samples.size() / 16000.0}};
        res.set_content(j.dump(), "application/json");
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

    std::fprintf(stderr, "pf-api: model '%s', %zu engines, %zu queue, listening on %s:%d\n", g_model_id.c_str(),
                 g_engines.size(), g_queue_limit, bind_addr, port);
    if (!srv.listen(bind_addr, port)) {
        std::fprintf(stderr, "pf-api: cannot listen on %s:%d\n", bind_addr, port);
        return 4;
    }
    return 0;
}
