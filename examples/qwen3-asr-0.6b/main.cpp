#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-cpp.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <nlohmann/json.hpp>

#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

struct request {
    int arrival_ms;
    std::string path;
};

static void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static int parse_int(std::string_view text, int min, int max) {
    int value;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    require(result.ec == std::errc{} && result.ptr == text.data() + text.size() && value >= min && value <= max,
            "integer out of range: " + std::string(text));
    return value;
}

static std::vector<request> read_requests(const std::string & path) {
    std::ifstream input(path);
    require(input.is_open(), "could not open trace: " + path);
    std::vector<request> result;
    std::string line;
    while (std::getline(input, line)) {
        const size_t tab = line.find('\t');
        require(tab != std::string::npos && tab + 1 < line.size() && result.size() < 128, "invalid request trace");
        const int arrival = parse_int(std::string_view(line).substr(0, tab), 0, 60000);
        require(result.empty() || arrival >= result.back().arrival_ms, "arrival times must be ordered");
        result.push_back({arrival, line.substr(tab + 1)});
    }
    require(input.eof() && !result.empty(), "could not read a nonempty request trace");
    return result;
}

static void emit(const json & event) {
    std::cout << event.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
}

struct slot {
    int id = -1;
    llama_pos pos = 0;
    llama_token pending = LLAMA_TOKEN_NULL;
    std::vector<llama_token> tokens;
    double admitted_ms = 0;
    double prefill_ms = 0;
    double first_token_ms = 0;
    double audio_ms = 0;
};

static void prefill(mtmd_context * audio, llama_context * decoder, const std::string & prompt,
                    const std::string & path, llama_seq_id seq, int max_tokens, slot & state) {
    require(std::filesystem::file_size(path) <= 4 * 1024 * 1024, "audio file too large: " + path);
    const auto media = mtmd_helper_bitmap_init_from_file(audio, path.c_str(), false, mtmd_helper_init_opt_default());
    mtmd::bitmap bitmap(media.bitmap);
    require(bitmap.ptr && mtmd_bitmap_is_audio(bitmap.ptr.get()), "expected audio: " + path);
    const int rate = mtmd_get_audio_sample_rate(audio);
    require(rate > 0 && bitmap.n_bytes() > 0 && bitmap.n_bytes() <= size_t(rate) * 30 * sizeof(float),
            "expected 0-30 s of audio");
    state.audio_ms = 1000.0 * bitmap.n_bytes() / (rate * sizeof(float));
    const mtmd_input_text text = {prompt.data(), prompt.size(), true, true};
    const mtmd_bitmap * bitmaps[] = {bitmap.ptr.get()};
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    require(mtmd_tokenize(audio, chunks.ptr.get(), &text, bitmaps, 1) == 0 && chunks.size() > 0, "audio tokenization failed");
    const auto n_pos = mtmd_helper_get_n_pos(chunks.ptr.get());
    require(n_pos > 0 && size_t(n_pos) + max_tokens <= llama_n_ctx_seq(decoder), "prompt and output exceed slot context");
    require(mtmd_helper_eval_chunks(audio, decoder, chunks.ptr.get(), 0, seq, llama_n_batch(decoder), true, &state.pos) == 0,
            "audio prefill failed: " + path);
}

static void run(mtmd_context * audio, llama_context * decoder, const std::string & prompt,
                const std::vector<request> & requests, int capacity, int max_tokens, bool report) {
    std::array<slot, 2> slots;
    llama_batch_ext_ptr batch(llama_batch_ext_init(decoder));
    llama_sampler_ptr sampler(llama_sampler_init_greedy());
    require(batch && sampler, "could not allocate decode state");
    const auto * vocab = llama_model_get_vocab(llama_get_model(decoder));
    const auto start = clock_type::now();
    const auto elapsed = [&] {
        return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    };
    size_t next = 0;
    size_t done = 0;
    size_t decode_calls = 0;
    size_t paired_calls = 0;
    double decode_ms = 0;

    auto sample = [&](int seq, int row) {
        auto & state = slots[seq];
        const llama_token token = llama_sampler_sample(sampler.get(), decoder, row);
        require(token != LLAMA_TOKEN_NULL, "sampling failed");
        if (state.tokens.empty()) {
            state.first_token_ms = elapsed();
        }
        const bool eog = llama_vocab_is_eog(vocab, token);
        if (!eog) {
            state.tokens.push_back(token);
        }
        if (!eog && state.tokens.size() < size_t(max_tokens)) {
            state.pending = token;
            return;
        }

        require(llama_memory_seq_rm(llama_get_memory(decoder), seq, -1, -1), "could not release sequence KV");
        if (report) {
            const auto raw = common_detokenize(decoder, state.tokens, true);
            constexpr std::string_view text_marker = "<asr_text>";
            const auto marker = raw.find(text_marker);
            const auto text = marker == std::string::npos ? raw : raw.substr(marker + text_marker.size());
            emit({{"event", "finish"}, {"id", state.id}, {"slot", seq}, {"finish_ms", elapsed()},
                  {"arrival_ms", requests[state.id].arrival_ms}, {"admitted_ms", state.admitted_ms},
                  {"prefill_ms", state.prefill_ms}, {"first_token_ms", state.first_token_ms},
                  {"audio_ms", state.audio_ms}, {"stop", eog ? "eog" : "limit"},
                  {"tokens", state.tokens}, {"raw", raw}, {"text", text}});
        }
        state = {};
        ++done;
    };

    while (done < requests.size()) {
        // Prefill one arrival per iteration, then advance all decode-ready slots.
        for (int seq = 0; seq < capacity && next < requests.size() && requests[next].arrival_ms <= elapsed(); ++seq) {
            auto & state = slots[seq];
            if (state.id != -1) continue;
            const size_t id = next++;
            state.id = int(id);
            state.admitted_ms = elapsed();
            if (report) {
                emit({{"event", "admit"}, {"id", id}, {"slot", seq}, {"at_ms", state.admitted_ms}});
            }
            prefill(audio, decoder, prompt, requests[id].path, seq, max_tokens, state);
            llama_synchronize(decoder);
            state.prefill_ms = elapsed() - state.admitted_ms;
            sample(seq, -1);
            break;
        }

        llama_batch_ext_clear(batch.get());
        std::array<int32_t, 2> rows = {-1, -1};
        std::vector<int> ids;
        for (int seq = 0; seq < capacity; ++seq) {
            auto & state = slots[seq];
            if (state.id == -1) continue;
            const int32_t row = llama_batch_ext_add_token(batch.get(), seq, state.pending);
            require(row >= 0 && llama_batch_ext_set_pos(batch.get(), row, &state.pos) &&
                    llama_batch_ext_set_output_logits(batch.get(), row, true), "could not build decode batch");
            ++state.pos;
            rows[seq] = row;
            ids.push_back(state.id);
        }
        if (ids.empty()) {
            if (next < requests.size()) {
                std::this_thread::sleep_until(start + std::chrono::milliseconds(requests[next].arrival_ms));
            }
            continue;
        }
        const double begin = elapsed();
        require(llama_process(decoder, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decoder batch failed");
        llama_synchronize(decoder);
        const double duration = elapsed() - begin;
        decode_ms += duration;
        ++decode_calls;
        paired_calls += ids.size() == 2;
        if (report && ids.size() == 2) {
            emit({{"event", "paired_decode"}, {"ids", ids}, {"at_ms", begin}, {"decode_ms", duration}});
        }
        for (int seq = 0; seq < capacity; ++seq) {
            if (rows[seq] >= 0) sample(seq, rows[seq]);
        }
    }
    if (report) {
        emit({{"event", "summary"}, {"capacity", capacity}, {"makespan_ms", elapsed()},
              {"decode_calls", decode_calls}, {"decode_ms", decode_ms},
              {"paired_calls", paired_calls}, {"requests", requests.size()}});
    }
}

int main(int argc, char ** argv) {
    if (argc < 6 || argc > 7) {
        std::cerr << "usage: llama-qwen3-asr-batch MODEL MMPROJ CPU|HTP0:0 SLOTS TRACE.tsv [MAX_TOKENS]\n";
        return 2;
    }
    try {
        const int capacity = parse_int(argv[4], 1, 2);
        const int max_tokens = argc == 7 ? parse_int(argv[6], 1, 256) : 96;
        const auto requests = read_requests(argv[5]);
        const std::string device_name = argv[3];
        require(device_name == "CPU" || device_name.rfind("HTP", 0) == 0, "expected CPU or an HTP device name");
        ggml_backend_load_all();
        const bool cpu = device_name == "CPU";
        ggml_backend_dev_t devices[] = {cpu ? nullptr : ggml_backend_dev_by_name(device_name.c_str()), nullptr};
        require(cpu || devices[0] != nullptr, "requested HTP device is unavailable");
        auto model_params = llama_model_default_params();
        model_params.devices = devices;
        model_params.n_gpu_layers = cpu ? 0 : 999;
        llama_model_ptr model(llama_model_load_from_file(argv[1], model_params));
        require(model != nullptr, "could not load model");
        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        ctx_params.n_seq_max = 2;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;
        ctx_params.n_threads = 4;
        ctx_params.n_threads_batch = 4;
        ctx_params.flash_attn_type = cpu ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context_ptr decoder(llama_init_from_model(model.get(), ctx_params));
        require(decoder != nullptr, "could not initialize decoder");
        auto audio_params = mtmd_context_params_default();
        audio_params.use_gpu = !cpu;
        audio_params.device = devices[0];
        audio_params.n_threads = 4;
        audio_params.flash_attn_type = ctx_params.flash_attn_type;
        audio_params.warmup = false;
        mtmd::context_ptr audio(mtmd_init_from_file(argv[2], model.get(), audio_params));
        require(audio && mtmd_support_audio(audio.get()), "could not load audio projector");

        const auto templates = common_chat_templates_init(model.get(), "");
        const auto preset = common_chat_get_asr_prompt(templates.get());
        require(!preset.needs_audio_length, "this example expects the Qwen3-ASR template");
        common_chat_templates_inputs inputs;
        if (!preset.system.empty()) {
            common_chat_msg system;
            system.role = "system";
            system.content = preset.system;
            inputs.messages.push_back(system);
        }
        common_chat_msg user;
        user.role = "user";
        const std::string marker = mtmd_default_marker();
        user.content = preset.media_first ? marker + preset.user : preset.user + marker;
        inputs.messages.push_back(user);
        inputs.add_generation_prompt = true;
        const auto prompt = common_chat_templates_apply(templates.get(), inputs).prompt.str();

        run(audio.get(), decoder.get(), prompt, {{0, requests.front().path}}, 1, max_tokens, false);
        run(audio.get(), decoder.get(), prompt, requests, capacity, max_tokens, true);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
