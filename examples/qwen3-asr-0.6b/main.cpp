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
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

static double millis(clock_type::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

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
    mtmd::input_chunks_ptr chunks;
    size_t chunk_index = 0;
    bool audio_encoded = false;
    double admitted_ms = 0;
    double prefill_ms = 0;
    double prepare_ms = 0;
    double encode_ms = 0;
    double decoder_prefill_ms = 0;
    double first_token_ms = 0;
    double audio_ms = 0;
};

static void prepare(mtmd_context * audio, llama_context * decoder, const std::string & prompt,
                    const std::string & path, int max_tokens, slot & state) {
    const auto started = clock_type::now();
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
    state.chunks.reset(mtmd_input_chunks_init());
    require(state.chunks && mtmd_tokenize(audio, state.chunks.get(), &text, bitmaps, 1) == 0 &&
            mtmd_input_chunks_size(state.chunks.get()) > 0, "audio tokenization failed");
    const auto n_pos = mtmd_helper_get_n_pos(state.chunks.get());
    require(n_pos > 0 && size_t(n_pos) + max_tokens <= llama_n_ctx_seq(decoder), "prompt and output exceed slot context");
    state.prepare_ms = millis(clock_type::now() - started);
}

static void prefill_step(mtmd_context * audio, llama_context * decoder, llama_seq_id seq, slot & state) {
    const auto started = clock_type::now();
    const auto * chunk = mtmd_input_chunks_get(state.chunks.get(), state.chunk_index);
    const auto type = mtmd_input_chunk_get_type(chunk);
    require(type == MTMD_INPUT_CHUNK_TYPE_TEXT || type == MTMD_INPUT_CHUNK_TYPE_AUDIO, "unsupported prompt chunk");
    if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO && !state.audio_encoded) {
        require(mtmd_encode_chunk(audio, chunk) == 0, "audio encode failed");
        state.audio_encoded = true;
        state.encode_ms += millis(clock_type::now() - started);
        return;
    }

    const bool last = state.chunk_index + 1 == mtmd_input_chunks_size(state.chunks.get());
    const auto n_batch = llama_n_batch(decoder);
    if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        require(!last, "expected a text suffix after audio");
        require(mtmd_helper_decode_image_chunk(audio, decoder, chunk, mtmd_get_output_embd(audio), state.pos,
                    seq, n_batch, &state.pos, nullptr, nullptr) == 0, "audio decoder prefill failed");
        state.audio_encoded = false;
    } else {
        require(mtmd_helper_eval_chunk_single(audio, decoder, chunk, state.pos, seq, n_batch, last, &state.pos) == 0,
                "text prefill failed");
    }
    llama_synchronize(decoder);
    state.decoder_prefill_ms += millis(clock_type::now() - started);
    ++state.chunk_index;
}

static void prefill_packed(mtmd_context * audio, llama_context * decoder, llama_batch_ext * batch,
                            llama_seq_id seq, slot & state) {
    require(mtmd_helper_get_n_tokens(state.chunks.get()) <= llama_n_batch(decoder), "packed prompt exceeds the batch size");
    llama_batch_ext_clear(batch);
    const bool mrope = mtmd_decode_use_mrope(audio);
    const size_t embd_size = llama_model_n_embd_inp(llama_get_model(decoder));
    const double start_ms = state.encode_ms;
    const auto started = clock_type::now();
    int32_t row = -1;
    const size_t n_chunks = mtmd_input_chunks_size(state.chunks.get());
    for (size_t i = 0; i < n_chunks; ++i) {
        const auto * chunk = mtmd_input_chunks_get(state.chunks.get(), i);
        const auto type = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const auto * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            for (size_t j = 0; j < n_tokens; ++j) {
                row = llama_batch_ext_add_token(batch, seq, tokens[j]);
                require(row >= 0 && llama_batch_ext_set_pos(batch, row, &state.pos), "could not add prompt token");
                ++state.pos;
            }
        } else {
            require(type == MTMD_INPUT_CHUNK_TYPE_AUDIO && !mtmd_decode_use_non_causal(audio, chunk), "packed prefill needs causal audio");
            const auto encode_started = clock_type::now();
            require(mtmd_encode_chunk(audio, chunk) == 0, "audio encode failed");
            state.encode_ms += millis(clock_type::now() - encode_started);
            const auto * embd = mtmd_get_output_embd(audio);
            const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            require(embd && mtmd_input_chunk_get_n_pos(chunk) == llama_pos(n_tokens), "packed prefill needs one position per audio row");
            for (size_t j = 0; j < n_tokens; ++j) {
                row = llama_batch_ext_add_embd(batch, seq, {embd + j * embd_size, 1, embd_size});
                const std::array<llama_pos, 4> pos = {state.pos, state.pos, state.pos, state.pos};
                require(row >= 0 && llama_batch_ext_set_pos(batch, row, mrope ? pos.data() : &state.pos), "could not add audio embedding");
                ++state.pos;
            }
        }
    }
    require(row >= 0 && llama_batch_ext_set_output_logits(batch, row, true) &&
            llama_process(decoder, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "packed decoder prefill failed");
    llama_synchronize(decoder);
    state.decoder_prefill_ms = millis(clock_type::now() - started) - (state.encode_ms - start_ms);
    state.chunk_index = n_chunks;
}

static void run(mtmd_context * audio, llama_context * decoder, const std::string & prompt,
                const std::vector<request> & requests, int capacity, int max_tokens, std::string_view packing,
                std::string_view schedule, double budget_ms, bool report) {
    std::array<slot, 2> slots;
    llama_batch_ext_ptr batch(llama_batch_ext_init(decoder));
    llama_sampler_ptr sampler(llama_sampler_init_greedy());
    require(batch && sampler, "could not allocate decode state");
    const auto * vocab = llama_model_get_vocab(llama_get_model(decoder));
    const auto start = clock_type::now();
    const auto elapsed = [&] { return millis(clock_type::now() - start); };
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
                  {"prepare_ms", state.prepare_ms}, {"encode_ms", state.encode_ms},
                  {"decoder_prefill_ms", state.decoder_prefill_ms},
                  {"audio_ms", state.audio_ms}, {"stop", eog ? "eog" : "limit"},
                  {"tokens", state.tokens}, {"raw", raw}, {"text", text}});
        }
        state = {};
        ++done;
    };

    // The projector has one output buffer, so only one slot can prefill at a time.
    int prefilling = -1;

    // Batch 2: the scheduling policy decides who owns the gap between two decode
    // steps. "blocking" is the original behaviour and is kept bit-for-bit so the
    // A/B has a real control; the others move the decode step in front of the
    // prefill work and bound how much prefill one round may take.
    const bool decode_first = schedule != "blocking";
    const bool chunked = schedule == "interleaved";
    const bool idle_only = schedule == "idle";

    auto any_pending = [&]() {
        for (auto & state : slots) {
            if (state.pending != LLAMA_TOKEN_NULL) return true;
        }
        return false;
    };

    auto step_decode = [&]() -> bool {
        llama_batch_ext_clear(batch.get());
        std::array<int32_t, 2> rows = {-1, -1};
        std::vector<int> ids;
        for (int seq = 0; seq < capacity; ++seq) {
            auto & state = slots[seq];
            if (state.pending == LLAMA_TOKEN_NULL) continue;
            const int32_t row = llama_batch_ext_add_token(batch.get(), seq, state.pending);
            require(row >= 0 && llama_batch_ext_set_pos(batch.get(), row, &state.pos) &&
                    llama_batch_ext_set_output_logits(batch.get(), row, true), "could not build decode batch");
            ++state.pos;
            rows[seq] = row;
            ids.push_back(state.id);
        }
        if (ids.empty()) return false;
        const double begin = elapsed();
        require(llama_process(decoder, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decoder batch failed");
        const double processed = elapsed();
        llama_synchronize(decoder);
        const double duration = elapsed() - begin;
        if (std::getenv("QWEN_TIMING")) {
            std::fprintf(stderr, "qwen-timing decode process %.3f sync %.3f total %.3f ms rows %zu\n",
                         processed - begin, duration - (processed - begin), duration, ids.size());
        }
        decode_ms += duration;
        ++decode_calls;
        paired_calls += ids.size() == 2;
        if (report && ids.size() == 2) {
            emit({{"event", "paired_decode"}, {"ids", ids}, {"at_ms", begin}, {"decode_ms", duration}});
        }
        for (int seq = 0; seq < capacity; ++seq) {
            if (rows[seq] >= 0) sample(seq, rows[seq]);
        }
        return true;
    };

    auto step_admit = [&]() -> bool {
        if (prefilling >= 0) return false;
        // "idle" never starts a prefill while another slot is mid-generation, so an
        // active sequence is never made to wait for a newcomer.
        if (idle_only && any_pending()) return false;
        for (int seq = 0; seq < capacity; ++seq) {
            auto & state = slots[seq];
            if (state.id != -1) continue;
            if (next >= requests.size() || requests[next].arrival_ms > elapsed()) continue;
            const size_t id = next++;
            state.id = int(id);
            state.admitted_ms = elapsed();
            if (report) {
                emit({{"event", "admit"}, {"id", id}, {"slot", seq}, {"at_ms", state.admitted_ms}});
            }
            prepare(audio, decoder, prompt, requests[id].path, max_tokens, state);
            prefilling = seq;
            return true;
        }
        return false;
    };

    auto step_prefill = [&]() -> bool {
        if (prefilling < 0) return false;
        auto & state = slots[prefilling];
        const size_t n_chunks = mtmd_input_chunks_size(state.chunks.get());
        const double budget_begin = elapsed();
        bool worked = false;
        do {
            if (packing == "packed") {
                prefill_packed(audio, decoder, batch.get(), prefilling, state);
            } else {
                prefill_step(audio, decoder, prefilling, state);
            }
            worked = true;
            // One chunk per round is what bounds the stall: an active sequence
            // then waits for one prefill chunk, not for the whole prompt.
            if (!chunked) continue;
            if (budget_ms <= 0) break;
            if (elapsed() - budget_begin >= budget_ms) break;
        } while (state.chunk_index < n_chunks);

        if (state.chunk_index == n_chunks) {
            state.prefill_ms = elapsed() - state.admitted_ms;
            state.chunks.reset();
            const int seq = prefilling;
            prefilling = -1;
            sample(seq, -1);
        }
        return worked;
    };

    while (done < requests.size()) {
        bool progress = false;
        if (decode_first) {
            progress |= step_decode();
            progress |= step_admit();
            progress |= step_prefill();
        } else {
            progress |= step_admit();
            progress |= step_prefill();
            progress |= step_decode();
        }
        if (progress) continue;
        if (next < requests.size() && prefilling < 0) {
            std::this_thread::sleep_until(start + std::chrono::milliseconds(requests[next].arrival_ms));
            continue;
        }
        // Nothing runnable and nothing to wait for: either a prefill is mid-flight
        // (step_prefill would have progressed) or the run is finished.
        if (prefilling < 0 && !any_pending()) break;
    }

    if (report) {
        emit({{"event", "summary"}, {"capacity", capacity}, {"packing", packing}, {"schedule", schedule},
              {"budget_ms", budget_ms}, {"makespan_ms", elapsed()},
              {"decode_calls", decode_calls}, {"decode_ms", decode_ms},
              {"paired_calls", paired_calls}, {"requests", requests.size()}});
    }
}

int main(int argc, char ** argv) {
    if (argc < 6 || argc > 10) {
        std::cerr << "usage: llama-qwen3-asr-batch MODEL MMPROJ CPU|HTP0:0 SLOTS TRACE.tsv "
                     "[MAX_TOKENS [eager|windowed|packed [blocking|decode-first|interleaved|idle [PREFILL_BUDGET_MS]]]]\n";
        return 2;
    }
    try {
        const int capacity = parse_int(argv[4], 1, 2);
        const int max_tokens = argc >= 7 ? parse_int(argv[6], 1, 256) : 96;
        const std::string_view packing = argc >= 8 ? argv[7] : "eager";
        require(packing == "windowed" || packing == "packed" || packing == "eager",
                "expected eager, windowed or packed packing");
        const std::string_view schedule = argc >= 9 ? argv[8] : "blocking";
        require(schedule == "blocking" || schedule == "decode-first" || schedule == "interleaved" || schedule == "idle",
                "expected blocking, decode-first, interleaved or idle schedule");
        // A negative budget means "one chunk per round", which is the strongest
        // interleave; zero and above are milliseconds of prefill per round.
        const double budget_ms = argc >= 10 ? static_cast<double>(parse_int(argv[9], -1, 10000)) : -1.0;
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
        // Batch 1 disabled flash attention on HTP, on the assumption that the
        // fallback path without HMX is slower. Measured on this model it is not:
        // 2501 ms against 2980 ms for one request, because the decoder prefill
        // goes from 1043 to 761 ms. QWEN_FLASH_ATTN=0 restores the old arm.
        const char * fa_env = std::getenv("QWEN_FLASH_ATTN");
        const bool fa_off = fa_env && std::string(fa_env) == "0";
        ctx_params.flash_attn_type = (!cpu && fa_off) ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
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

        run(audio.get(), decoder.get(), prompt, {{0, requests.front().path}}, 1, max_tokens, packing, "blocking", -1.0, false);
        run(audio.get(), decoder.get(), prompt, requests, capacity, max_tokens, packing, schedule, budget_ms, true);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
