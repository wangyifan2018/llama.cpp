#include "models/models.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

struct frame {
    std::vector<float> hidden;
    std::vector<int32_t> codes;
};

using values = std::vector<float>;

struct prefix {
    values hidden;
    std::vector<values> keys;
    std::vector<values> vals;
};

struct tail_adapter {
    int32_t rank = 0;
    values projection;
    std::vector<values> heads, biases;

    tail_adapter(const char * path, int vocab, int n_heads = 10) {
        std::ifstream file(path, std::ios::binary);
        file.read(reinterpret_cast<char *>(&rank), sizeof(rank));
        if (!file || rank < 1 || rank > 128) {
            throw std::runtime_error("invalid tail adapter rank");
        }
        auto read = [&](size_t count) {
            values out(count);
            file.read(reinterpret_cast<char *>(out.data()), count * sizeof(float));
            if (!file) {
                throw std::runtime_error("truncated tail adapter");
            }
            return out;
        };
        projection = read(rank * vocab);
        for (int i = 0; i < n_heads; ++i) {
            heads.push_back(read(rank * vocab));
        }
        for (int i = 0; i < n_heads; ++i) {
            biases.push_back(read(vocab));
        }
    }
};

struct replay : mtmd_graph_qwen3tts_gen {
    replay(mtmd_subcontext & subctx, const mtmd_image_f32 & image, const frame & input)
        : mtmd_graph_qwen3tts_gen(&subctx, image, MTMD_SUBCTX_GEN_PROCESS_GEN_CODE, 0, 1.0f, 0),
          cg(*this, 0, 1.0f), input(input), keys(n_layer), vals(n_layer) {}

    code_gen cg;
    unsigned layer_mask = ~0u;
    const frame & input;
    std::vector<ggml_tensor *> keys, vals;
    std::vector<std::pair<ggml_tensor *, values>> floats;
    std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> ints;
    std::vector<ggml_tensor *> targets, drafts, second_drafts;

    ggml_tensor * input_f32(int64_t rows, const values & data) {
        auto * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rows, data.size() / rows);
        ggml_set_input(t);
        ggml_set_output(t); // Keep inputs live across timed graph executions.

        floats.emplace_back(t, data);
        return t;
    }

    ggml_tensor * input_i32(std::vector<int32_t> data) {
        auto * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, data.size());
        ggml_set_input(t);
        ggml_set_output(t); // Keep inputs live across timed graph executions.

        ints.emplace_back(t, std::move(data));
        return t;
    }

    void output(ggml_tensor * t) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    ggml_tensor * score(ggml_tensor * hidden, int depth) {
        auto * weights = model.gen_code_head_w;
        auto * head = ggml_view_2d(ctx0, weights, weights->ne[0], weights->ne[1], weights->nb[1], (depth - 1) * weights->nb[2]);
        auto * logits = ggml_mul_mat(ctx0, head, hidden);
        output(logits);
        return logits;
    }

    ggml_tensor * forward(int start, int width, const std::vector<ggml_tensor *> & prefix_codes = {}) {
        auto * hidden = start == 0 ? input_f32(input.hidden.size(), input.hidden) : nullptr;
        std::vector<int32_t> positions;
        for (int pos = start; pos < start + width; ++pos) {
            positions.push_back(pos);
            if (pos == 0) {
                continue;
            }
            auto * weights = model.gen_code_out_embd_w;
            if (pos > 1) {
                auto * table = model.gen_code_embd_w;
                weights = ggml_view_2d(ctx0, table, table->ne[0], table->ne[1], table->nb[1], (pos - 2) * table->nb[2]);
            }
            const int32_t id = input.codes.at(pos - 1);
            if (id < 0 || id >= weights->ne[1]) {
                throw std::runtime_error("capture contains an invalid code");
            }
            auto * code = prefix_codes.empty() ? input_i32({id}) : prefix_codes.at(pos - start);
            auto * embd = ggml_get_rows(ctx0, weights, code);
            hidden = hidden ? ggml_concat(ctx0, hidden, embd, 1) : embd;
        }
        auto * cur = cg.project_in(hidden);
        auto * pos_ids = input_i32(std::move(positions));
        ggml_tensor * mask = nullptr;
        if (width > 1) {
            const int total = start + width;
            values bias(total * width, 0.0f);
            for (int row = 0; row < width; ++row) {
                for (int col = start + row + 1; col < total; ++col) {
                    bias[row * total + col] = -std::numeric_limits<float>::infinity();
                }
            }
            mask = input_f32(total, bias);
            if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
                mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
            }
        }
        for (int i = 0; i < n_layer; ++i) {
            if (layer_mask & (1u << i)) {
                cur = cg.layer_forward(cur, model.layers[i], pos_ids, keys[i], vals[i], i, mask);
            }
        }
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), model.gen_code_norm_w);
    }

    ggml_tensor * row(ggml_tensor * hidden, int index) {
        return ggml_view_1d(ctx0, hidden, hidden->ne[0], index * hidden->nb[1]);
    }

    void full(int width, bool with_drafts, bool window3) {
        const int last = model.gen_code_head_w->ne[2];
        for (int start = 0; start <= last; start += width) {
            const int count = std::min(width, last + 1 - start);
            auto * hidden = forward(start, count);
            for (int i = 0; i < count; ++i) {
                const int depth = start + i;
                if (depth > 0) {
                    targets.push_back(score(row(hidden, i), depth));
                }
                if (with_drafts && depth > 0 && depth < last) {
                    drafts.push_back(score(row(hidden, i), depth + 1));
                    if (window3 && depth + 2 <= last) {
                        second_drafts.push_back(score(row(hidden, i), depth + 2));
                    }
                }
            }
        }
    }

    void greedy_ar(bool forced, const tail_adapter * correction) {
        auto * hidden = row(forward(0, 2), 1);
        const int last = model.gen_code_head_w->ne[2];
        for (int depth = 1; depth <= last; ++depth) {
            auto * logits = score(hidden, depth);
            if (correction) {
                const int i = depth - 1;
                auto * features = ggml_mul_mat(ctx0, input_f32(logits->ne[0], correction->projection), logits);
                auto * delta = ggml_mul_mat(ctx0, input_f32(correction->rank, correction->heads[i]), features);
                logits = ggml_add(ctx0, logits, ggml_add(ctx0, delta, input_f32(logits->ne[0], correction->biases[i])));
                output(logits);
            }
            targets.push_back(logits);
            if (depth < last) {
                if (forced) {
                    hidden = row(forward(depth + 1, 1), 0);
                } else {
                    hidden = row(forward(depth + 1, 1, {ggml_argmax(ctx0, logits)}), 0);
                }
            }
        }
    }

    void ar_prefix_tail(const tail_adapter & adapter, const char * split, const tail_adapter * second, bool exact_mid) {
        const int prefix_depth = exact_mid ? 9 : 4;
        auto * hidden = row(forward(0, 2), 1);
        for (int depth = 1; depth <= prefix_depth; ++depth) {
            auto * logits = score(hidden, depth);
            targets.push_back(logits);
            hidden = row(forward(depth + 1, 1, {ggml_argmax(ctx0, logits)}), 0);
        }

        auto logits_with_offset = [&](ggml_tensor * state, int depth, ggml_tensor * features, const tail_adapter & model_adapter) {
            const int head = depth - 6;
            const int64_t vocab = model.gen_code_head_w->ne[1];
            auto * correction = ggml_mul_mat(ctx0, input_f32(model_adapter.rank, model_adapter.heads[head]), features);
            correction = ggml_add(ctx0, correction, input_f32(vocab, model_adapter.biases[head]));
            auto * logits = ggml_add(ctx0, score(state, depth), correction);
            output(logits);
            return logits;
        };

        if (exact_mid) {
            auto * anchor = score(hidden, 10);
            targets.push_back(anchor);
            auto * features = ggml_mul_mat(ctx0, input_f32(anchor->ne[0], second->projection), anchor);
            for (int depth = 11; depth <= 15; ++depth) {
                targets.push_back(logits_with_offset(hidden, depth, features, *second));
            }
            return;
        }

        auto * anchor = score(hidden, 5);
        targets.push_back(anchor);
        const int64_t vocab = anchor->ne[0];
        auto * features = ggml_mul_mat(ctx0, input_f32(vocab, adapter.projection), anchor);
        std::vector<ggml_tensor *> codes;
        const int end = split ? 9 : 15;
        if (split && std::string(split) != "oracle") {
            codes.push_back(ggml_argmax(ctx0, anchor));
        }
        for (int depth = 6; depth <= end; ++depth) {
            auto * logits = logits_with_offset(hidden, depth, features, adapter);
            targets.push_back(logits);
            if (!codes.empty()) {
                codes.push_back(ggml_argmax(ctx0, logits));
            }
        }
        if (!split) {
            return;
        }

        hidden = row(forward(6, 5, codes), 4);
        anchor = score(hidden, 10);
        targets.push_back(anchor);
        features = second ? ggml_mul_mat(ctx0, input_f32(vocab, second->projection), anchor) : nullptr;
        for (int depth = 11; depth <= 15; ++depth) {
            auto * logits = second ? logits_with_offset(hidden, depth, features, *second) : score(hidden, depth);
            targets.push_back(logits);
        }
    }

    void grouped(bool student_prefix, const std::vector<values> & biases) {
        const int split = 4;
        std::vector<ggml_tensor *> codes;
        auto * hidden = row(forward(0, 2), 1);
        for (int depth = 1; depth <= split; ++depth) {
            auto * logits = score(hidden, depth);
            targets.push_back(logits);
            if (student_prefix) {
                auto * bias = input_f32(logits->ne[0], biases.at(depth - 1));
                codes.push_back(ggml_argmax(ctx0, ggml_add(ctx0, logits, bias)));
            }
        }
        hidden = row(forward(2, split, codes), split - 1);
        for (int depth = split + 1; depth <= model.gen_code_head_w->ne[2]; ++depth) {
            targets.push_back(score(hidden, depth));
        }
    }

    void round(int depth, int width, const prefix & state) {
        for (int i = 0; i < n_layer; ++i) {
            keys[i] = input_f32(state.keys[i].size() / depth, state.keys[i]);
            vals[i] = input_f32(state.vals[i].size() / depth, state.vals[i]);
        }
        if (width > 1) {
            auto * previous = input_f32(state.hidden.size(), state.hidden);
            for (int offset = 0; offset < width - 1; ++offset) {
                drafts.push_back(score(previous, depth + offset));
            }
        }
        auto * hidden = forward(depth, width);
        for (int i = 0; i < width; ++i) {
            targets.push_back(score(row(hidden, i), depth + i));
        }
    }

    void allocate(ggml_gallocr_t allocator) {
        if (!ggml_gallocr_alloc_graph(allocator, gf)) {
            throw std::runtime_error("could not allocate replay graph");
        }
        for (const auto & [tensor, data] : floats) {
            ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
        }
        for (const auto & [tensor, data] : ints) {
            ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
        }
    }
};

static values read_tensor(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    values data(ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    return data;
}

static void compute(ggml_backend_t cpu, replay & graph) {
    if (ggml_backend_graph_compute(cpu, graph.gf) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("replay graph failed");
    }
}

static prefix make_prefix(mtmd_subcontext & subctx, const mtmd_image_f32 & image, const frame & input,
                          ggml_backend_t cpu, ggml_gallocr_t allocator, int depth) {
    replay graph(subctx, image, input);
    ggml_tensor * hidden = nullptr;
    for (int i = 0; i < depth; ++i) {
        hidden = graph.forward(i, 1);
    }
    graph.output(hidden);
    for (int i = 0; i < graph.n_layer; ++i) {
        graph.output(graph.keys[i]);
        graph.output(graph.vals[i]);
    }
    graph.allocate(allocator);
    compute(cpu, graph);
    prefix state;
    state.hidden = read_tensor(hidden);
    for (int i = 0; i < graph.n_layer; ++i) {
        state.keys.push_back(read_tensor(graph.keys[i]));
        state.vals.push_back(read_tensor(graph.vals[i]));
    }
    return state;
}

struct greedy_result {
    std::vector<int32_t> codes;
    int rounds = 0;
    int drafts = 0;
    int accepted = 0;
};

static int32_t choose(const values & logits) {
    return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static greedy_result greedy_rounds(mtmd_subcontext & subctx, const mtmd_image_f32 & image, const frame & input,
                                  ggml_backend_t cpu, ggml_gallocr_t allocator, int start_spec) {
    greedy_result result;
    prefix state;
    int32_t last_code = input.codes.at(0);
    const int last_depth = subctx.model.gen_code_head_w->ne[2];
    for (int depth = 1; depth <= last_depth;) {
        replay graph(subctx, image, input);
        const bool draft = depth >= start_spec && depth < last_depth;
        ggml_tensor * hidden = nullptr;
        ggml_tensor * prediction = nullptr;
        if (depth == 1) {
            hidden = graph.forward(0, 2);
        } else {
            for (int layer = 0; layer < graph.n_layer; ++layer) {
                graph.keys[layer] = graph.input_f32(state.keys[layer].size() / depth, state.keys[layer]);
                graph.vals[layer] = graph.input_f32(state.vals[layer].size() / depth, state.vals[layer]);
            }
            std::vector<ggml_tensor *> ids{graph.input_i32({last_code})};
            if (draft) {
                auto * previous = graph.input_f32(state.hidden.size(), state.hidden);
                prediction = graph.score(previous, depth);
                ids.push_back(ggml_argmax(graph.ctx0, prediction));
            }
            hidden = graph.forward(depth, draft ? 2 : 1, ids);
        }
        const int first_row = depth == 1 ? 1 : 0;
        auto * target = graph.score(graph.row(hidden, first_row), depth);
        auto * bonus = draft ? graph.score(graph.row(hidden, 1), depth + 1) : nullptr;
        graph.output(hidden);
        for (int layer = 0; layer < graph.n_layer; ++layer) {
            graph.output(graph.keys[layer]);
            graph.output(graph.vals[layer]);
        }
        graph.allocate(allocator);
        compute(cpu, graph);
        ++result.rounds;
        last_code = choose(read_tensor(target));
        result.codes.push_back(last_code);
        const bool accept = draft && choose(read_tensor(prediction)) == last_code;
        if (draft) {
            ++result.drafts;
        }
        if (accept) {
            ++result.accepted;
            last_code = choose(read_tensor(bonus));
            result.codes.push_back(last_code);
        }
        const int kept = accept ? 2 : 1;
        if (depth + kept <= last_depth) {
            auto data = read_tensor(hidden);
            const int row = first_row + int(accept);
            state.hidden.assign(data.begin() + row * hidden->ne[0], data.begin() + (row + 1) * hidden->ne[0]);
            state.keys.resize(graph.n_layer);
            state.vals.resize(graph.n_layer);
            for (int layer = 0; layer < graph.n_layer; ++layer) {
                // Read only the committed prefix. A rejected token never enters the next round.
                auto read_prefix = [depth, kept](ggml_tensor * tensor, values & out) {
                    out.resize(tensor->ne[0] * (depth + kept));
                    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
                };
                read_prefix(graph.keys[layer], state.keys[layer]);
                read_prefix(graph.vals[layer], state.vals[layer]);
            }
        }
        depth += kept;
    }
    return result;
}

static greedy_result greedy_whole(mtmd_subcontext & subctx, const mtmd_image_f32 & image, const frame & input,
                                 ggml_backend_t cpu, ggml_gallocr_t allocator) {
    replay graph(subctx, image, input);
    graph.greedy_ar(false, nullptr);
    graph.allocate(allocator);
    compute(cpu, graph);
    greedy_result result;
    result.rounds = 1;
    for (auto * t : graph.targets) {
        result.codes.push_back(choose(read_tensor(t)));
    }
    return result;
}

static void write_tensor(std::ofstream & out, ggml_tensor * t) {
    const auto data = read_tensor(t);
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char ** argv) try {
    if (argc != 7) {
        fprintf(stderr, "usage: %s MMPROJ FRAMES.bin SCORES.bin THREADS REPEATS FRAME_INDEX (-1 skips timing)\n", argv[0]);
        return 2;
    }
    const int threads = std::stoi(argv[4]);
    const int repeats = std::stoi(argv[5]);
    const int index = std::stoi(argv[6]);
    if (threads < 1 || repeats < 1 || index < -1) {
        throw std::runtime_error("invalid benchmark parameters");
    }
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pinned(nullptr, ggml_threadpool_free);
    mtmd_backends backends(nullptr, threads, 8192, nullptr, nullptr);
    auto cpu = backends.backend_cpu;
    auto buft = ggml_backend_get_default_buffer_type(cpu);
    ggml_backend_cpu_set_n_threads(cpu, threads);
    if (std::getenv("TTS_SPEC_CPU_REF")) {
        ggml_backend_cpu_set_use_ref(cpu, true);
    }
    if (const char * cores = std::getenv("TTS_SPEC_PIN_MASK")) {
        const uint64_t mask = std::stoull(cores, nullptr, 16);
        auto params = ggml_threadpool_params_default(threads);
        int count = 0;
        for (unsigned i = 0; i < 64; ++i) {
            params.cpumask[i] = (mask >> i) & 1;
            count += params.cpumask[i];
        }
        if (count != threads) {
            throw std::runtime_error("pin mask must specify one core per thread");
        }
        params.strict_cpu = true;
        params.poll = 0;
        pinned.reset(ggml_threadpool_new(&params));
        if (!pinned) {
            throw std::runtime_error("could not create pinned thread pool");
        }
        ggml_backend_cpu_set_threadpool(cpu, pinned.get());
    }
    auto models = mtmd_model_load(argv[1], {}, buft);
    if (!models.gen_a || models.gen_a->proj_type != PROJECTOR_TYPE_QWEN3TTS_GEN) {
        throw std::runtime_error("expected Qwen3-TTS mmproj");
    }
    const auto attention = std::getenv("TTS_SPEC_FLASH") ? MTMD_FLASH_ATTN_TYPE_ENABLED : MTMD_FLASH_ATTN_TYPE_DISABLED;
    mtmd_subcontext subctx(*models.gen_a, backends, {attention, false});
    mtmd_image_f32 image;
    image.set_size({1, 1}, false, false);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(ggml_gallocr_new(buft), ggml_gallocr_free);

    std::ifstream in(argv[2], std::ios::binary);
    std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
    if (!in || !out) {
        throw std::runtime_error("could not open capture or output");
    }
    const auto * head = models.gen_a->gen_code_head_w;
    const bool window3 = std::getenv("TTS_SPEC_WINDOW3") != nullptr;
    const char * weights_path = std::getenv("TTS_SPEC_TRAIN_EXPORT");
    if (weights_path) {
        if (!ggml_is_contiguous(head) || head->type != GGML_TYPE_F16) {
            throw std::runtime_error("training export expects contiguous F16 heads");
        }
        std::vector<ggml_fp16_t> weights(ggml_nelements(head));
        ggml_backend_tensor_get(head, weights.data(), 0, ggml_nbytes(head));
        std::ofstream dump(weights_path, std::ios::binary | std::ios::trunc);
        const int32_t dims[] = {int32_t(head->ne[2]), int32_t(head->ne[1]), int32_t(head->ne[0])};
        dump.write(reinterpret_cast<const char *>(dims), sizeof(dims));
        dump.write(reinterpret_cast<const char *>(weights.data()), ggml_nbytes(head));
        if (!dump) {
            throw std::runtime_error("could not write training head weights");
        }
    }
    const int32_t shape[] = {int32_t(head->ne[2]), int32_t(head->ne[1])};
    out.write(reinterpret_cast<const char *>(shape), sizeof(shape));

    std::unique_ptr<tail_adapter> adapter;
    if (const char * path = std::getenv("TTS_SPEC_TAIL")) {
        if (shape[0] != 15 || shape[1] != 2048) {
            throw std::runtime_error("expected 15 heads with vocabulary 2048");
        }
        adapter = std::make_unique<tail_adapter>(path, shape[1]);
    }
    const char * tail_split = std::getenv("TTS_SPEC_TAIL_SPLIT");
    std::unique_ptr<tail_adapter> second_adapter;
    if (adapter && tail_split && std::string(tail_split) != "raw" && std::string(tail_split) != "oracle") {
        second_adapter = std::make_unique<tail_adapter>(tail_split, shape[1]);
    }
    const bool exact_mid = std::getenv("TTS_SPEC_TAIL_AR9") != nullptr;
    if (exact_mid && !second_adapter) {
        throw std::runtime_error("exact mid prefix requires a second adapter");
    }
    unsigned layer_mask = ~0u;
    if (const char * digits = std::getenv("TTS_SPEC_LAYERS")) {
        layer_mask = 0;
        for (const char * c = digits; *c; ++c) {
            if (*c < '0' || *c - '0' >= int(models.gen_a->layers.size())) {
                throw std::runtime_error("invalid predictor layer index");
            }
            layer_mask |= 1u << (*c - '0');
        }
        if (layer_mask == 0) {
            throw std::runtime_error("predictor layer list is empty");
        }
    }
    std::unique_ptr<tail_adapter> ar_correction;
    if (const char * path = std::getenv("TTS_SPEC_AR_CORRECTION")) {
        ar_correction = std::make_unique<tail_adapter>(path, shape[1], shape[0]);
    }
    const char * grouped_mode = std::getenv("TTS_SPEC_GROUPED");
    const bool force_prefix = grouped_mode && std::string(grouped_mode) == "ar-forced";
    const bool greedy_baseline = force_prefix || (grouped_mode && std::string(grouped_mode) == "ar-greedy");
    const bool student_prefix = grouped_mode && !greedy_baseline && std::string(grouped_mode) != "teacher";
    std::vector<values> biases;
    if (student_prefix) {
        std::ifstream file(grouped_mode, std::ios::binary);
        for (int i = 0; i < 4; ++i) {
            auto & bias = biases.emplace_back(shape[1]);
            file.read(reinterpret_cast<char *>(bias.data()), bias.size() * sizeof(float));
        }
        if (!file) {
            throw std::runtime_error("could not read four grouped prefix bias rows");
        }
    }

    const char * greedy_mode = std::getenv("TTS_SPEC_GREEDY_ROUNDS");
    const bool whole = greedy_mode && std::string(greedy_mode) == "whole";
    const int start_spec = greedy_mode && !whole ? std::stoi(greedy_mode) : shape[0] + 1;
    if (start_spec < 2) {
        throw std::runtime_error("greedy speculation requires a committed first code");
    }

    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (greedy_mode) {
        fprintf(stdout, "frame,repeat,rounds,drafts,accepted,wall_ms\n");
    } else {
        fprintf(stdout, "depth,width,repeat,compute_ms\n");
    }

    for (int n = 0; in.peek() != EOF; ++n) {
        frame input{values(models.gen_a->n_mmproj_embd()), std::vector<int32_t>(head->ne[2] + 1)};
        in.read(reinterpret_cast<char *>(input.hidden.data()), input.hidden.size() * sizeof(float));
        in.read(reinterpret_cast<char *>(input.codes.data()), input.codes.size() * sizeof(int32_t));
        if (!in) {
            throw std::runtime_error("truncated frame capture");
        }
        if (greedy_mode) {
            greedy_result result;
            for (int iteration = 0; iteration < repeats; ++iteration) {
                const auto start = std::chrono::steady_clock::now();
                result = whole ? greedy_whole(subctx, image, input, cpu, alloc.get())
                               : greedy_rounds(subctx, image, input, cpu, alloc.get(), start_spec);
                const auto stop = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
                fprintf(stdout, "%d,%d,%d,%d,%d,%.6f\n", n, iteration, result.rounds, result.drafts, result.accepted, ms);
            }
            out.write(reinterpret_cast<const char *>(result.codes.data()), result.codes.size() * sizeof(int32_t));
            continue;
        }

        if (grouped_mode || adapter) {
            replay graph(subctx, image, input);
            if (adapter) {
                graph.ar_prefix_tail(*adapter, tail_split, second_adapter.get(), exact_mid);
            } else if (greedy_baseline) {
                graph.layer_mask = layer_mask;
                graph.greedy_ar(force_prefix, ar_correction.get());
            } else {
                graph.grouped(student_prefix, biases);
            }
            graph.allocate(alloc.get());
            compute(cpu, graph);
            if (index == n) {
                for (int warm = 0; warm < 8; ++warm) {
                    compute(cpu, graph);
                }
                for (int i = 0; i < repeats; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    compute(cpu, graph);
                    const auto stop = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
                    fprintf(stdout, "0,%d,%d,%.6f\n", greedy_baseline ? 16 : 2, i, ms);
                }
            }
            for (auto * t : graph.targets) {
                write_tensor(out, t);
            }
            continue;
        }
        if (index == n) {
            for (int depth = 2; depth <= shape[0]; ++depth) {
                if (!window3 && depth != 2 && depth != 8 && depth != 14) {
                    continue;
                }
                const int widths = window3 ? std::min(3, shape[0] + 1 - depth) : 2;
                const auto state = make_prefix(subctx, image, input, cpu, alloc.get(), depth);
                // Rotate width order to limit thermal and frequency bias.
                for (int block = 0; block < (window3 ? 6 : 4); ++block) {
                    for (int arm = 0; arm < widths; ++arm) {
                        const int width = 1 + ((block + arm) % widths);
                        replay graph(subctx, image, input);
                        graph.round(depth, width, state);
                        graph.allocate(alloc.get());
                        for (int i = 0; i < 3; ++i) {
                            compute(cpu, graph);
                        }
                        for (int i = 0; i < repeats; ++i) {
                            const auto start = std::chrono::steady_clock::now();
                            compute(cpu, graph);
                            const auto stop = std::chrono::steady_clock::now();
                            const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
                            fprintf(stdout, "%d,%d,%d,%.6f\n", depth, width, block * repeats + i, ms);
                        }
                    }
                }
            }
        }
        for (int width = 1; width <= (weights_path ? 1 : window3 ? 3 : 2); ++width) {
            replay graph(subctx, image, input);
            graph.full(width, width == 1, window3);
            graph.allocate(alloc.get());
            compute(cpu, graph);
            for (auto * t : graph.targets) {
                write_tensor(out, t);
            }
            for (auto * t : graph.drafts) {
                write_tensor(out, t);
            }
            for (auto * t : graph.second_drafts) {
                write_tensor(out, t);
            }
        }
    }
    if (!out || out.tellp() <= std::streampos(sizeof(shape))) {
        throw std::runtime_error("empty or incomplete replay output");
    }
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
}
