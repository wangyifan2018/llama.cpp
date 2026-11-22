// mmabench: is the code predictor bandwidth bound, and how much does batching help
//
// Builds the exact matmul shapes of one code predictor step (5 layers + one head) with q8_0
// weights and runs `steps` chained steps for several batch sizes. The numbers answer one
// question: does per-frame cost fall like 1/N (bandwidth bound) or flatten early (compute bound).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

static const int N_EMBD  = 1024;
static const int N_FF    = 3072;
static const int N_HEAD  = 2048;
static const int N_LAYER = 5;
static const int N_BOOK  = 15;

static double now_ms() { return (double) ggml_time_us() / 1000.0; }

static void fill_q8(ggml_backend_t backend, ggml_tensor * t, int seed) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
    std::vector<float> src((size_t) ne0 * ne1);
    srand(seed);
    for (auto & v : src) v = (float) (rand() % 2000 - 1000) / 1000.0f * 0.02f;
    std::vector<char> dst(ggml_row_size(GGML_TYPE_Q8_0, ne0) * ne1);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), dst.data(), 0, ne1, (int) ne0, nullptr);
    ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
}

// --- graph equivalence ---------------------------------------------------------------
//
// Claim checked here: batching independent sequences does not let them interfere. One chain of
// `steps` matmul blocks with a per row code fed back; run it once with `rows` rows, then once per
// row with 1 row, and compare the final activations row by row.

static std::vector<float> run_chain(ggml_backend_t backend, int rows, int steps, const std::vector<int> & codes) {
    ggml_init_params ip = { (size_t) 512 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, rows);
    ggml_set_input(x);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
    ggml_set_input(idx);

    std::vector<ggml_tensor *> wv(N_LAYER), wo(N_LAYER), wg(N_LAYER), wu(N_LAYER), wd(N_LAYER);
    for (int il = 0; il < N_LAYER; il++) {
        wv[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_EMBD);
        wo[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2 * N_EMBD, N_EMBD);
        wg[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_FF);
        wu[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_FF);
        wd[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_FF, N_EMBD);
    }
    ggml_tensor * emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, 2048);

    ggml_tensor * cur = x;
    for (int s = 0; s < steps; s++) {
        for (int il = 0; il < N_LAYER; il++) {
            ggml_tensor * v  = ggml_mul_mat(ctx, wv[il], cur);
            ggml_tensor * v2 = ggml_concat(ctx, v, v, 0);
            cur = ggml_add(ctx, cur, ggml_mul_mat(ctx, wo[il], v2));
            ggml_tensor * gate = ggml_mul_mat(ctx, wg[il], cur);
            ggml_tensor * up   = ggml_mul_mat(ctx, wu[il], cur);
            ggml_tensor * m    = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            cur = ggml_add(ctx, cur, ggml_mul_mat(ctx, wd[il], m));
        }
        cur = ggml_add(ctx, cur, ggml_get_rows(ctx, emb, idx));
        cur = ggml_scale(ctx, ggml_rms_norm(ctx, cur, 1e-6f), 1.0f);   // keep the chain bounded
    }
    ggml_set_output(cur);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) { ggml_free(ctx); return {}; }

    std::vector<float> xdata((size_t) N_EMBD * rows, 0.01f);
    ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * sizeof(float));
    ggml_backend_tensor_set(idx, codes.data(), 0, codes.size() * sizeof(int));
    for (int il = 0; il < N_LAYER; il++) {
        fill_q8(backend, wv[il], 10 + il);
        fill_q8(backend, wo[il], 20 + il);
        fill_q8(backend, wg[il], 30 + il);
        fill_q8(backend, wu[il], 40 + il);
        fill_q8(backend, wd[il], 50 + il);
    }
    {
        std::vector<float> e((size_t) N_EMBD * 2048);
        srand(77);
        for (auto & v : e) v = (float) (rand() % 2000 - 1000) / 1000.0f * 0.02f;
        ggml_backend_tensor_set(emb, e.data(), 0, e.size() * sizeof(float));
    }

    ggml_backend_graph_compute(backend, gf);
    std::vector<float> out((size_t) N_EMBD * rows);
    ggml_backend_tensor_get(cur, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static int run_equiv(ggml_backend_t backend, int steps) {
    const int K = 4;
    std::vector<int> codes(K);
    for (int i = 0; i < K; i++) codes[i] = (i * 37 + 11) % 2048;

    std::vector<float> batched = run_chain(backend, K, steps, codes);
    if (batched.empty()) { printf("batched run failed\n"); return 1; }

    double worst = 0.0;
    for (int i = 0; i < K; i++) {
        std::vector<float> single = run_chain(backend, 1, steps, { codes[i] });
        if (single.empty()) { printf("single run %d failed\n", i); return 1; }
        double d = 0.0;
        for (int j = 0; j < N_EMBD; j++) {
            d = std::max(d, (double) std::fabs(batched[(size_t) j + (size_t) i * N_EMBD] - single[(size_t) j]));
        }
        printf("row %d vs single run: max |diff| = %.3e\n", i, d);
        worst = std::max(worst, d);
    }
    printf("worst %.3e -> %s\n", worst, worst < 1e-4 ? "EQUIVALENT" : "NOT EQUIVALENT");
    return worst < 1e-4 ? 0 : 1;
}
int main(int argc, char ** argv) {
    int n_threads = 4, steps = N_BOOK, reps = 8;
    std::string device = "CPU";
    bool equiv = false;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-t")) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--equiv")) equiv = true;
        else if (!strcmp(argv[i], "-dev")) device = argv[++i];
        else if (!strcmp(argv[i], "-s")) steps     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r")) reps      = atoi(argv[++i]);
        else { printf("usage: %s [-t threads] [-s steps] [-r reps]\n", argv[0]); return 1; }
    }

    const size_t bytes_layer = (size_t) N_EMBD * N_EMBD + (size_t) N_EMBD * 2 * N_EMBD
                             + 3 * (size_t) N_FF * N_EMBD;
    const size_t w_bytes = (size_t) N_LAYER * bytes_layer + (size_t) N_HEAD * N_EMBD;

    ggml_backend_load_all();   // picks up dynamic backends, e.g. ggml-hexagon on a device

    ggml_backend_t backend = nullptr;
    if (device == "CPU") {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    } else {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
        if (dev == nullptr) {
            fprintf(stderr, "unknown device %s\n", device.c_str());
            return 1;
        }
        backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            fprintf(stderr, "failed to init %s\n", device.c_str());
            return 1;
        }
    }

    if (equiv) { const int rc = run_equiv(backend, steps); ggml_backend_free(backend); return rc; }

    printf("weights %.1f MB per step, %d steps per frame, %d threads\n", (double) w_bytes / 1e6, steps, n_threads);
    printf("%5s %10s %12s %10s %10s %8s\n", "rows", "ms/step", "ms/frame", "GB/s", "GFLOP/s", "MAC/B");

    for (int rows : { 1, 2, 4, 8, 16, 32 }) {
        ggml_init_params ip = { (size_t) 512 * 1024 * 1024, nullptr, true };  // no_alloc: backend allocs
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, rows);
        ggml_set_input(x);

        std::vector<ggml_tensor *> wv(N_LAYER), wo(N_LAYER), wg(N_LAYER), wu(N_LAYER), wd(N_LAYER);
        for (int il = 0; il < N_LAYER; il++) {
            wv[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_EMBD);
            wo[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2 * N_EMBD, N_EMBD);
            wg[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_FF);
            wu[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_FF);
            wd[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_FF, N_EMBD);
        }
        ggml_tensor * wh = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_HEAD);

        ggml_tensor * cur = x;
        for (int s = 0; s < steps; s++) {
            for (int il = 0; il < N_LAYER; il++) {
                ggml_tensor * v  = ggml_mul_mat(ctx, wv[il], cur);
                ggml_tensor * v2 = ggml_concat(ctx, v, v, 0);      // same bytes as the GQA expansion
                ggml_tensor * o  = ggml_mul_mat(ctx, wo[il], v2);
                cur = ggml_add(ctx, cur, o);
                ggml_tensor * gate = ggml_mul_mat(ctx, wg[il], cur);
                ggml_tensor * up   = ggml_mul_mat(ctx, wu[il], cur);
                ggml_tensor * m    = ggml_mul(ctx, ggml_silu(ctx, gate), up);
                cur = ggml_add(ctx, cur, ggml_mul_mat(ctx, wd[il], m));
            }
            ggml_tensor * lg   = ggml_mul_mat(ctx, wh, cur);           // [vocab, rows]
            ggml_tensor * back = ggml_view_2d(ctx, lg, N_EMBD, rows, lg->nb[1], 0);
            cur = ggml_add(ctx, cur, back);
        }
        ggml_set_output(cur);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, cur);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) { printf("%5d  alloc failed\n", rows); ggml_free(ctx); continue; }

        std::vector<float> xdata((size_t) N_EMBD * rows, 0.01f);
        ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * sizeof(float));
        for (int il = 0; il < N_LAYER; il++) {
            fill_q8(backend, wv[il], 10 + il);
            fill_q8(backend, wo[il], 20 + il);
            fill_q8(backend, wg[il], 30 + il);
            fill_q8(backend, wu[il], 40 + il);
            fill_q8(backend, wd[il], 50 + il);
        }
        fill_q8(backend, wh, 60);

        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            const double t0 = now_ms();
            ggml_backend_graph_compute(backend, gf);
            const double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }

        const double ms_step  = best / steps;
        const double ms_frame = best / (double) rows;
        const double gbs      = (double) w_bytes / (ms_step * 1e6);
        const double gflops   = 2.0 * (double) w_bytes * rows / (ms_step * 1e6);
        printf("%5d %10.2f %12.2f %10.2f %10.1f %8.1f\n", rows, ms_step, ms_frame, gbs, gflops, (double) rows);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return 0;
}