// vocbench: run code2wav (GEN_WAV) alone on one backend, to time it and to A/B two devices
//
// the full TTS pipeline needs the backbone and the code predictor too; this tool feeds synthetic
// codes straight to the vocoder so the two backends can be compared on identical input.

#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    printf("usage: %s -m backbone.gguf -mm mmproj.gguf [-dev CPU|HTP0] [-n frames] [-c chunks] [-o out.f32]\n", argv0);
}

int main(int argc, char ** argv) {
    std::string path_model, path_mmproj, device = "CPU", out_path;
    int n_frames = 8;
    int n_chunks = 3;

    for (int i = 1; i < argc; i++) {
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if      (!strcmp(argv[i], "-m"))   path_model  = next();
        else if (!strcmp(argv[i], "-mm"))  path_mmproj = next();
        else if (!strcmp(argv[i], "-dev")) device      = next();
        else if (!strcmp(argv[i], "-n"))   n_frames    = atoi(next());
        else if (!strcmp(argv[i], "-c"))   n_chunks    = atoi(next());
        else if (!strcmp(argv[i], "-o"))   out_path    = next();
        else { usage(argv[0]); return 1; }
    }
    if (path_model.empty() || path_mmproj.empty()) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // the backbone is only needed for the vocab, keep it off the backend
    llama_model * model = llama_model_load_from_file(path_model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path_model.c_str());
        return 1;
    }

    mtmd_context_params cparams = mtmd_context_params_default();
    // resolve every name, including CPU, so both arms go through the same code path
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
    if (!dev) {
        fprintf(stderr, "no such device: %s\n", device.c_str());
        return 1;
    }
    cparams.device  = dev;
    cparams.use_gpu = strcmp(device.c_str(), "CPU") != 0;
    mtmd_context * mctx = mtmd_init_from_file(path_mmproj.c_str(), model, cparams);
    if (!mctx) {
        fprintf(stderr, "failed to load %s\n", path_mmproj.c_str());
        return 1;
    }

    const auto info = mtmd_gen_audio_get_info(mctx);
    if (info.type == MTMD_GEN_AUDIO_TYPE_NONE) {
        fprintf(stderr, "mmproj has no audio generation\n");
        return 1;
    }

    mtmd_gen_state_params sp = mtmd_gen_state_params_default();
    sp.seed = 1234;
    mtmd_gen_state * state = mtmd_gen_state_init(mctx, sp);
    if (!state) {
        fprintf(stderr, "failed to init gen state\n");
        return 1;
    }

    // deterministic codes: all 16 codebooks use 2048 entries (semantic_codebook_size 4096 is unused)
    const int n_cb = 16;
    std::vector<int32_t> codes((size_t) n_frames * n_cb);
    uint32_t rng = 1234;
    for (size_t i = 0; i < codes.size(); i++) {
        rng = rng * 1664525u + 1013904223u;
        codes[i] = (int32_t) ((rng >> 8) % 2048u);
    }

    std::vector<float> pcm;
    for (int c = 0; c < n_chunks; c++) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes   = codes.data();
        inp.n_codes = codes.size();

        mtmd_gen_out out = {};
        const auto t0 = std::chrono::steady_clock::now();
        if (mtmd_gen_audio_process(mctx, state, &inp, &out) != 0) {
            fprintf(stderr, "gen_wav failed on chunk %d\n", c);
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("chunk %d: %6.1f ms  (%zu samples, %.2f ms/frame)\n", c, ms, out.n_samples, ms / n_frames);
        pcm.insert(pcm.end(), out.audio, out.audio + out.n_samples);
    }

    printf("total %zu samples = %.3f s of audio\n", pcm.size(), (double) pcm.size() / 24000.0);

    if (!out_path.empty()) {
        FILE * f = fopen(out_path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", out_path.c_str());
            return 1;
        }
        fwrite(pcm.data(), sizeof(float), pcm.size(), f);
        fclose(f);
        printf("wrote %s\n", out_path.c_str());
    }

    mtmd_gen_state_free(state);
    mtmd_free(mctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
