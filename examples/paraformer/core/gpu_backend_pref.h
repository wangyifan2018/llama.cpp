#pragma once
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

// Minimal device selection for the Paraformer example. The CLI sets
// PARAFORMER_DEVICE from --device; paraformer.cpp forces this path with
// CRISPASR_PARAFORMER_GPU=1.
inline ggml_backend_t crispasr_init_gpu_backend() {
    const char * name = std::getenv("PARAFORMER_DEVICE");
    if (!name || !*name) {
        return nullptr;
    }
    ggml_backend_t backend = ggml_backend_init_by_name(name, nullptr);
    if (!backend) {
        fprintf(stderr, "paraformer: cannot initialize device %s\n", name);
    }
    return backend;
}
