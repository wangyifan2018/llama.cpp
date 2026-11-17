#pragma once
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace core_cpu_backend {
inline ggml_backend_t init() { return ggml_backend_cpu_init(); }
inline void set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_cpu_set_n_threads(backend, n_threads);
}
inline bool is_metal(ggml_backend_t) { return false; }
} // namespace core_cpu_backend
