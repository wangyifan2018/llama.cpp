#pragma once
#include "ggml-backend.h"
#include "ggml.h"

namespace core_sched_prof {
inline ggml_status compute(ggml_backend_sched_t sched, ggml_cgraph * gf, const char *) {
    return ggml_backend_sched_graph_compute(sched, gf);
}
} // namespace core_sched_prof
