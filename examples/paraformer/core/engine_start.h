#pragma once

#include "paraformer.h"

#include <memory>
#include <vector>

using paraformer_context_ptr = std::unique_ptr<paraformer_context, decltype(&paraformer_free)>;

inline paraformer_context_ptr paraformer_start_engine(paraformer_weights* weights, paraformer_context_params params,
                                                      const std::vector<float>& warmup) {
    paraformer_context_ptr ctx(paraformer_init_shared(weights, params), &paraformer_free);
    if (ctx && !warmup.empty() && !paraformer_warmup(ctx.get(), warmup.data(), (int) warmup.size())) {
        ctx.reset();
    }
    return ctx;
}
