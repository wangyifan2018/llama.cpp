#pragma once
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Minimal device selection for the Paraformer example. The CLI sets
// PARAFORMER_DEVICE from --device; paraformer.cpp forces this path with
// CRISPASR_PARAFORMER_GPU=1.
//
// Picking a virtual session uses the GGML_HEXAGON_DEVICES form "HTP0:0", but
// ggml_backend_dev_by_name matches the registered device name exactly, and the
// registry only knows "HTP0". Accept that form by dropping the session suffix;
// the session itself is selected through GGML_HEXAGON_DEVICES, not here.
inline ggml_backend_t crispasr_init_gpu_backend() {
    const char * name = std::getenv("PARAFORMER_DEVICE");
    if (!name || !*name) {
        return nullptr;
    }
    ggml_backend_t backend = ggml_backend_init_by_name(name, nullptr);
    if (!backend) {
        const char * colon = std::strchr(name, ':');
        if (colon) {
            char dev[64];
            const size_t len = std::min((size_t) (colon - name), sizeof(dev) - 1);
            std::memcpy(dev, name, len);
            dev[len] = '\0';
            backend = ggml_backend_init_by_name(dev, nullptr);
            if (backend) {
                fprintf(stderr, "paraformer: device %s resolved to %s; set GGML_HEXAGON_DEVICES=%s to pick the session\n",
                        name, dev, name);
                return backend;
            }
        }
        fprintf(stderr, "paraformer: cannot initialize device %s\n", name);
    }
    return backend;
}
