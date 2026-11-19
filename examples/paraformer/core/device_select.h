#pragma once

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

// Pick the default compute device once, the same way in every entry point.
//
// The rule: use the first registered device whose name contains "HTP"; if there
// is none, stay on CPU. A host build has no HTP device registered, so it lands
// on CPU without failing, and a phone run gets the DSP without needing a flag.
//
// `--device` still wins. Pass the parsed device in and only call this when the
// flag was absent, so an explicit request is never overridden.
//
// This has to run before anything reads the device string. The first version
// lived after setenv("PARAFORMER_DEVICE"), so an auto-selected run printed
// "using HTP0:0" and then brought the engine up on CPU: the log said one thing
// and /health.backends[0].actual said another.
//
// True when a backend device whose name contains "HTP" is registered.
inline bool pf_htp_present() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const char* name = ggml_backend_dev_name(ggml_backend_dev_get(i));
        if (name && std::strstr(name, "HTP")) {
            return true;
        }
    }
    return false;
}

// The returned pointer is a string literal, so it outlives every caller.
inline const char* pf_default_device() {
    return pf_htp_present() ? "HTP0:0" : "CPU";
}

// Report the choice once. `have_htp` is separate from the returned name so the
// message can distinguish "no HTP device registered" from "HTP disabled by a
// flag", which are different failures to debug.
inline void pf_report_device(const char* device, bool htp_present, bool suppressed, bool service = false) {
    if (std::strcmp(device, "CPU") != 0) {
        std::fprintf(stderr, "pf: --device not given, using %s (pass --device CPU to force CPU)\n", device);
    } else if (htp_present && suppressed) {
        std::fprintf(stderr, "pf: --device not given, staying on CPU (HTP available but suppressed)\n");
    } else if (service) {
        // The server refuses to start without an HTP engine unless
        // --allow-cpu-fallback is given, so on a host build this line is
        // followed by that refusal. Say why, otherwise the two lines look
        // contradictory: "using CPU" and then "refusing to serve on CPU only".
        std::fprintf(stderr,
                     "pf: --device not given, using CPU (no HTP device registered; "
                     "a service run needs --allow-cpu-fallback on this host)\n");
    } else {
        std::fprintf(stderr, "pf: --device not given, using CPU (no HTP device registered)\n");
    }
}