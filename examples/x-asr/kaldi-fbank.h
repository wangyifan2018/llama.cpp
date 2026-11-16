#pragma once

#include <vector>

// Append complete 80-bin Kaldi fbank frames. Flush reflects the final edge.
void xasr_fbank_append(const std::vector<float> & pcm, bool flush, std::vector<float> & frames);
