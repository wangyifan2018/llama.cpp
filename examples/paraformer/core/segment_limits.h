#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pf_segment {
constexpr size_t sample_rate = 16000;
constexpr size_t max_samples = sample_rate * 30;

inline bool finite(const std::vector<float>& samples) {
    return std::all_of(samples.begin(), samples.end(), [](float sample) { return std::isfinite(sample); });
}
} // namespace pf_segment
