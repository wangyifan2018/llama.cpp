// Adapted from CrispASR. See LICENSE.crispasr.

#include "kaldi-fbank.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace {

constexpr int win = 400;
constexpr int hop = 160;
constexpr int lead = 120;
constexpr int n_fft = 512;
constexpr int n_bins = n_fft / 2;
constexpr int n_mels = 80;
constexpr float pi = 3.14159265358979323846f;

static void fft_radix2(float * re, float * im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * pi / (float)len;
        const float wre = std::cos(ang);
        const float wim = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                const float tr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
                const float ti = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
                re[i + j + len / 2] = re[i + j] - tr;
                im[i + j + len / 2] = im[i + j] - ti;
                re[i + j] += tr;
                im[i + j] += ti;
                const float nr = cr * wre - ci * wim;
                ci = cr * wim + ci * wre;
                cr = nr;
            }
        }
    }
}

struct fbank_filters {
    std::array<float, win> window;
    std::array<std::array<float, n_bins>, n_mels> weights = {};
    std::array<std::pair<int, int>, n_mels> ranges;

    fbank_filters() {
        for (int i = 0; i < win; ++i) {
            const float hann = 0.5f - 0.5f * std::cos(2.0f * pi * float(i) / float(win - 1));
            window[i] = std::pow(hann, 0.85f);
        }
        auto mel = [](float hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); };
        const float low = mel(20.0f);
        const float step = (mel(7600.0f) - low) / (n_mels + 1);
        for (int m = 0; m < n_mels; ++m) {
            const float left = low + m * step;
            const float center = left + step;
            const float right = center + step;
            int first = n_bins;
            int last = 0;
            for (int k = 0; k < n_bins; ++k) {
                const float x = mel(float(k) * 16000.0f / n_fft);
                if (x > left && x < right) {
                    weights[m][k] = x <= center ? (x - left) / (center - left) : (right - x) / (right - center);
                    first = std::min(first, k);
                    last = k + 1;
                }
            }
            ranges[m] = {first, last};
        }
    }
};

} // namespace

void xasr_fbank_append(const std::vector<float> & pcm, bool flush, std::vector<float> & frames) {
    const int n = static_cast<int>(pcm.size());
    const int first_frame = static_cast<int>(frames.size() / n_mels);
    const int ready = flush ? (n + hop / 2) / hop : n < win - lead ? 0 : (n - (win - lead)) / hop + 1;
    if (ready <= first_frame) {
        return;
    }

    static const fbank_filters filters;
    frames.resize(size_t(ready) * n_mels);
    for (int t = first_frame; t < ready; ++t) {
        std::array<float, n_fft> re = {};
        std::array<float, n_fft> im = {};
        float dc = 0.0f;
        for (int i = 0; i < win; ++i) {
            int j = t * hop - lead + i;
            while (j < 0 || j >= n) {
                j = j < 0 ? -j - 1 : 2 * n - 1 - j;
            }
            re[i] = pcm[j];
            dc += re[i];
        }
        dc /= win;
        for (int i = 0; i < win; ++i) {
            re[i] -= dc;
        }
        for (int i = win - 1; i > 0; --i) {
            re[i] -= 0.97f * re[i - 1];
        }
        re[0] -= 0.97f * re[0];
        for (int i = 0; i < win; ++i) {
            re[i] *= filters.window[i];
        }
        fft_radix2(re.data(), im.data(), n_fft);
        std::array<float, n_bins> power;
        for (int k = 0; k < n_bins; ++k) {
            power[k] = re[k] * re[k] + im[k] * im[k];
        }
        for (int m = 0; m < n_mels; ++m) {
            float sum = 0.0f;
            for (int k = filters.ranges[m].first; k < filters.ranges[m].second; ++k) {
                sum += power[k] * filters.weights[m][k];
            }
            frames[t * n_mels + m] = std::log(std::max(sum, std::numeric_limits<float>::epsilon()));
        }
    }
}
