#include "mlcpp/dl/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mlcpp {

int argmax_logits(const std::vector<double>& logits) {
    int best = 0;
    for (std::size_t j = 1; j < logits.size(); ++j)
        if (logits[j] > logits[best]) best = static_cast<int>(j);
    return best;
}

int sample_logits(const std::vector<double>& logits, const SampleConfig& cfg, Rng& rng) {
    std::size_t V = logits.size();
    double t = cfg.temperature <= 0.0 ? 1e-6 : cfg.temperature;

    // softmax có nhiệt độ
    double mx = *std::max_element(logits.begin(), logits.end());
    std::vector<double> p(V);
    double sum = 0.0;
    for (std::size_t j = 0; j < V; ++j) {
        p[j] = std::exp((logits[j] - mx) / t);
        sum += p[j];
    }
    for (std::size_t j = 0; j < V; ++j) p[j] /= sum;

    // Lọc top-k và/hoặc top-p: chỉ giữ các token đứng đầu.
    bool filter = cfg.top_k > 0 || cfg.top_p < 1.0;
    if (filter) {
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });

        std::vector<char> keep(V, 0);
        double cum = 0.0;
        int kept = 0;
        for (int i : idx) {
            keep[i] = 1;
            cum += p[i];
            ++kept;
            if (cfg.top_k > 0 && kept >= cfg.top_k) break;
            if (cfg.top_p < 1.0 && cum >= cfg.top_p) break;
        }
        double s2 = 0.0;
        for (std::size_t j = 0; j < V; ++j) {
            if (!keep[j]) p[j] = 0.0;
            s2 += p[j];
        }
        for (std::size_t j = 0; j < V; ++j) p[j] /= s2;  // chuẩn hóa lại
    }

    // Lấy mẫu theo phân phối tích lũy.
    double r = rng.uniform();
    double acc = 0.0;
    for (std::size_t j = 0; j < V; ++j) {
        acc += p[j];
        if (r <= acc) return static_cast<int>(j);
    }
    return static_cast<int>(V - 1);
}

}  // namespace mlcpp
