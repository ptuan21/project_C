#pragma once
// mlcpp — lấy mẫu token từ logits theo các chiến lược xác suất.
#include <vector>

#include "mlcpp/prob/random.hpp"

namespace mlcpp {

struct SampleConfig {
    double temperature = 1.0;  // <1 sắc nét hơn (ổn định), >1 ngẫu nhiên hơn
    int top_k = 0;             // 0 = tắt; >0 chỉ giữ k token xác suất cao nhất
    double top_p = 1.0;        // nucleus: giữ tập nhỏ nhất có tổng xác suất >= top_p
};

// Lấy mẫu một chỉ số token từ logits theo cấu hình. Dùng rng để tái lập được.
int sample_logits(const std::vector<double>& logits, const SampleConfig& cfg, Rng& rng);

// Chọn token có logit lớn nhất (greedy / argmax).
int argmax_logits(const std::vector<double>& logits);

}  // namespace mlcpp
