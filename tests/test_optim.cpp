#include <vector>

#include "mlcpp/core/matrix.hpp"
#include "mlcpp/dl/nn.hpp"
#include "mlcpp/prob/random.hpp"
#include "mlcpp/dl/sampling.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_optim() {
    std::printf("[test_optim]\n");

    // --- clip_grad_norm ---
    Tensor a(Matrix(1, 2, 0.0));
    a.grad()(0, 0) = 3.0;
    a.grad()(0, 1) = 4.0;  // ||g|| = 5
    std::vector<Tensor> params{a};
    double norm = clip_grad_norm(params, 1.0);
    CHECK_NEAR(norm, 5.0, 1e-9);                         // trả về chuẩn trước khi cắt
    double after = std::sqrt(a.grad()(0, 0) * a.grad()(0, 0) + a.grad()(0, 1) * a.grad()(0, 1));
    CHECK_NEAR(after, 1.0, 1e-6);                        // sau khi cắt: ||g|| ≈ max_norm
    CHECK_NEAR(a.grad()(0, 0) / a.grad()(0, 1), 3.0 / 4.0, 1e-6);  // giữ hướng

    // Không cắt nếu đã nhỏ hơn ngưỡng.
    Tensor b(Matrix(1, 1, 0.0));
    b.grad()(0, 0) = 0.5;
    std::vector<Tensor> p2{b};
    clip_grad_norm(p2, 10.0);
    CHECK_NEAR(b.grad()(0, 0), 0.5, 1e-12);

    // --- cosine_lr ---
    CHECK_NEAR(cosine_lr(0, 1000, 1.0, 100), 0.0, 1e-9);       // bắt đầu warmup từ 0
    CHECK_NEAR(cosine_lr(100, 1000, 1.0, 100), 1.0, 1e-9);     // đỉnh tại hết warmup
    CHECK(cosine_lr(1000, 1000, 1.0, 100) < 0.2);             // cuối lịch: giảm còn ~10%

    // --- argmax_logits ---
    std::vector<double> logits{0.1, 2.5, -1.0, 0.3};
    CHECK(argmax_logits(logits) == 1);

    // --- sample_logits: nhiệt độ rất thấp ~ argmax ---
    Rng rng(1);
    SampleConfig cold;
    cold.temperature = 0.01;
    int hits = 0;
    for (int i = 0; i < 50; ++i)
        if (sample_logits(logits, cold, rng) == 1) ++hits;
    CHECK(hits == 50);

    // --- top_k = 1 luôn chọn token đỉnh ---
    SampleConfig k1;
    k1.top_k = 1;
    bool all_top = true;
    for (int i = 0; i < 50; ++i)
        if (sample_logits(logits, k1, rng) != 1) all_top = false;
    CHECK(all_top);

    // --- không lọc: mọi token đều có thể xuất hiện ---
    std::vector<double> flat{0.0, 0.0, 0.0, 0.0};
    SampleConfig open;
    std::vector<int> seen(4, 0);
    for (int i = 0; i < 4000; ++i) seen[sample_logits(flat, open, rng)] = 1;
    CHECK(seen[0] && seen[1] && seen[2] && seen[3]);
}
