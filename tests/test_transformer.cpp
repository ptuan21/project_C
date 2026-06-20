#include <vector>

#include "mlcpp/dl/nn.hpp"
#include "mlcpp/prob/random.hpp"
#include "mlcpp/models/transformer.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_transformer() {
    std::printf("[test_transformer]\n");

    Rng rng(7);
    GPTConfig cfg{5, 8, 2, 1, 6};  // vocab nhỏ, 2 đầu attention, để kiểm tra nhanh
    GPT model(cfg, rng);

    std::vector<int> ids{1, 3, 2, 4};
    std::vector<int> targets{3, 2, 4, 0};
    auto loss_fn = [&]() -> Tensor { return cross_entropy(model.forward(ids), targets); };

    // Gradient giải tích.
    for (auto& p : model.params()) p.zero_grad();
    Tensor loss = loss_fn();
    CHECK(loss.value()(0, 0) > 0.0);
    loss.backward();

    // Kiểm tra bằng sai phân hữu hạn trên tok_emb (params[0]) — nhờ weight tying nó
    // được dùng ở CẢ embedding đầu vào lẫn chiếu đầu ra, nên xác thực đồng thời:
    // embedding, layernorm, attention(softmax/transpose/mul), gelu, residual, tied head, cross_entropy.
    Tensor tok_emb = model.params()[0];
    const double eps = 1e-4;
    int checked = 0;
    for (std::size_t i = 0; i < tok_emb.rows() && checked < 6; ++i)
        for (std::size_t j = 0; j < tok_emb.cols() && checked < 6; ++j) {
            double orig = tok_emb.value()(i, j);
            tok_emb.value()(i, j) = orig + eps;
            double lp = loss_fn().value()(0, 0);
            tok_emb.value()(i, j) = orig - eps;
            double lm = loss_fn().value()(0, 0);
            tok_emb.value()(i, j) = orig;
            double numeric = (lp - lm) / (2.0 * eps);
            CHECK_NEAR(numeric, tok_emb.grad()(i, j), 1e-4);
            ++checked;
        }

    // Kiểm tra hội tụ: GPT nhỏ phải ghi nhớ được một chuỗi ngắn.
    std::vector<int> seq{1, 2, 3, 4, 0, 1};  // input (độ dài <= block_size)
    std::vector<int> tgt{2, 3, 4, 0, 1, 2};  // dịch 1
    Adam opt(model.params(), 0.01);
    double first = 0.0, last = 0.0;
    for (int e = 0; e < 300; ++e) {
        opt.zero_grad();
        Tensor l = cross_entropy(model.forward(seq), tgt);
        l.backward();
        opt.step();
        if (e == 0) first = l.value()(0, 0);
        last = l.value()(0, 0);
    }
    CHECK(last < first * 0.2);  // loss giảm mạnh => mô hình học được
}
