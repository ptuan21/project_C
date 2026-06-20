#include <functional>
#include <vector>

#include "mlcpp/dl/autograd.hpp"
#include "mlcpp/dl/nn.hpp"
#include "mlcpp/prob/random.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_autograd() {
    std::printf("[test_autograd]\n");

    Rng rng(123);

    // Mạng nhỏ 2 -> 4 -> 3 để kiểm tra toàn bộ chuỗi autograd.
    Linear l1(2, 4, rng);
    Linear l2(4, 3, rng);

    Matrix Xv = Matrix::from_rows({{0.5, -1.0}, {1.0, 2.0}, {-1.5, 0.3}, {0.2, 0.7}});
    std::vector<int> labels{0, 2, 1, 0};

    auto forward_loss = [&]() -> Tensor {
        Tensor x(Xv);
        Tensor h = relu(l1.forward(x));
        Tensor o = l2.forward(h);
        return cross_entropy(o, labels);
    };

    // Gradient giải tích từ backward.
    for (auto& p : l1.params()) p.zero_grad();
    for (auto& p : l2.params()) p.zero_grad();
    Tensor loss = forward_loss();
    loss.backward();

    // Kiểm tra bằng sai phân hữu hạn trung tâm trên một số tham số của W1.
    Tensor W1 = l1.params()[0];
    const double eps = 1e-4;
    int checked = 0;
    for (std::size_t i = 0; i < W1.rows() && checked < 5; ++i)
        for (std::size_t j = 0; j < W1.cols() && checked < 5; ++j) {
            double orig = W1.value()(i, j);
            W1.value()(i, j) = orig + eps;
            double lp = forward_loss().value()(0, 0);
            W1.value()(i, j) = orig - eps;
            double lm = forward_loss().value()(0, 0);
            W1.value()(i, j) = orig;
            double numeric = (lp - lm) / (2.0 * eps);
            CHECK_NEAR(numeric, W1.grad()(i, j), 1e-4);
            ++checked;
        }

    // Kiểm tra hội tụ: học XOR (bài toán phi tuyến kinh điển) bằng Adam.
    Matrix Xx = Matrix::from_rows({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    std::vector<int> yx{0, 1, 1, 0};
    Linear h1(2, 8, rng), h2(8, 2, rng);
    std::vector<Tensor> params;
    for (auto& p : h1.params()) params.push_back(p);
    for (auto& p : h2.params()) params.push_back(p);
    Adam opt(params, 0.05);

    double final_loss = 0.0;
    for (int e = 0; e < 3000; ++e) {
        opt.zero_grad();
        Tensor x(Xx);
        Tensor out = h2.forward(relu(h1.forward(x)));
        Tensor l = cross_entropy(out, yx);
        l.backward();
        opt.step();
        final_loss = l.value()(0, 0);
    }
    CHECK(final_loss < 0.05);  // mạng phải học được XOR
}
