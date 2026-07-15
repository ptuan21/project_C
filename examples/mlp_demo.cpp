#include <cmath>
#include <cstdio>
#include <vector>

#include "mlcpp/dl/autograd.hpp"
#include "mlcpp/dl/nn.hpp"
#include "mlcpp/prob/random.hpp"

using namespace mlcpp;

// Sinh dữ liệu: lớp 0 = vòng trong (bán kính nhỏ), lớp 1 = vành ngoài.
static void make_rings(Rng& rng, int n, Matrix& X, std::vector<int>& y) {
    X = Matrix(n, 2, 0.0);
    y.resize(n);
    for (int i = 0; i < n; ++i) {
        int label = i % 2;
        double r = label ? rng.uniform(1.5, 2.5) : rng.uniform(0.0, 1.0);
        double a = rng.uniform(0.0, 2.0 * M_PI);
        X(i, 0) = r * std::cos(a) + rng.gaussian(0.0, 0.05);
        X(i, 1) = r * std::sin(a) + rng.gaussian(0.0, 0.05);
        y[i] = label;
    }
}

static double accuracy(Linear& l1, Linear& l2, const Matrix& X, const std::vector<int>& y) {
    Tensor x(X);
    Tensor out = l2.forward(relu(l1.forward(x)));
    int correct = 0;
    for (std::size_t i = 0; i < X.rows(); ++i) {
        int pred = out.value()(i, 0) >= out.value()(i, 1) ? 0 : 1;
        if (pred == y[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(X.rows());
}

int main() {
    std::printf("=== mlcpp demo: Tầng 4 (MLP + autograd) ===\n\n");
    Rng rng(2026);

    Matrix Xtr, Xte;
    std::vector<int> ytr, yte;
    make_rings(rng, 600, Xtr, ytr);  // tập huấn luyện
    make_rings(rng, 200, Xte, yte);  // tập kiểm tra (chưa thấy khi train)

    // MLP: 2 -> 16 -> 2
    Linear l1(2, 16, rng), l2(16, 2, rng);
    std::vector<Tensor> params;
    for (auto& p : l1.params()) params.push_back(p);
    for (auto& p : l2.params()) params.push_back(p);
    Adam opt(params, 0.01);

    std::printf("epoch |   loss | train acc | test acc\n");
    for (int e = 1; e <= 400; ++e) {
        opt.zero_grad();
        Tensor x(Xtr);
        Tensor out = l2.forward(relu(l1.forward(x)));
        Tensor loss = cross_entropy(out, ytr);
        loss.backward();
        opt.step();

        if (e % 50 == 0 || e == 1)
            std::printf("%5d | %.4f |   %.3f   |  %.3f\n", e, loss.value()(0, 0),
                        accuracy(l1, l2, Xtr, ytr), accuracy(l1, l2, Xte, yte));
    }

    std::printf("\nMLP học được ranh giới phi tuyến mà logistic regression tuyến tính không thể.\n");
    std::printf("Train acc ~ test acc => mô hình tổng quát hóa tốt, không overfit.\n");
    return 0;
}
