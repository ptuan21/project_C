// Demo Tầng 3: linear regression (2 cách) + logistic regression.
#include <cstdio>

#include "mlcpp/linreg.hpp"
#include "mlcpp/logreg.hpp"
#include "mlcpp/matrix.hpp"
#include "mlcpp/random.hpp"

using namespace mlcpp;

int main() {
    std::printf("=== mlcpp demo: Tầng 3 (ML cổ điển) ===\n\n");

    Rng rng(2026);

    // 1) Linear regression trên dữ liệu y = 4*x + 7 + nhiễu
    const int N = 200;
    Matrix X(N, 1, 0.0), y(N, 1, 0.0);
    for (int i = 0; i < N; ++i) {
        double x = rng.uniform(-5.0, 5.0);
        X(i, 0) = x;
        y(i, 0) = 4.0 * x + 7.0 + rng.gaussian(0.0, 1.0);
    }

    LinearRegression ne;
    ne.fit_normal_equation(X, y);
    std::printf("[Linear] Normal equation:    bias=%.3f  slope=%.3f  mse=%.4f\n",
                ne.weights()(0, 0), ne.weights()(1, 0), ne.mse(X, y));

    LinearRegression gd;
    gd.fit_gradient_descent(X, y, 0.02, 5000);
    std::printf("[Linear] Gradient descent:   bias=%.3f  slope=%.3f  mse=%.4f\n",
                gd.weights()(0, 0), gd.weights()(1, 0), gd.mse(X, y));
    std::printf("         (sự thật: bias=7.0, slope=4.0)\n\n");

    // 2) Logistic regression: hai cụm tách tuyến tính được
    const int M = 500;
    Matrix Xc(M, 2, 0.0), yc(M, 1, 0.0);
    for (int i = 0; i < M; ++i) {
        int label = i % 2;
        double c = label ? 2.0 : -2.0;
        Xc(i, 0) = c + rng.gaussian(0.0, 0.8);
        Xc(i, 1) = c + rng.gaussian(0.0, 0.8);
        yc(i, 0) = static_cast<double>(label);
    }
    LogisticRegression clf;
    clf.fit(Xc, yc, 0.1, 3000);
    std::printf("[Logistic] accuracy = %.1f%%\n", 100.0 * clf.accuracy(Xc, yc));

    // Dự đoán vài điểm mới
    Matrix probe = Matrix::from_rows({{2.0, 2.0}, {-2.0, -2.0}, {0.0, 0.0}});
    Matrix p = clf.predict_proba(probe);
    for (std::size_t i = 0; i < probe.rows(); ++i)
        std::printf("   P(y=1 | (%.1f, %.1f)) = %.3f\n", probe(i, 0), probe(i, 1), p(i, 0));

    return 0;
}
