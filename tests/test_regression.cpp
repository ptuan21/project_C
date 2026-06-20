#include "mlcpp/ml/linreg.hpp"
#include "mlcpp/ml/logreg.hpp"
#include "mlcpp/core/matrix.hpp"
#include "mlcpp/prob/random.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_regression() {
    std::printf("[test_regression]\n");

    // --- Linear regression: dữ liệu khớp chính xác y = 2*x1 + 3*x2 + 1 ---
    Matrix X = Matrix::from_rows({{1, 1}, {2, 1}, {1, 2}, {3, 2}, {2, 3}, {0, 0}});
    Matrix y(X.rows(), 1, 0.0);
    for (std::size_t i = 0; i < X.rows(); ++i)
        y(i, 0) = 2.0 * X(i, 0) + 3.0 * X(i, 1) + 1.0;

    LinearRegression lr;
    lr.fit_normal_equation(X, y);
    CHECK_NEAR(lr.weights()(0, 0), 1.0, 1e-6);  // bias
    CHECK_NEAR(lr.weights()(1, 0), 2.0, 1e-6);  // hệ số x1
    CHECK_NEAR(lr.weights()(2, 0), 3.0, 1e-6);  // hệ số x2
    CHECK(lr.mse(X, y) < 1e-12);

    // Gradient descent phải hội tụ gần nghiệm chuẩn.
    LinearRegression gd;
    gd.fit_gradient_descent(X, y, 0.05, 20000);
    CHECK_NEAR(gd.weights()(1, 0), 2.0, 0.05);
    CHECK_NEAR(gd.weights()(2, 0), 3.0, 0.05);
    CHECK(gd.mse(X, y) < 1e-3);

    // --- Logistic regression: phân loại nhị phân tách tuyến tính được ---
    Rng rng(7);
    const int N = 400;
    Matrix Xc(N, 2, 0.0);
    Matrix yc(N, 1, 0.0);
    for (int i = 0; i < N; ++i) {
        int label = i % 2;
        // lớp 1 quanh (2,2), lớp 0 quanh (-2,-2)
        double cx = label ? 2.0 : -2.0;
        Xc(i, 0) = cx + rng.gaussian(0.0, 0.6);
        Xc(i, 1) = cx + rng.gaussian(0.0, 0.6);
        yc(i, 0) = static_cast<double>(label);
    }
    LogisticRegression clf;
    clf.fit(Xc, yc, 0.1, 2000);
    CHECK(clf.accuracy(Xc, yc) > 0.95);
}
