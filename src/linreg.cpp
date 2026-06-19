#include "mlcpp/linreg.hpp"

#include <stdexcept>

#include "mlcpp/linalg.hpp"

namespace mlcpp {

// Thêm cột toàn số 1 ở đầu để biểu diễn bias: [1 | X].
static Matrix augment_bias(const Matrix& X) {
    Matrix A(X.rows(), X.cols() + 1, 1.0);
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) A(i, j + 1) = X(i, j);
    return A;
}

void LinearRegression::fit_normal_equation(const Matrix& X, const Matrix& y) {
    if (y.rows() != X.rows() || y.cols() != 1)
        throw std::invalid_argument("fit_normal_equation: y phải là (n_samples x 1)");
    Matrix A = augment_bias(X);
    Matrix At = A.transpose();
    Matrix AtA = At.matmul(A);   // (d+1) x (d+1)
    Matrix Aty = At.matmul(y);   // (d+1) x 1
    w_ = solve(AtA, Aty);        // giải (AᵀA) w = Aᵀy
}

void LinearRegression::fit_gradient_descent(const Matrix& X, const Matrix& y, double lr,
                                            int epochs) {
    if (y.rows() != X.rows() || y.cols() != 1)
        throw std::invalid_argument("fit_gradient_descent: y phải là (n_samples x 1)");
    Matrix A = augment_bias(X);
    Matrix At = A.transpose();
    std::size_t n = A.rows();
    w_ = Matrix(A.cols(), 1, 0.0);

    double scale = 1.0 / static_cast<double>(n);
    for (int e = 0; e < epochs; ++e) {
        Matrix pred = A.matmul(w_);            // n x 1
        Matrix err = pred - y;                  // n x 1
        Matrix grad = At.matmul(err) * scale;   // (d+1) x 1
        w_ = w_ - grad * lr;
    }
}

Matrix LinearRegression::predict(const Matrix& X) const {
    if (w_.rows() != X.cols() + 1)
        throw std::runtime_error("predict: mô hình chưa được huấn luyện hoặc sai số chiều");
    return augment_bias(X).matmul(w_);
}

double LinearRegression::mse(const Matrix& X, const Matrix& y) const {
    Matrix pred = predict(X);
    double s = 0.0;
    for (std::size_t i = 0; i < y.rows(); ++i) {
        double e = pred(i, 0) - y(i, 0);
        s += e * e;
    }
    return s / static_cast<double>(y.rows());
}

}  // namespace mlcpp
