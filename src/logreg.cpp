#include "mlcpp/logreg.hpp"

#include <cmath>
#include <stdexcept>

namespace mlcpp {

static Matrix augment_bias(const Matrix& X) {
    Matrix A(X.rows(), X.cols() + 1, 1.0);
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) A(i, j + 1) = X(i, j);
    return A;
}

static double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

void LogisticRegression::fit(const Matrix& X, const Matrix& y, double lr, int epochs) {
    if (y.rows() != X.rows() || y.cols() != 1)
        throw std::invalid_argument("fit: y phải là (n_samples x 1)");
    Matrix A = augment_bias(X);
    Matrix At = A.transpose();
    std::size_t n = A.rows();
    w_ = Matrix(A.cols(), 1, 0.0);

    double scale = 1.0 / static_cast<double>(n);
    for (int e = 0; e < epochs; ++e) {
        Matrix z = A.matmul(w_);  // n x 1
        // p = sigmoid(z); err = p - y  — cùng dạng gradient với linear regression.
        Matrix err(n, 1, 0.0);
        for (std::size_t i = 0; i < n; ++i) err(i, 0) = sigmoid(z(i, 0)) - y(i, 0);
        Matrix grad = At.matmul(err) * scale;
        w_ = w_ - grad * lr;
    }
}

Matrix LogisticRegression::predict_proba(const Matrix& X) const {
    if (w_.rows() != X.cols() + 1)
        throw std::runtime_error("predict_proba: mô hình chưa huấn luyện hoặc sai số chiều");
    Matrix z = augment_bias(X).matmul(w_);
    Matrix p(z.rows(), 1, 0.0);
    for (std::size_t i = 0; i < z.rows(); ++i) p(i, 0) = sigmoid(z(i, 0));
    return p;
}

Matrix LogisticRegression::predict(const Matrix& X, double thr) const {
    Matrix p = predict_proba(X);
    Matrix out(p.rows(), 1, 0.0);
    for (std::size_t i = 0; i < p.rows(); ++i) out(i, 0) = p(i, 0) >= thr ? 1.0 : 0.0;
    return out;
}

double LogisticRegression::accuracy(const Matrix& X, const Matrix& y, double thr) const {
    Matrix pred = predict(X, thr);
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y.rows(); ++i)
        if (pred(i, 0) == y(i, 0)) ++correct;
    return static_cast<double>(correct) / static_cast<double>(y.rows());
}

}  // namespace mlcpp
