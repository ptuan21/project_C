#pragma once
// mlcpp — hồi quy logistic (logistic regression) cho phân loại nhị phân.
// P(y=1 | x) = sigmoid(X * w + b). Huấn luyện bằng gradient descent.
#include "mlcpp/matrix.hpp"

namespace mlcpp {

class LogisticRegression {
public:
    void fit(const Matrix& X, const Matrix& y, double lr = 0.1, int epochs = 1000);

    Matrix predict_proba(const Matrix& X) const;         // xác suất, (n_samples x 1)
    Matrix predict(const Matrix& X, double thr = 0.5) const;  // nhãn 0/1
    double accuracy(const Matrix& X, const Matrix& y, double thr = 0.5) const;

    const Matrix& weights() const { return w_; }

private:
    Matrix w_;  // (n_features+1) x 1, w_(0,0) là bias
};

}  // namespace mlcpp
