#pragma once
// mlcpp — hồi quy tuyến tính (linear regression).
// y ≈ X * w + b. Bias được xử lý nội bộ bằng cách thêm một cột toàn số 1.
#include "mlcpp/matrix.hpp"

namespace mlcpp {

class LinearRegression {
public:
    // Nghiệm chuẩn (normal equation): w = (Xᵀ X)⁻¹ Xᵀ y. Chính xác, không cần lặp.
    void fit_normal_equation(const Matrix& X, const Matrix& y);

    // Gradient descent: tối thiểu hóa MSE qua nhiều epoch.
    void fit_gradient_descent(const Matrix& X, const Matrix& y, double lr = 0.01,
                              int epochs = 1000);

    Matrix predict(const Matrix& X) const;  // trả về (n_samples x 1)
    double mse(const Matrix& X, const Matrix& y) const;

    const Matrix& weights() const { return w_; }  // (n_features+1) x 1, w_(0,0) là bias

private:
    Matrix w_;  // trọng số kèm bias ở hàng 0
};

}  // namespace mlcpp
