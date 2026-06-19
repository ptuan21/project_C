#include "mlcpp/nn.hpp"

#include <cmath>

namespace mlcpp {

Linear::Linear(std::size_t in_features, std::size_t out_features, Rng& rng)
    : W_(Matrix(in_features, out_features, 0.0)), b_(Matrix(1, out_features, 0.0)) {
    // Khởi tạo He: phù hợp với ReLU, giữ phương sai tín hiệu ổn định.
    double std = std::sqrt(2.0 / static_cast<double>(in_features));
    for (std::size_t i = 0; i < in_features; ++i)
        for (std::size_t j = 0; j < out_features; ++j) W_.value()(i, j) = rng.gaussian(0.0, std);
}

Tensor Linear::forward(const Tensor& x) const { return add(matmul(x, W_), b_); }

void SGD::zero_grad() {
    for (auto& p : params_) p.zero_grad();
}

void SGD::step() {
    for (auto& p : params_)
        for (std::size_t i = 0; i < p.value().rows(); ++i)
            for (std::size_t j = 0; j < p.value().cols(); ++j)
                p.value()(i, j) -= lr_ * p.grad()(i, j);
}

Adam::Adam(std::vector<Tensor> params, double lr, double beta1, double beta2, double eps)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (auto& p : params_) {
        m_.emplace_back(p.value().rows(), p.value().cols(), 0.0);
        v_.emplace_back(p.value().rows(), p.value().cols(), 0.0);
    }
}

void Adam::zero_grad() {
    for (auto& p : params_) p.zero_grad();
}

void Adam::step() {
    ++t_;
    double bc1 = 1.0 - std::pow(beta1_, t_);  // hiệu chỉnh thiên lệch
    double bc2 = 1.0 - std::pow(beta2_, t_);
    for (std::size_t k = 0; k < params_.size(); ++k) {
        Matrix& w = params_[k].value();
        Matrix& g = params_[k].grad();
        Matrix& m = m_[k];
        Matrix& v = v_[k];
        for (std::size_t i = 0; i < w.rows(); ++i)
            for (std::size_t j = 0; j < w.cols(); ++j) {
                double gr = g(i, j);
                m(i, j) = beta1_ * m(i, j) + (1.0 - beta1_) * gr;
                v(i, j) = beta2_ * v(i, j) + (1.0 - beta2_) * gr * gr;
                double mhat = m(i, j) / bc1;
                double vhat = v(i, j) / bc2;
                w(i, j) -= lr_ * mhat / (std::sqrt(vhat) + eps_);
            }
    }
}

}  // namespace mlcpp
