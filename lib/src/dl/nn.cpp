#include "mlcpp/dl/nn.hpp"

#include <algorithm>
#include <cmath>

#include "mlcpp/core/quantize.hpp"

namespace mlcpp {

QuantStats quantize_params_in_place(const std::vector<Tensor>& params) {
    QuantStats st;
    for (const auto& p : params) {
        Matrix& W = p.value();
        QuantizedMatrix qm = quantize_per_col(W);
        Matrix deq = dequantize(qm);
        for (std::size_t i = 0; i < W.rows(); ++i)
            for (std::size_t j = 0; j < W.cols(); ++j) W(i, j) = deq(i, j);
        st.float_bytes += W.size() * 4;  // float32
        st.int8_bytes += qm.bytes();     // int8 + scale
    }
    return st;
}

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

Adam::Adam(std::vector<Tensor> params, double lr, double beta1, double beta2, double eps,
           double weight_decay)
    : params_(std::move(params)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      wd_(weight_decay) {
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
                // AdamW: suy giảm trọng số tách rời khỏi bước Adam.
                w(i, j) -= lr_ * (mhat / (std::sqrt(vhat) + eps_) + wd_ * w(i, j));
            }
    }
}

double clip_grad_norm(const std::vector<Tensor>& params, double max_norm) {
    double total = 0.0;
    for (const auto& p : params) {
        const Matrix& g = p.grad();
        for (std::size_t i = 0; i < g.rows(); ++i)
            for (std::size_t j = 0; j < g.cols(); ++j) total += g(i, j) * g(i, j);
    }
    total = std::sqrt(total);
    if (total > max_norm) {
        double scale = max_norm / (total + 1e-6);
        for (auto& p : params) {
            Matrix& g = p.grad();
            for (std::size_t i = 0; i < g.rows(); ++i)
                for (std::size_t j = 0; j < g.cols(); ++j) g(i, j) *= scale;
        }
    }
    return total;
}

double cosine_lr(int step, int total_steps, double base_lr, int warmup_steps) {
    if (step < warmup_steps) return base_lr * static_cast<double>(step) / std::max(1, warmup_steps);
    double progress = static_cast<double>(step - warmup_steps) /
                      std::max(1, total_steps - warmup_steps);
    if (progress > 1.0) progress = 1.0;
    // Giảm cosine từ base_lr về 10% base_lr.
    double cos = 0.5 * (1.0 + std::cos(3.14159265358979 * progress));
    return base_lr * (0.1 + 0.9 * cos);
}

}  // namespace mlcpp
