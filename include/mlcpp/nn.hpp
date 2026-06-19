#pragma once
// mlcpp — lớp mạng (layers) và bộ tối ưu (optimizers) cho mini DL framework.
#include <vector>

#include "mlcpp/autograd.hpp"
#include "mlcpp/random.hpp"

namespace mlcpp {

// Lớp tuyến tính (fully-connected): y = x · W + b.
class Linear {
public:
    Linear(std::size_t in_features, std::size_t out_features, Rng& rng);
    Tensor forward(const Tensor& x) const;
    std::vector<Tensor> params() const { return {W_, b_}; }

private:
    Tensor W_;  // in x out, khởi tạo He
    Tensor b_;  // 1 x out, khởi tạo 0
};

// SGD đơn giản: w -= lr * grad.
class SGD {
public:
    SGD(std::vector<Tensor> params, double lr) : params_(std::move(params)), lr_(lr) {}
    void zero_grad();
    void step();

private:
    std::vector<Tensor> params_;
    double lr_;
};

// Adam: thích nghi tốc độ học theo moment bậc 1 và 2.
class Adam {
public:
    Adam(std::vector<Tensor> params, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999,
         double eps = 1e-8);
    void zero_grad();
    void step();

private:
    std::vector<Tensor> params_;
    std::vector<Matrix> m_, v_;  // moment cho từng tham số
    double lr_, beta1_, beta2_, eps_;
    int t_ = 0;
};

}  // namespace mlcpp
