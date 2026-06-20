#pragma once
// mlcpp — lớp mạng (layers) và bộ tối ưu (optimizers) cho mini DL framework.
#include <vector>

#include "mlcpp/dl/autograd.hpp"
#include "mlcpp/prob/random.hpp"

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

// Adam / AdamW: thích nghi tốc độ học theo moment bậc 1 và 2.
// weight_decay > 0 bật AdamW (suy giảm trọng số tách rời) giúp ổn định & chống overfit.
class Adam {
public:
    Adam(std::vector<Tensor> params, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999,
         double eps = 1e-8, double weight_decay = 0.0);
    void zero_grad();
    void step();

    void set_lr(double lr) { lr_ = lr; }  // đổi tốc độ học giữa chừng (cho LR schedule)
    double lr() const { return lr_; }

private:
    std::vector<Tensor> params_;
    std::vector<Matrix> m_, v_;  // moment cho từng tham số
    double lr_, beta1_, beta2_, eps_, wd_;
    int t_ = 0;
};

// Cắt gradient theo chuẩn L2 toàn cục: nếu ||g|| > max_norm thì co lại.
// Trả về chuẩn gradient trước khi cắt. Giúp huấn luyện ổn định, tránh "nổ" gradient.
double clip_grad_norm(const std::vector<Tensor>& params, double max_norm);

// Lịch học rate: warmup tuyến tính rồi giảm theo cosine về ~0.
double cosine_lr(int step, int total_steps, double base_lr, int warmup_steps);

}  // namespace mlcpp
