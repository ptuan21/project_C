#pragma once
// mlcpp — autograd engine: vi phân tự động ngược (reverse-mode) trên Matrix.
// Đồ thị tính toán được dựng động mỗi lần forward (kiểu PyTorch define-by-run).
#include <functional>
#include <memory>
#include <vector>

#include "mlcpp/matrix.hpp"

namespace mlcpp {

// Một nút trong đồ thị tính toán: giá trị, gradient tích lũy, và cách lan ngược.
struct Node {
    Matrix value;
    Matrix grad;  // cùng kích thước value, khởi tạo 0
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward_fn;  // cộng dồn grad vào các parent

    explicit Node(Matrix v) : value(std::move(v)), grad(value.rows(), value.cols(), 0.0) {}
};

// Lớp bọc nhẹ quanh shared_ptr<Node> để dùng cho tiện.
class Tensor {
public:
    std::shared_ptr<Node> node;

    Tensor() = default;
    explicit Tensor(Matrix v) : node(std::make_shared<Node>(std::move(v))) {}

    Matrix& value() const { return node->value; }
    Matrix& grad() const { return node->grad; }
    std::size_t rows() const { return node->value.rows(); }
    std::size_t cols() const { return node->value.cols(); }

    void zero_grad() const { node->grad.fill(0.0); }

    // Lan ngược từ một tensor vô hướng (1x1), thường là loss.
    void backward() const;
};

// --- Các phép có đạo hàm ---
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);  // b có thể là (1 x cols) -> broadcast bias
Tensor relu(const Tensor& a);

// Softmax + cross-entropy gộp (ổn định số học). labels là chỉ số lớp [0..C-1].
// Trả về loss vô hướng (1x1) đã lấy trung bình theo batch.
Tensor cross_entropy(const Tensor& logits, const std::vector<int>& labels);

}  // namespace mlcpp
