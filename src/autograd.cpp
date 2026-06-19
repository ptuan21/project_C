#include "mlcpp/autograd.hpp"

#include <cmath>
#include <unordered_set>

namespace mlcpp {

// dst += src (cộng dồn gradient theo từng phần tử).
static void accumulate(Matrix& dst, const Matrix& src) {
    for (std::size_t i = 0; i < dst.rows(); ++i)
        for (std::size_t j = 0; j < dst.cols(); ++j) dst(i, j) += src(i, j);
}

void Tensor::backward() const {
    // Sắp xếp topo các nút bằng DFS.
    std::vector<Node*> topo;
    std::unordered_set<Node*> visited;
    std::function<void(Node*)> build = [&](Node* n) {
        if (visited.count(n)) return;
        visited.insert(n);
        for (auto& p : n->parents) build(p.get());
        topo.push_back(n);
    };
    build(node.get());

    // Seed: d(loss)/d(loss) = 1 (loss là vô hướng 1x1).
    node->grad.fill(0.0);
    node->grad(0, 0) = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        if ((*it)->backward_fn) (*it)->backward_fn();
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    Tensor out(a.value().matmul(b.value()));
    out.node->parents = {a.node, b.node};
    Node* an = a.node.get();
    Node* bn = b.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [an, bn, on]() {
        // dA += dOut · Bᵀ ; dB += Aᵀ · dOut
        Matrix da = on->grad.matmul(bn->value.transpose());
        accumulate(an->grad, da);
        Matrix db = an->value.transpose().matmul(on->grad);
        accumulate(bn->grad, db);
    };
    return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
    bool broadcast = (b.rows() == 1 && a.rows() != 1);  // b là vector bias hàng
    Matrix v(a.rows(), a.cols(), 0.0);
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j)
            v(i, j) = a.value()(i, j) + b.value()(broadcast ? 0 : i, j);

    Tensor out(std::move(v));
    out.node->parents = {a.node, b.node};
    Node* an = a.node.get();
    Node* bn = b.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [an, bn, on, broadcast]() {
        accumulate(an->grad, on->grad);
        if (broadcast) {
            // Gộp gradient theo các hàng về vector bias.
            for (std::size_t j = 0; j < on->grad.cols(); ++j) {
                double s = 0.0;
                for (std::size_t i = 0; i < on->grad.rows(); ++i) s += on->grad(i, j);
                bn->grad(0, j) += s;
            }
        } else {
            accumulate(bn->grad, on->grad);
        }
    };
    return out;
}

Tensor relu(const Tensor& a) {
    Matrix v(a.rows(), a.cols(), 0.0);
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) {
            double x = a.value()(i, j);
            v(i, j) = x > 0.0 ? x : 0.0;
        }
    Tensor out(std::move(v));
    out.node->parents = {a.node};
    Node* an = a.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [an, on]() {
        for (std::size_t i = 0; i < an->grad.rows(); ++i)
            for (std::size_t j = 0; j < an->grad.cols(); ++j)
                if (an->value(i, j) > 0.0) an->grad(i, j) += on->grad(i, j);
    };
    return out;
}

Tensor cross_entropy(const Tensor& logits, const std::vector<int>& labels) {
    std::size_t n = logits.rows();
    std::size_t C = logits.cols();
    Matrix probs(n, C, 0.0);
    double loss = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        double mx = logits.value()(i, 0);  // trừ max để ổn định số học
        for (std::size_t j = 1; j < C; ++j) mx = std::max(mx, logits.value()(i, j));
        double sum = 0.0;
        for (std::size_t j = 0; j < C; ++j) {
            double e = std::exp(logits.value()(i, j) - mx);
            probs(i, j) = e;
            sum += e;
        }
        for (std::size_t j = 0; j < C; ++j) probs(i, j) /= sum;
        loss += -std::log(probs(i, static_cast<std::size_t>(labels[i])) + 1e-12);
    }
    loss /= static_cast<double>(n);

    Tensor out(Matrix(1, 1, loss));
    out.node->parents = {logits.node};
    Node* ln = logits.node.get();
    Node* on = out.node.get();
    // probs và labels được giữ lại trong lambda để dùng khi lan ngược.
    out.node->backward_fn = [ln, on, probs, labels, n, C]() {
        double scale = on->grad(0, 0) / static_cast<double>(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < C; ++j) {
                double y = (static_cast<int>(j) == labels[i]) ? 1.0 : 0.0;
                ln->grad(i, j) += (probs(i, j) - y) * scale;
            }
    };
    return out;
}

}  // namespace mlcpp
