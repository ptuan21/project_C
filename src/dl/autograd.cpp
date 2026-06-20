#include "mlcpp/dl/autograd.hpp"

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

Tensor mul(const Tensor& a, double s) {
    Tensor out(a.value() * s);
    out.node->parents = {a.node};
    Node* an = a.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [an, on, s]() {
        for (std::size_t i = 0; i < an->grad.rows(); ++i)
            for (std::size_t j = 0; j < an->grad.cols(); ++j) an->grad(i, j) += on->grad(i, j) * s;
    };
    return out;
}

Tensor transpose(const Tensor& a) {
    Tensor out(a.value().transpose());
    out.node->parents = {a.node};
    Node* an = a.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [an, on]() {
        // d(aᵀ)/da: cộng dồn theo chuyển vị của gradient ra.
        for (std::size_t i = 0; i < an->grad.rows(); ++i)
            for (std::size_t j = 0; j < an->grad.cols(); ++j) an->grad(i, j) += on->grad(j, i);
    };
    return out;
}

Tensor softmax_rows(const Tensor& a) {
    std::size_t n = a.rows(), C = a.cols();
    Matrix s(n, C, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double mx = a.value()(i, 0);
        for (std::size_t j = 1; j < C; ++j) mx = std::max(mx, a.value()(i, j));
        double sum = 0.0;
        for (std::size_t j = 0; j < C; ++j) {
            double e = std::exp(a.value()(i, j) - mx);
            s(i, j) = e;
            sum += e;
        }
        for (std::size_t j = 0; j < C; ++j) s(i, j) /= sum;
    }
    Tensor out(s);
    out.node->parents = {a.node};
    Node* an = a.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [an, on, s, n, C]() {
        // dx_i = s_i * (g_i - Σ_j g_j s_j) cho từng hàng.
        for (std::size_t i = 0; i < n; ++i) {
            double dot = 0.0;
            for (std::size_t j = 0; j < C; ++j) dot += on->grad(i, j) * s(i, j);
            for (std::size_t j = 0; j < C; ++j)
                an->grad(i, j) += s(i, j) * (on->grad(i, j) - dot);
        }
    };
    return out;
}

Tensor gelu(const Tensor& a) {
    const double inv_sqrt2 = 0.7071067811865476;
    const double inv_sqrt2pi = 0.3989422804014327;
    Matrix v(a.rows(), a.cols(), 0.0);
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) {
            double x = a.value()(i, j);
            v(i, j) = 0.5 * x * (1.0 + std::erf(x * inv_sqrt2));
        }
    Tensor out(std::move(v));
    out.node->parents = {a.node};
    Node* an = a.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [an, on, inv_sqrt2, inv_sqrt2pi]() {
        for (std::size_t i = 0; i < an->grad.rows(); ++i)
            for (std::size_t j = 0; j < an->grad.cols(); ++j) {
                double x = an->value(i, j);
                double cdf = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
                double pdf = inv_sqrt2pi * std::exp(-0.5 * x * x);
                an->grad(i, j) += on->grad(i, j) * (cdf + x * pdf);
            }
    };
    return out;
}

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps) {
    std::size_t n = x.rows(), D = x.cols();
    Matrix xhat(n, D, 0.0);
    std::vector<double> inv_std(n, 0.0);
    Matrix v(n, D, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double mu = 0.0;
        for (std::size_t j = 0; j < D; ++j) mu += x.value()(i, j);
        mu /= static_cast<double>(D);
        double var = 0.0;
        for (std::size_t j = 0; j < D; ++j) {
            double d = x.value()(i, j) - mu;
            var += d * d;
        }
        var /= static_cast<double>(D);
        double inv = 1.0 / std::sqrt(var + eps);
        inv_std[i] = inv;
        for (std::size_t j = 0; j < D; ++j) {
            xhat(i, j) = (x.value()(i, j) - mu) * inv;
            v(i, j) = gamma.value()(0, j) * xhat(i, j) + beta.value()(0, j);
        }
    }
    Tensor out(std::move(v));
    out.node->parents = {x.node, gamma.node, beta.node};
    Node* xn = x.node.get();
    Node* gn = gamma.node.get();
    Node* bn = beta.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [xn, gn, bn, on, xhat, inv_std, n, D]() {
        double Dd = static_cast<double>(D);
        for (std::size_t i = 0; i < n; ++i) {
            double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
            for (std::size_t j = 0; j < D; ++j) {
                double dxhat = on->grad(i, j) * gn->value(0, j);
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * xhat(i, j);
                gn->grad(0, j) += on->grad(i, j) * xhat(i, j);
                bn->grad(0, j) += on->grad(i, j);
            }
            for (std::size_t j = 0; j < D; ++j) {
                double dxhat = on->grad(i, j) * gn->value(0, j);
                xn->grad(i, j) +=
                    inv_std[i] * (dxhat - sum_dxhat / Dd - xhat(i, j) * sum_dxhat_xhat / Dd);
            }
        }
    };
    return out;
}

Tensor embedding(const Tensor& table, const std::vector<int>& ids) {
    std::size_t T = ids.size(), D = table.cols();
    Matrix v(T, D, 0.0);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t j = 0; j < D; ++j) v(t, j) = table.value()(static_cast<std::size_t>(ids[t]), j);
    Tensor out(std::move(v));
    out.node->parents = {table.node};
    Node* tn = table.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [tn, on, ids, T, D]() {
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t j = 0; j < D; ++j)
                tn->grad(static_cast<std::size_t>(ids[t]), j) += on->grad(t, j);
    };
    return out;
}

Tensor slice_cols(const Tensor& a, std::size_t c0, std::size_t w) {
    std::size_t R = a.rows();
    Matrix v(R, w, 0.0);
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < w; ++j) v(i, j) = a.value()(i, c0 + j);
    Tensor out(std::move(v));
    out.node->parents = {a.node};
    Node* an = a.node.get();
    Node* on = out.node.get();
    out.node->backward_fn = [an, on, c0, w, R]() {
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < w; ++j) an->grad(i, c0 + j) += on->grad(i, j);
    };
    return out;
}

Tensor concat_cols(const std::vector<Tensor>& parts) {
    std::size_t R = parts.front().rows();
    std::size_t total = 0;
    for (const auto& p : parts) total += p.cols();

    Matrix v(R, total, 0.0);
    std::vector<Node*> pn;
    std::vector<std::size_t> off, wid;
    std::size_t c = 0;
    for (const auto& p : parts) {
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < p.cols(); ++j) v(i, c + j) = p.value()(i, j);
        pn.push_back(p.node.get());
        off.push_back(c);
        wid.push_back(p.cols());
        c += p.cols();
    }
    Tensor out(std::move(v));
    for (const auto& p : parts) out.node->parents.push_back(p.node);
    Node* on = out.node.get();
    out.node->backward_fn = [pn, off, wid, on, R]() {
        for (std::size_t k = 0; k < pn.size(); ++k)
            for (std::size_t i = 0; i < R; ++i)
                for (std::size_t j = 0; j < wid[k]; ++j)
                    pn[k]->grad(i, j) += on->grad(i, off[k] + j);
    };
    return out;
}

}  // namespace mlcpp
