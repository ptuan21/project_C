#include "mlcpp/core/quantize.hpp"

#include <cmath>
#include <stdexcept>

namespace mlcpp {

QuantizedMatrix quantize_per_col(const Matrix& W) {
    QuantizedMatrix qm;
    qm.rows = W.rows();
    qm.cols = W.cols();
    qm.q.resize(W.rows() * W.cols());
    qm.scale.resize(W.cols());

    for (std::size_t j = 0; j < W.cols(); ++j) {
        double amax = 0.0;
        for (std::size_t i = 0; i < W.rows(); ++i) amax = std::max(amax, std::fabs(W(i, j)));
        double s = amax / 127.0;
        if (s == 0.0) s = 1.0;  // cột toàn 0
        qm.scale[j] = static_cast<float>(s);
        for (std::size_t i = 0; i < W.rows(); ++i) {
            double r = std::round(W(i, j) / s);
            if (r > 127) r = 127;
            if (r < -127) r = -127;
            qm.q[i * W.cols() + j] = static_cast<std::int8_t>(r);
        }
    }
    return qm;
}

Matrix dequantize(const QuantizedMatrix& qm) {
    Matrix W(qm.rows, qm.cols, 0.0);
    for (std::size_t i = 0; i < qm.rows; ++i)
        for (std::size_t j = 0; j < qm.cols; ++j)
            W(i, j) = static_cast<double>(qm.q[i * qm.cols + j]) * qm.scale[j];
    return W;
}

Matrix qmatmul(const Matrix& X, const QuantizedMatrix& Wq) {
    if (X.cols() != Wq.rows)
        throw std::invalid_argument("qmatmul: X.cols phải bằng Wq.rows");
    std::size_t n = X.rows(), K = Wq.rows, M = Wq.cols;
    Matrix out(n, M, 0.0);
    // Thứ tự i-k-j thân thiện cache: truy cập q theo hàng (liên tục).
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            double xik = X(i, k);
            const std::int8_t* qrow = &Wq.q[k * M];
            for (std::size_t j = 0; j < M; ++j) out(i, j) += xik * static_cast<double>(qrow[j]);
        }
        for (std::size_t j = 0; j < M; ++j) out(i, j) *= static_cast<double>(Wq.scale[j]);
    }
    return out;
}

Matrix qmatmul_i8(const Matrix& X, const QuantizedMatrix& Wq) {
    if (X.cols() != Wq.rows)
        throw std::invalid_argument("qmatmul_i8: X.cols phải bằng Wq.rows");
    std::size_t n = X.rows(), K = Wq.rows, M = Wq.cols;
    Matrix out(n, M, 0.0);

    std::vector<std::int8_t> qx(K);
    std::vector<std::int32_t> acc(M);
    for (std::size_t i = 0; i < n; ++i) {
        // Lượng tử hóa hàng activation X[i,:] đối xứng theo hàng.
        double amax = 0.0;
        for (std::size_t k = 0; k < K; ++k) amax = std::max(amax, std::fabs(X(i, k)));
        double sx = amax / 127.0;
        if (sx == 0.0) sx = 1.0;
        for (std::size_t k = 0; k < K; ++k) {
            double r = std::round(X(i, k) / sx);
            if (r > 127) r = 127;
            if (r < -127) r = -127;
            qx[k] = static_cast<std::int8_t>(r);
        }

        // Tích lũy số nguyên int32: acc[j] = Σ_k qx[k] * qw[k,j].
        for (std::size_t j = 0; j < M; ++j) acc[j] = 0;
        for (std::size_t k = 0; k < K; ++k) {
            std::int32_t xk = qx[k];
            if (xk == 0) continue;
            const std::int8_t* qrow = &Wq.q[k * M];
            for (std::size_t j = 0; j < M; ++j) acc[j] += xk * static_cast<std::int32_t>(qrow[j]);
        }

        // Giải lượng tử: out = acc * scale_x * scale_w.
        for (std::size_t j = 0; j < M; ++j)
            out(i, j) = static_cast<double>(acc[j]) * sx * static_cast<double>(Wq.scale[j]);
    }
    return out;
}

}  // namespace mlcpp
