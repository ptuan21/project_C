#include "mlcpp/linalg.hpp"

#include <cmath>
#include <stdexcept>

namespace mlcpp {

Matrix solve(const Matrix& A_in, const Matrix& B_in) {
    std::size_t n = A_in.rows();
    if (A_in.cols() != n) throw std::invalid_argument("solve: A phải vuông");
    if (B_in.rows() != n) throw std::invalid_argument("solve: B.rows phải bằng A.rows");
    std::size_t m = B_in.cols();

    Matrix A = A_in;  // bản sao để biến đổi tại chỗ
    Matrix B = B_in;

    // Khử tiến với pivoting từng phần.
    for (std::size_t k = 0; k < n; ++k) {
        // Tìm hàng có |pivot| lớn nhất để ổn định số học.
        std::size_t piv = k;
        double best = std::fabs(A(k, k));
        for (std::size_t i = k + 1; i < n; ++i) {
            double v = std::fabs(A(i, k));
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best < 1e-12) throw std::runtime_error("solve: ma trận suy biến (singular)");

        if (piv != k) {  // hoán đổi hàng k và piv ở cả A lẫn B
            for (std::size_t j = 0; j < n; ++j) std::swap(A(k, j), A(piv, j));
            for (std::size_t j = 0; j < m; ++j) std::swap(B(k, j), B(piv, j));
        }

        for (std::size_t i = k + 1; i < n; ++i) {
            double f = A(i, k) / A(k, k);
            for (std::size_t j = k; j < n; ++j) A(i, j) -= f * A(k, j);
            for (std::size_t j = 0; j < m; ++j) B(i, j) -= f * B(k, j);
        }
    }

    // Thế ngược cho từng cột của B.
    Matrix X(n, m, 0.0);
    for (std::size_t c = 0; c < m; ++c) {
        for (std::size_t ii = 0; ii < n; ++ii) {
            std::size_t i = n - 1 - ii;
            double s = B(i, c);
            for (std::size_t j = i + 1; j < n; ++j) s -= A(i, j) * X(j, c);
            X(i, c) = s / A(i, i);
        }
    }
    return X;
}

Matrix inverse(const Matrix& A) {
    if (A.rows() != A.cols()) throw std::invalid_argument("inverse: A phải vuông");
    return solve(A, Matrix::identity(A.rows()));
}

}  // namespace mlcpp
