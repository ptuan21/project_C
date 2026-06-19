#include "mlcpp/linalg.hpp"
#include "mlcpp/matrix.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_linalg() {
    std::printf("[test_linalg]\n");

    // Giải hệ: [[2,1],[1,3]] x = [[5],[10]]  => x = [1, 3]
    Matrix A = Matrix::from_rows({{2, 1}, {1, 3}});
    Matrix b = Matrix::from_rows({{5}, {10}});
    Matrix x = solve(A, b);
    CHECK_NEAR(x(0, 0), 1.0, 1e-9);
    CHECK_NEAR(x(1, 0), 3.0, 1e-9);

    // Nghịch đảo: A * A⁻¹ = I
    Matrix Ainv = inverse(A);
    Matrix I = A.matmul(Ainv);
    CHECK_NEAR(I(0, 0), 1.0, 1e-9);
    CHECK_NEAR(I(0, 1), 0.0, 1e-9);
    CHECK_NEAR(I(1, 0), 0.0, 1e-9);
    CHECK_NEAR(I(1, 1), 1.0, 1e-9);

    // Pivoting: pivot đầu bằng 0 vẫn giải được nhờ hoán đổi hàng.
    Matrix P = Matrix::from_rows({{0, 1}, {1, 0}});
    Matrix pb = Matrix::from_rows({{2}, {3}});
    Matrix px = solve(P, pb);
    CHECK_NEAR(px(0, 0), 3.0, 1e-9);
    CHECK_NEAR(px(1, 0), 2.0, 1e-9);

    // Ma trận suy biến phải ném ngoại lệ.
    bool threw = false;
    try {
        Matrix S = Matrix::from_rows({{1, 2}, {2, 4}});
        solve(S, Matrix::from_rows({{1}, {2}}));
    } catch (...) {
        threw = true;
    }
    CHECK(threw);
}
