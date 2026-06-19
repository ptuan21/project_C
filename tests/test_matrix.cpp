#include "mlcpp/matrix.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_matrix() {
    std::printf("[test_matrix]\n");

    // Khởi tạo và truy cập
    Matrix a(2, 3, 1.5);
    CHECK(a.rows() == 2 && a.cols() == 3);
    CHECK_NEAR(a(1, 2), 1.5, 1e-12);

    // Ma trận đơn vị
    Matrix I = Matrix::identity(3);
    CHECK_NEAR(I(0, 0), 1.0, 1e-12);
    CHECK_NEAR(I(0, 1), 0.0, 1e-12);

    // Tích ma trận: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    Matrix x = Matrix::from_rows({{1, 2}, {3, 4}});
    Matrix y = Matrix::from_rows({{5, 6}, {7, 8}});
    Matrix z = x.matmul(y);
    CHECK_NEAR(z(0, 0), 19.0, 1e-9);
    CHECK_NEAR(z(0, 1), 22.0, 1e-9);
    CHECK_NEAR(z(1, 0), 43.0, 1e-9);
    CHECK_NEAR(z(1, 1), 50.0, 1e-9);

    // Nhân với ma trận đơn vị giữ nguyên giá trị
    Matrix xi = x.matmul(Matrix::identity(2));
    CHECK_NEAR(xi(1, 1), 4.0, 1e-9);

    // Chuyển vị
    Matrix t = x.transpose();
    CHECK_NEAR(t(0, 1), 3.0, 1e-12);
    CHECK_NEAR(t(1, 0), 2.0, 1e-12);

    // Cộng, trừ, nhân vô hướng, Hadamard
    Matrix s = x + y;
    CHECK_NEAR(s(0, 0), 6.0, 1e-12);
    Matrix d = y - x;
    CHECK_NEAR(d(1, 1), 4.0, 1e-12);
    Matrix sc = 2.0 * x;
    CHECK_NEAR(sc(0, 1), 4.0, 1e-12);
    Matrix h = x.hadamard(y);
    CHECK_NEAR(h(1, 1), 32.0, 1e-12);
}
