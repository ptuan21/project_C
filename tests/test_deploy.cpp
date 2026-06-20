#include <vector>

#include "mlcpp/core/matrix.hpp"
#include "mlcpp/core/quantize.hpp"
#include "mlcpp/core/serialize.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_deploy() {
    std::printf("[test_deploy]\n");

    // --- serialize: lưu rồi nạp phải khớp (sai số float32) ---
    Matrix A = Matrix::from_rows({{1.5, -2.25, 3.0}, {0.125, 100.0, -0.5}});
    Matrix B = Matrix::from_rows({{7.0}, {8.0}});
    save_matrices("build/_test_ser.bin", {A, B});
    std::vector<Matrix> loaded = load_matrices("build/_test_ser.bin");
    CHECK(loaded.size() == 2);
    CHECK(loaded[0].rows() == 2 && loaded[0].cols() == 3);
    CHECK_NEAR(loaded[0](1, 1), 100.0, 1e-4);
    CHECK_NEAR(loaded[0](0, 1), -2.25, 1e-6);
    CHECK_NEAR(loaded[1](1, 0), 8.0, 1e-6);

    // --- quantize: sai số mỗi phần tử <= 0.5 * scale (đảm bảo bởi làm tròn) ---
    Matrix W = Matrix::from_rows({{1.0, -5.0}, {2.0, 10.0}, {-3.0, 0.0}});
    QuantizedMatrix qm = quantize_per_col(W);
    Matrix Wd = dequantize(qm);
    for (std::size_t i = 0; i < W.rows(); ++i)
        for (std::size_t j = 0; j < W.cols(); ++j)
            CHECK(std::fabs(W(i, j) - Wd(i, j)) <= 0.5 * qm.scale[j] + 1e-6);

    // int8 nhẹ hơn float nhiều
    std::size_t float_bytes = W.rows() * W.cols() * 4;
    CHECK(qm.bytes() < float_bytes);

    // --- qmatmul ≈ matmul, sai số bị chặn bởi (Σ|X|) * 0.5 * scale ---
    Matrix X = Matrix::from_rows({{0.5, 1.0, -2.0}, {1.0, -1.0, 0.5}});
    Matrix ref = X.matmul(W);          // float thật
    Matrix qout = qmatmul(X, qm);      // dùng trọng số int8
    for (std::size_t i = 0; i < X.rows(); ++i) {
        double sum_absx = 0.0;
        for (std::size_t k = 0; k < X.cols(); ++k) sum_absx += std::fabs(X(i, k));
        for (std::size_t j = 0; j < W.cols(); ++j) {
            double bound = sum_absx * 0.5 * qm.scale[j] + 1e-6;
            CHECK(std::fabs(ref(i, j) - qout(i, j)) <= bound);
        }
    }

    // --- qmatmul_i8 (int8 đầy đủ): phải bằng (X lượng tử) @ (W lượng tử) ---
    // Tự lượng tử hóa X per-row giống hàm thư viện rồi giải lượng tử để có kết quả kỳ vọng.
    Matrix Xa(X.rows(), X.cols(), 0.0);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        double amax = 0.0;
        for (std::size_t k = 0; k < X.cols(); ++k) amax = std::max(amax, std::fabs(X(i, k)));
        double sx = amax / 127.0;
        if (sx == 0.0) sx = 1.0;
        for (std::size_t k = 0; k < X.cols(); ++k) {
            double r = std::round(X(i, k) / sx);
            if (r > 127) r = 127;
            if (r < -127) r = -127;
            Xa(i, k) = r * sx;
        }
    }
    Matrix expected = Xa.matmul(dequantize(qm));
    Matrix i8 = qmatmul_i8(X, qm);
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < W.cols(); ++j) CHECK_NEAR(i8(i, j), expected(i, j), 1e-6);
}
