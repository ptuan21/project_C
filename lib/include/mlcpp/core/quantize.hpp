#pragma once
// mlcpp — lượng tử hóa int8 (weight-only) để model nhẹ hơn ~4x cho deploy.
// Mỗi cột (1 đặc trưng đầu ra) có một hệ số scale riêng (per-column symmetric).
#include <cstdint>
#include <vector>

#include "mlcpp/core/matrix.hpp"

namespace mlcpp {

struct QuantizedMatrix {
    std::size_t rows = 0, cols = 0;
    std::vector<std::int8_t> q;  // rows*cols, lưu theo hàng
    std::vector<float> scale;    // độ dài = cols (mỗi cột một scale)

    std::size_t bytes() const { return q.size() + scale.size() * sizeof(float); }
};

// Lượng tử hóa đối xứng theo từng cột: q = round(W / scale), scale = max|cột| / 127.
QuantizedMatrix quantize_per_col(const Matrix& W);

// Giải lượng tử về Matrix float: W ≈ q * scale.
Matrix dequantize(const QuantizedMatrix& qm);

// Nhân ma trận dùng trọng số int8: X (n x rows) * Wq (rows x cols) -> (n x cols).
// Hoạt động X giữ ở float (weight-only quantization).
Matrix qmatmul(const Matrix& X, const QuantizedMatrix& Wq);

// int8 GEMM đầy đủ: lượng tử hóa activation X per-row động (int8), tích lũy số nguyên int32,
// rồi giải lượng tử bằng (scale_x[i] * scale_w[j]). Số học toàn int -> nhanh hơn trên bo mạch yếu.
Matrix qmatmul_i8(const Matrix& X, const QuantizedMatrix& Wq);

}  // namespace mlcpp
