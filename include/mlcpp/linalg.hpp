#pragma once
// mlcpp — đại số tuyến tính: giải hệ phương trình & nghịch đảo bằng phân rã LU.
#include "mlcpp/matrix.hpp"

namespace mlcpp {

// Giải AX = B bằng khử Gauss với pivoting từng phần.
// A: n x n (vuông). B: n x m. Trả về X: n x m.
// Ném std::runtime_error nếu A gần suy biến (singular).
Matrix solve(const Matrix& A, const Matrix& B);

// Nghịch đảo ma trận vuông (giải A * X = I).
Matrix inverse(const Matrix& A);

}  // namespace mlcpp
