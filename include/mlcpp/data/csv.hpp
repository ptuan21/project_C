#pragma once
// mlcpp — đọc file CSV số vào Matrix.
#include <string>
#include <vector>

#include "mlcpp/core/matrix.hpp"

namespace mlcpp {

struct CsvData {
    std::vector<std::string> headers;  // rỗng nếu has_header == false
    Matrix values;                     // n_samples x n_features
};

// Đọc CSV chứa số, phân tách bằng dấu phẩy.
// Ném std::runtime_error nếu mở file lỗi hoặc số cột không đồng nhất.
CsvData read_csv(const std::string& path, bool has_header = true);

}  // namespace mlcpp
