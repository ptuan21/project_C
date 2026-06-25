#pragma once
// mlcpp — lưu/nạp trọng số ra file nhị phân (float32) để deploy.
// Train một lần trên máy mạnh -> lưu file -> nạp trên bo mạch yếu.
#include <string>
#include <vector>

#include "mlcpp/core/matrix.hpp"

namespace mlcpp {

// Định dạng: magic "MLCP", version, số ma trận, rồi (rows, cols, dữ liệu float32) mỗi ma trận.
void save_matrices(const std::string& path, const std::vector<Matrix>& mats);
std::vector<Matrix> load_matrices(const std::string& path);

}  // namespace mlcpp
