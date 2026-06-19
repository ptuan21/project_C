#pragma once
// mlcpp — đọc dữ liệu MNIST ở định dạng IDX (file giải nén .ubyte).
#include <string>
#include <vector>

#include "mlcpp/matrix.hpp"

namespace mlcpp {

struct Mnist {
    Matrix images;            // n x 784, giá trị pixel đã chuẩn hóa về [0, 1]
    std::vector<int> labels;  // n nhãn trong [0, 9]
};

// Đọc cặp file ảnh + nhãn. max_items > 0 để giới hạn số mẫu (load nhanh khi thử).
// Ném std::runtime_error nếu mở file lỗi hoặc magic number sai.
Mnist load_mnist(const std::string& images_path, const std::string& labels_path,
                 std::size_t max_items = 0);

}  // namespace mlcpp
