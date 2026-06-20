#include "mlcpp/data/mnist.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace mlcpp {

static std::uint32_t read_u32_be(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16) |
           (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]);
}

Mnist load_mnist(const std::string& images_path, const std::string& labels_path,
                 std::size_t max_items) {
    std::ifstream fi(images_path, std::ios::binary);
    if (!fi) throw std::runtime_error("load_mnist: không mở được file ảnh: " + images_path);
    std::ifstream fl(labels_path, std::ios::binary);
    if (!fl) throw std::runtime_error("load_mnist: không mở được file nhãn: " + labels_path);

    if (read_u32_be(fi) != 0x00000803) throw std::runtime_error("load_mnist: magic ảnh sai");
    std::uint32_t n_img = read_u32_be(fi);
    std::uint32_t rows = read_u32_be(fi);
    std::uint32_t cols = read_u32_be(fi);

    if (read_u32_be(fl) != 0x00000801) throw std::runtime_error("load_mnist: magic nhãn sai");
    std::uint32_t n_lab = read_u32_be(fl);
    if (n_img != n_lab) throw std::runtime_error("load_mnist: số ảnh != số nhãn");

    std::size_t n = n_img;
    if (max_items > 0 && max_items < n) n = max_items;
    std::size_t pixels = static_cast<std::size_t>(rows) * cols;

    Mnist out;
    out.images = Matrix(n, pixels, 0.0);
    out.labels.resize(n);

    std::vector<unsigned char> buf(pixels);
    for (std::size_t i = 0; i < n; ++i) {
        fi.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(pixels));
        for (std::size_t j = 0; j < pixels; ++j)
            out.images(i, j) = static_cast<double>(buf[j]) / 255.0;  // chuẩn hóa
        unsigned char lab = 0;
        fl.read(reinterpret_cast<char*>(&lab), 1);
        out.labels[i] = static_cast<int>(lab);
    }
    return out;
}

}  // namespace mlcpp
