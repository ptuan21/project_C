#include "mlcpp/core/serialize.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace mlcpp {

static void write_u32(std::ofstream& f, std::uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}
static std::uint32_t read_u32(std::ifstream& f) {
    std::uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 4);
    return v;
}

void save_matrices(const std::string& path, const std::vector<Matrix>& mats) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_matrices: không mở được file: " + path);
    f.write("MLCP", 4);
    write_u32(f, 1);                                      // version
    write_u32(f, static_cast<std::uint32_t>(mats.size()));
    for (const auto& m : mats) {
        write_u32(f, static_cast<std::uint32_t>(m.rows()));
        write_u32(f, static_cast<std::uint32_t>(m.cols()));
        for (std::size_t i = 0; i < m.rows(); ++i)
            for (std::size_t j = 0; j < m.cols(); ++j) {
                float v = static_cast<float>(m(i, j));   // lưu float32 cho gọn
                f.write(reinterpret_cast<const char*>(&v), 4);
            }
    }
}

std::vector<Matrix> load_matrices(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_matrices: không mở được file: " + path);
    char magic[4];
    f.read(magic, 4);
    if (magic[0] != 'M' || magic[1] != 'L' || magic[2] != 'C' || magic[3] != 'P')
        throw std::runtime_error("load_matrices: sai magic (file không hợp lệ)");
    read_u32(f);  // version
    std::uint32_t count = read_u32(f);

    std::vector<Matrix> mats;
    mats.reserve(count);
    for (std::uint32_t k = 0; k < count; ++k) {
        std::uint32_t r = read_u32(f), c = read_u32(f);
        Matrix m(r, c, 0.0);
        for (std::uint32_t i = 0; i < r; ++i)
            for (std::uint32_t j = 0; j < c; ++j) {
                float v = 0.0f;
                f.read(reinterpret_cast<char*>(&v), 4);
                m(i, j) = static_cast<double>(v);
            }
        mats.push_back(std::move(m));
    }
    return mats;
}

}  // namespace mlcpp
