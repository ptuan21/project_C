#pragma once
// mlcpp::Matrix — ma trận dày đặc (dense), lưu theo hàng (row-major).
// Đây là nền tảng cho mọi thuật toán ML/DL về sau.
#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace mlcpp {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);

    // Các hàm khởi tạo tiện dụng
    static Matrix identity(std::size_t n);
    static Matrix from_rows(std::initializer_list<std::initializer_list<double>> rows);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }

    // Truy cập phần tử (r, c). Không kiểm tra biên ở bản phát hành để nhanh.
    double& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    double operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    // Con trỏ dữ liệu thô — cần cho BLAS (Accelerate).
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    Matrix transpose() const;
    Matrix matmul(const Matrix& other) const;  // tích ma trận

    // Phép toán theo từng phần tử (element-wise)
    Matrix operator+(const Matrix& o) const;
    Matrix operator-(const Matrix& o) const;
    Matrix operator*(double s) const;        // nhân vô hướng
    Matrix hadamard(const Matrix& o) const;  // tích từng phần tử

    void fill(double v);
    std::string to_string() const;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

Matrix operator*(double s, const Matrix& m);

}  // namespace mlcpp
