#include "mlcpp/matrix.hpp"

#include <sstream>
#include <stdexcept>

#ifdef MLCPP_USE_ACCELERATE
#include <Accelerate/Accelerate.h>  // cblas_dgemm tối ưu cho Apple Silicon
#endif

namespace mlcpp {

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix Matrix::identity(std::size_t n) {
    Matrix m(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
    return m;
}

Matrix Matrix::from_rows(std::initializer_list<std::initializer_list<double>> rows) {
    std::size_t r = rows.size();
    std::size_t c = r ? rows.begin()->size() : 0;
    Matrix m(r, c, 0.0);
    std::size_t i = 0;
    for (const auto& row : rows) {
        if (row.size() != c) throw std::invalid_argument("from_rows: số cột không đồng nhất");
        std::size_t j = 0;
        for (double v : row) m(i, j++) = v;
        ++i;
    }
    return m;
}

Matrix Matrix::transpose() const {
    Matrix t(cols_, rows_, 0.0);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = 0; j < cols_; ++j) t(j, i) = (*this)(i, j);
    return t;
}

Matrix Matrix::matmul(const Matrix& o) const {
    if (cols_ != o.rows_)
        throw std::invalid_argument("matmul: kích thước không khớp (A.cols != B.rows)");
    Matrix r(rows_, o.cols_, 0.0);
#ifdef MLCPP_USE_ACCELERATE
    // C = 1.0 * A * B + 0.0 * C  (row-major)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(rows_), static_cast<int>(o.cols_), static_cast<int>(cols_),
                1.0, data_.data(), static_cast<int>(cols_),
                o.data_.data(), static_cast<int>(o.cols_),
                0.0, r.data_.data(), static_cast<int>(o.cols_));
#else
    // Bản ngây thơ, thứ tự i-k-j để thân thiện với cache.
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t k = 0; k < cols_; ++k) {
            double a = (*this)(i, k);
            for (std::size_t j = 0; j < o.cols_; ++j) r(i, j) += a * o(k, j);
        }
#endif
    return r;
}

static void check_same_shape(const Matrix& a, const Matrix& b, const char* op) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::invalid_argument(std::string(op) + ": kích thước hai ma trận khác nhau");
}

Matrix Matrix::operator+(const Matrix& o) const {
    check_same_shape(*this, o, "operator+");
    Matrix r(rows_, cols_, 0.0);
    for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] + o.data_[i];
    return r;
}

Matrix Matrix::operator-(const Matrix& o) const {
    check_same_shape(*this, o, "operator-");
    Matrix r(rows_, cols_, 0.0);
    for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] - o.data_[i];
    return r;
}

Matrix Matrix::operator*(double s) const {
    Matrix r(rows_, cols_, 0.0);
    for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * s;
    return r;
}

Matrix Matrix::hadamard(const Matrix& o) const {
    check_same_shape(*this, o, "hadamard");
    Matrix r(rows_, cols_, 0.0);
    for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * o.data_[i];
    return r;
}

void Matrix::fill(double v) {
    for (auto& x : data_) x = v;
}

std::string Matrix::to_string() const {
    std::ostringstream os;
    os << "Matrix(" << rows_ << "x" << cols_ << ")\n";
    for (std::size_t i = 0; i < rows_; ++i) {
        os << "  [ ";
        for (std::size_t j = 0; j < cols_; ++j) os << (*this)(i, j) << ' ';
        os << "]\n";
    }
    return os.str();
}

Matrix operator*(double s, const Matrix& m) { return m * s; }

}  // namespace mlcpp
