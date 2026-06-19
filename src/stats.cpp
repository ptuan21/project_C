#include "mlcpp/stats.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mlcpp {

double mean(const std::vector<double>& v) {
    if (v.empty()) throw std::invalid_argument("mean: vector rỗng");
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double variance(const std::vector<double>& v, bool sample) {
    if (v.size() < 2) throw std::invalid_argument("variance: cần ít nhất 2 phần tử");
    double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return s / static_cast<double>(v.size() - (sample ? 1 : 0));
}

double stddev(const std::vector<double>& v, bool sample) { return std::sqrt(variance(v, sample)); }

double covariance(const std::vector<double>& a, const std::vector<double>& b, bool sample) {
    if (a.size() != b.size()) throw std::invalid_argument("covariance: hai vector khác độ dài");
    if (a.size() < 2) throw std::invalid_argument("covariance: cần ít nhất 2 phần tử");
    double ma = mean(a), mb = mean(b), s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += (a[i] - ma) * (b[i] - mb);
    return s / static_cast<double>(a.size() - (sample ? 1 : 0));
}

double correlation(const std::vector<double>& a, const std::vector<double>& b) {
    return covariance(a, b) / (stddev(a) * stddev(b));
}

double min(const std::vector<double>& v) {
    if (v.empty()) throw std::invalid_argument("min: vector rỗng");
    return *std::min_element(v.begin(), v.end());
}

double max(const std::vector<double>& v) {
    if (v.empty()) throw std::invalid_argument("max: vector rỗng");
    return *std::max_element(v.begin(), v.end());
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) throw std::invalid_argument("percentile: vector rỗng");
    if (p < 0.0 || p > 100.0) throw std::invalid_argument("percentile: p ngoài [0,100]");
    std::sort(v.begin(), v.end());
    // Nội suy tuyến tính giữa các hạng (kiểu numpy mặc định).
    double rank = (p / 100.0) * (static_cast<double>(v.size()) - 1.0);
    std::size_t lo = static_cast<std::size_t>(std::floor(rank));
    std::size_t hi = static_cast<std::size_t>(std::ceil(rank));
    double frac = rank - static_cast<double>(lo);
    return v[lo] + frac * (v[hi] - v[lo]);
}

double median(std::vector<double> v) { return percentile(std::move(v), 50.0); }

Matrix covariance_matrix(const Matrix& data, bool sample) {
    std::size_t n = data.rows();
    std::size_t d = data.cols();
    if (n < 2) throw std::invalid_argument("covariance_matrix: cần ít nhất 2 mẫu");

    std::vector<double> col_mean(d, 0.0);
    for (std::size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += data(i, j);
        col_mean[j] = s / static_cast<double>(n);
    }

    Matrix cov(d, d, 0.0);
    double denom = static_cast<double>(n - (sample ? 1 : 0));
    for (std::size_t j = 0; j < d; ++j)
        for (std::size_t k = j; k < d; ++k) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                s += (data(i, j) - col_mean[j]) * (data(i, k) - col_mean[k]);
            double c = s / denom;
            cov(j, k) = c;
            cov(k, j) = c;  // đối xứng
        }
    return cov;
}

}  // namespace mlcpp
