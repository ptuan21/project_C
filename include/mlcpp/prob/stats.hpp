#pragma once
// mlcpp — thống kê mô tả: mean, variance, covariance, correlation, percentile.
#include <vector>

#include "mlcpp/core/matrix.hpp"

namespace mlcpp {

double mean(const std::vector<double>& v);
double variance(const std::vector<double>& v, bool sample = true);
double stddev(const std::vector<double>& v, bool sample = true);
double covariance(const std::vector<double>& a, const std::vector<double>& b, bool sample = true);
double correlation(const std::vector<double>& a, const std::vector<double>& b);

double min(const std::vector<double>& v);
double max(const std::vector<double>& v);
double median(std::vector<double> v);          // sao chép rồi sắp xếp
double percentile(std::vector<double> v, double p);  // p trong [0, 100]

// Ma trận hiệp phương sai. data: n_samples hàng x n_features cột.
// Trả về ma trận n_features x n_features.
Matrix covariance_matrix(const Matrix& data, bool sample = true);

}  // namespace mlcpp
