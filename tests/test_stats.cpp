#include <vector>

#include "mlcpp/matrix.hpp"
#include "mlcpp/stats.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_stats() {
    std::printf("[test_stats]\n");

    std::vector<double> v{1, 2, 3, 4, 5};
    CHECK_NEAR(mean(v), 3.0, 1e-12);
    CHECK_NEAR(variance(v), 2.5, 1e-12);       // phương sai mẫu
    CHECK_NEAR(variance(v, false), 2.0, 1e-12);  // phương sai tổng thể
    CHECK_NEAR(median(v), 3.0, 1e-12);
    CHECK_NEAR(min(v), 1.0, 1e-12);
    CHECK_NEAR(max(v), 5.0, 1e-12);
    CHECK_NEAR(percentile(v, 0), 1.0, 1e-12);
    CHECK_NEAR(percentile(v, 100), 5.0, 1e-12);
    CHECK_NEAR(percentile(v, 50), 3.0, 1e-12);

    // Tương quan hoàn hảo: y = 2x + 1 -> correlation = 1
    std::vector<double> x{1, 2, 3, 4};
    std::vector<double> y{3, 5, 7, 9};
    CHECK_NEAR(correlation(x, y), 1.0, 1e-9);
    CHECK_NEAR(covariance(x, y), 2.0 * variance(x), 1e-9);

    // Tương quan âm hoàn hảo
    std::vector<double> yn{9, 7, 5, 3};
    CHECK_NEAR(correlation(x, yn), -1.0, 1e-9);

    // Ma trận hiệp phương sai: cột 1 và cột 2 giống hệt -> cov đối xứng, dương
    Matrix data = Matrix::from_rows({{1, 2}, {2, 4}, {3, 6}, {4, 8}});
    Matrix cov = covariance_matrix(data);
    CHECK(cov.rows() == 2 && cov.cols() == 2);
    CHECK_NEAR(cov(0, 1), cov(1, 0), 1e-12);          // đối xứng
    CHECK_NEAR(cov(1, 1), 4.0 * cov(0, 0), 1e-9);     // cột 2 = 2 * cột 1
}
