// Demo Tầng 1–2: ma trận + xác suất + thống kê.
// Sinh dữ liệu Gaussian 2 chiều có tương quan, rồi ước lượng lại thống kê.
#include <cstdio>
#include <vector>

#include "mlcpp/core/matrix.hpp"
#include "mlcpp/prob/random.hpp"
#include "mlcpp/prob/stats.hpp"

using namespace mlcpp;

static void print_histogram(const std::vector<double>& v, double lo, double hi, int bins) {
    std::vector<int> count(bins, 0);
    double width = (hi - lo) / bins;
    for (double x : v) {
        int b = static_cast<int>((x - lo) / width);
        if (b >= 0 && b < bins) ++count[b];
    }
    int maxc = 1;
    for (int c : count) maxc = c > maxc ? c : maxc;
    for (int i = 0; i < bins; ++i) {
        double center = lo + (i + 0.5) * width;
        int len = static_cast<int>(50.0 * count[i] / maxc);
        std::printf("  %6.2f | ", center);
        for (int k = 0; k < len; ++k) std::putchar('#');
        std::printf("  (%d)\n", count[i]);
    }
}

int main() {
    std::printf("=== mlcpp demo: Tầng 1–2 ===\n\n");

    // 1) Phép toán ma trận
    Matrix A = Matrix::from_rows({{1, 2, 3}, {4, 5, 6}});
    Matrix B = Matrix::from_rows({{7, 8}, {9, 10}, {11, 12}});
    std::printf("A x B =\n%s\n", A.matmul(B).to_string().c_str());

    // 2) Sinh mẫu Gaussian và vẽ histogram
    Rng rng(2026);
    const int N = 5000;
    std::vector<double> samples(N);
    for (int i = 0; i < N; ++i) samples[i] = rng.gaussian(0.0, 1.0);

    std::printf("Histogram của %d mẫu Gaussian(0, 1):\n", N);
    print_histogram(samples, -4.0, 4.0, 16);
    std::printf("  mean=%.4f  stddev=%.4f  median=%.4f\n\n", mean(samples), stddev(samples),
                median(samples));

    // 3) Dữ liệu 2 biến có tương quan: y = 0.8*x + nhiễu
    Matrix data(N, 2, 0.0);
    std::vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        double x = rng.gaussian(0.0, 1.0);
        double y = 0.8 * x + rng.gaussian(0.0, 0.5);
        data(i, 0) = x;
        data(i, 1) = y;
        xs[i] = x;
        ys[i] = y;
    }
    std::printf("Tương quan(x, y) = %.4f (kỳ vọng > 0)\n", correlation(xs, ys));
    std::printf("Ma trận hiệp phương sai:\n%s\n", covariance_matrix(data).to_string().c_str());

    std::printf("Hoàn tất. Thử sửa code và chạy lại `make demo`.\n");
    return 0;
}
