#include <vector>

#include "mlcpp/random.hpp"
#include "mlcpp/stats.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_random() {
    std::printf("[test_random]\n");
    Rng rng(42);

    const int N = 200000;

    // uniform [0,1): trung bình ~0.5, mọi giá trị trong [0,1)
    std::vector<double> u(N);
    bool in_range = true;
    for (int i = 0; i < N; ++i) {
        u[i] = rng.uniform();
        if (u[i] < 0.0 || u[i] >= 1.0) in_range = false;
    }
    CHECK(in_range);
    CHECK_NEAR(mean(u), 0.5, 0.01);
    CHECK_NEAR(variance(u), 1.0 / 12.0, 0.01);  // phương sai lý thuyết của U(0,1)

    // Gaussian(mean=3, sd=2): kiểm tra mean & stddev mẫu
    std::vector<double> g(N);
    for (int i = 0; i < N; ++i) g[i] = rng.gaussian(3.0, 2.0);
    CHECK_NEAR(mean(g), 3.0, 0.05);
    CHECK_NEAR(stddev(g), 2.0, 0.05);

    // Bernoulli(p=0.3): tỉ lệ 1 xấp xỉ 0.3
    int ones = 0;
    for (int i = 0; i < N; ++i) ones += rng.bernoulli(0.3);
    CHECK_NEAR(static_cast<double>(ones) / N, 0.3, 0.01);

    // Exponential(lambda=2): mean lý thuyết = 1/lambda = 0.5
    std::vector<double> e(N);
    for (int i = 0; i < N; ++i) e[i] = rng.exponential(2.0);
    CHECK_NEAR(mean(e), 0.5, 0.02);

    // Poisson(lambda=4): mean xấp xỉ 4
    std::vector<double> p(N);
    for (int i = 0; i < N; ++i) p[i] = rng.poisson(4.0);
    CHECK_NEAR(mean(p), 4.0, 0.05);
}
