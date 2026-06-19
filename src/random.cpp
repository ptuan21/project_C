#include "mlcpp/random.hpp"

#include <cmath>

namespace mlcpp {

Rng::Rng(std::uint64_t seed) : state_(seed ? seed : 0x9E3779B97F4A7C15ULL) {}

// xorshift64* — nhanh, chất lượng tốt cho mục đích học tập / mô phỏng.
std::uint64_t Rng::next_u64() {
    std::uint64_t x = state_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state_ = x;
    return x * 0x2545F4914F6CDD1DULL;
}

double Rng::uniform() {
    // Lấy 53 bit cao để tạo double trong [0, 1).
    return (next_u64() >> 11) * (1.0 / 9007199254740992.0);  // 2^53
}

double Rng::uniform(double a, double b) { return a + (b - a) * uniform(); }

// Box-Muller: từ 2 số uniform tạo ra 2 số Gaussian chuẩn.
double Rng::gaussian(double mean, double stddev) {
    if (has_spare_) {
        has_spare_ = false;
        return mean + stddev * spare_;
    }
    double u1, u2;
    do {
        u1 = uniform();
    } while (u1 <= 1e-12);  // tránh log(0)
    u2 = uniform();
    double mag = std::sqrt(-2.0 * std::log(u1));
    spare_ = mag * std::sin(2.0 * M_PI * u2);
    has_spare_ = true;
    return mean + stddev * (mag * std::cos(2.0 * M_PI * u2));
}

double Rng::exponential(double lambda) {
    double u;
    do {
        u = uniform();
    } while (u <= 1e-12);
    return -std::log(u) / lambda;
}

int Rng::bernoulli(double p) { return uniform() < p ? 1 : 0; }

// Thuật toán Knuth cho phân phối Poisson.
int Rng::poisson(double lambda) {
    double L = std::exp(-lambda);
    int k = 0;
    double p = 1.0;
    do {
        ++k;
        p *= uniform();
    } while (p > L);
    return k - 1;
}

}  // namespace mlcpp
