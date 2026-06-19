#pragma once
// mlcpp::Rng — bộ sinh số ngẫu nhiên và lấy mẫu từ các phân phối xác suất.
// Tự cài để hiểu cách từ uniform suy ra Gaussian, Poisson, ...
#include <cstdint>

namespace mlcpp {

class Rng {
public:
    explicit Rng(std::uint64_t seed = 0x9E3779B97F4A7C15ULL);

    double uniform();                       // [0, 1)
    double uniform(double a, double b);      // [a, b)
    double gaussian(double mean = 0.0, double stddev = 1.0);  // Box-Muller
    double exponential(double lambda = 1.0);
    int bernoulli(double p);                 // trả về 0 hoặc 1
    int poisson(double lambda);              // thuật toán Knuth

private:
    std::uint64_t next_u64();                // xorshift64*

    std::uint64_t state_;
    bool has_spare_ = false;                 // cache cho Box-Muller
    double spare_ = 0.0;
};

}  // namespace mlcpp
