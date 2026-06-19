#pragma once
// Khung test tối giản, không phụ thuộc thư viện ngoài.
#include <cmath>
#include <cstdio>

namespace test {

inline int& passed() {
    static int p = 0;
    return p;
}
inline int& failed() {
    static int f = 0;
    return f;
}

inline int summary() {
    std::printf("\n=== %d passed, %d failed ===\n", passed(), failed());
    return failed() == 0 ? 0 : 1;
}

}  // namespace test

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (cond) {                                                              \
            ++test::passed();                                                    \
        } else {                                                                 \
            ++test::failed();                                                    \
            std::printf("  [FAIL] %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                      \
    do {                                                                           \
        double _a = (a), _b = (b);                                                 \
        if (std::fabs(_a - _b) <= (eps)) {                                         \
            ++test::passed();                                                      \
        } else {                                                                   \
            ++test::failed();                                                      \
            std::printf("  [FAIL] %s:%d  CHECK_NEAR(%s, %s) -> %g vs %g\n",        \
                        __FILE__, __LINE__, #a, #b, _a, _b);                       \
        }                                                                          \
    } while (0)
