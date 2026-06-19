#include "test_framework.hpp"

// Khai báo các nhóm test (định nghĩa ở từng file test_*.cpp).
void test_matrix();
void test_random();
void test_stats();
void test_linalg();
void test_regression();
void test_autograd();

int main() {
    test_matrix();
    test_random();
    test_stats();
    test_linalg();
    test_regression();
    test_autograd();
    return test::summary();
}
