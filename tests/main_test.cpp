#include "test_framework.hpp"

// Khai báo các nhóm test (định nghĩa ở từng file test_*.cpp).
void test_matrix();
void test_random();
void test_stats();
void test_autograd();
void test_transformer();
void test_optim();
void test_tokenizer();
void test_deploy();

int main() {
    test_matrix();
    test_random();
    test_stats();
    test_autograd();
    test_transformer();
    test_optim();
    test_tokenizer();
    test_deploy();
    return test::summary();
}
