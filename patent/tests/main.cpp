#include "test_framework.hpp"

void test_bm25();
void test_risk();

int main() {
    test_bm25();
    test_risk();
    return test::summary();
}
