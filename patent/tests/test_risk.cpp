#include <algorithm>
#include <string>
#include <vector>

#include "patent/risk.hpp"
#include "test_framework.hpp"

using namespace patent;

void test_risk() {
    std::printf("[test_risk]\n");

    std::string claim =
        "1 . A rack comprising: a base; a plurality of arms pivotally connected to the base; "
        "and a lock to secure the arms to the base.";

    // tách thành 3 đặc điểm
    auto elems = split_claim_elements(claim);
    CHECK(elems.size() == 3);

    // sản phẩm khớp gần hết -> rủi ro cao
    RiskReport hi = assess_risk(claim, "a folding rack with a base and pivoting arms and a lock");
    CHECK(hi.elements.size() == 3);
    CHECK(hi.n_match >= 2);
    CHECK(hi.level == RiskLevel::High || hi.level == RiskLevel::Medium);
    CHECK(hi.elements[0].verdict == "khop");  // "a base" -> khớp

    // sản phẩm chẳng liên quan -> rủi ro thấp, có đặc điểm thiếu
    RiskReport lo = assess_risk(claim, "wireless bluetooth headphones with noise cancelling");
    CHECK(lo.level == RiskLevel::Low);
    CHECK(lo.any_missing);
    CHECK(lo.score < hi.score);

    CHECK(std::string(risk_label(RiskLevel::High)) == "CAO");
}
