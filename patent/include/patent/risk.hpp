#pragma once
// patent — P2: chấm rủi ro vi phạm bằng cách so từng đặc điểm của claim với sản phẩm.
// Theo "all-elements rule": vi phạm literal đòi hỏi MỌI đặc điểm của claim đều có mặt.
#include <string>
#include <vector>

namespace patent {

enum class RiskLevel { High, Medium, Low };

struct Element {
    std::string text;                   // câu chữ đặc điểm (limitation)
    std::vector<std::string> keyterms;  // từ khóa của đặc điểm
    std::vector<std::string> matched;   // từ khóa khớp với mô tả sản phẩm
    double coverage = 0.0;              // matched / keyterms
    std::string verdict;               // "khop" | "mot_phan" | "khong"
};

struct RiskReport {
    std::vector<Element> elements;
    int n_match = 0, n_partial = 0, n_none = 0;
    double score = 0.0;       // (khớp + 0.5*một_phần) / tổng
    RiskLevel level = RiskLevel::Low;
    bool any_missing = false;  // có đặc điểm hoàn toàn không thấy -> khó vi phạm literal
};

// Tách claim độc lập thành các đặc điểm (cắt phần sau "comprising:" theo ';').
std::vector<std::string> split_claim_elements(const std::string& claim);

// So claim với mô tả sản phẩm -> báo cáo rủi ro.
RiskReport assess_risk(const std::string& claim, const std::string& product_desc);

const char* risk_label(RiskLevel level);

}  // namespace patent
