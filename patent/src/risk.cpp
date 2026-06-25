#include "patent/risk.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_set>

#include "patent/bm25.hpp"

namespace patent {

static std::string lower(const std::string& s) {
    std::string o = s;
    for (char& c : o)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return o;
}

static std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::vector<std::string> split_claim_elements(const std::string& claim) {
    // Lấy phần thân sau "comprising:" (hoặc dấu ':' đầu tiên); nếu không có, dùng cả claim.
    std::string low = lower(claim);
    std::string body = claim;
    std::size_t cp = low.find("comprising");
    std::size_t colon = (cp != std::string::npos) ? claim.find(':', cp) : claim.find(':');
    if (colon != std::string::npos) body = claim.substr(colon + 1);

    std::vector<std::string> out;
    std::string cur;
    for (char ch : body) {
        if (ch == ';') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    out.push_back(cur);

    std::vector<std::string> elems;
    for (auto& piece : out) {
        std::string e = trim(piece);
        // bỏ dấu chấm cuối claim
        while (!e.empty() && (e.back() == '.' || e.back() == ',')) e.pop_back();
        // bỏ tiền tố "and " / "and, "
        std::string el = lower(e);
        if (el.rfind("and ", 0) == 0) e = trim(e.substr(4));
        if (e.size() < 4) continue;
        if (BM25::terms(e).empty()) continue;  // không có từ khóa -> bỏ
        elems.push_back(e);
    }
    return elems;
}

RiskReport assess_risk(const std::string& claim, const std::string& product_desc) {
    RiskReport r;
    std::unordered_set<std::string> prod;
    for (auto& t : BM25::terms(product_desc)) prod.insert(t);

    for (auto& etext : split_claim_elements(claim)) {
        Element e;
        e.text = etext;
        std::unordered_set<std::string> seen;
        for (auto& t : BM25::terms(etext))
            if (seen.insert(t).second) e.keyterms.push_back(t);
        for (auto& t : e.keyterms)
            if (prod.count(t)) e.matched.push_back(t);
        e.coverage = e.keyterms.empty() ? 0.0
                                        : static_cast<double>(e.matched.size()) / e.keyterms.size();
        if (e.coverage >= 0.5) {
            e.verdict = "khop";
            ++r.n_match;
        } else if (e.coverage > 0.0) {
            e.verdict = "mot_phan";
            ++r.n_partial;
        } else {
            e.verdict = "khong";
            ++r.n_none;
            r.any_missing = true;
        }
        r.elements.push_back(std::move(e));
    }

    std::size_t total = r.elements.size();
    r.score = total ? (r.n_match + 0.5 * r.n_partial) / total : 0.0;
    if (r.score >= 0.75)
        r.level = RiskLevel::High;
    else if (r.score >= 0.45)
        r.level = RiskLevel::Medium;
    else
        r.level = RiskLevel::Low;
    return r;
}

const char* risk_label(RiskLevel level) {
    switch (level) {
        case RiskLevel::High: return "CAO";
        case RiskLevel::Medium: return "TRUNG BÌNH";
        default: return "THẤP";
    }
}

}  // namespace patent
