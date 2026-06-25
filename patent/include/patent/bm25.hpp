#pragma once
// patent — chỉ mục BM25 trên text patent (chuẩn vàng tìm kiếm văn bản, thuần C++).
#include <string>
#include <unordered_map>
#include <vector>

namespace patent {

class BM25 {
public:
    struct Hit {
        int doc;
        double score;
        std::vector<std::string> matched;  // các từ trong truy vấn khớp tài liệu này
    };

    // Xây chỉ mục từ danh sách text tài liệu (tự tokenize bên trong).
    void build(const std::vector<std::string>& docs);

    // Trả về top-K tài liệu điểm cao nhất cho truy vấn.
    std::vector<Hit> search(const std::string& query, int topk) const;

    // Tách text thành các từ khóa (chữ thường, bỏ stopword & token quá ngắn).
    static std::vector<std::string> terms(const std::string& text);

    std::size_t num_docs() const { return doclen_.size(); }

private:
    double k1_ = 1.5, b_ = 0.75;
    double avgdl_ = 0.0;
    std::vector<int> doclen_;
    std::unordered_map<std::string, std::vector<std::pair<int, int>>> postings_;  // term -> [(doc,tf)]
    std::unordered_map<std::string, double> idf_;
};

}  // namespace patent
