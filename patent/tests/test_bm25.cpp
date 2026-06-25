#include <algorithm>
#include <string>
#include <vector>

#include "patent/bm25.hpp"
#include "test_framework.hpp"

using namespace patent;

static bool has(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

void test_bm25() {
    std::printf("[test_bm25]\n");

    // --- terms: hạ chữ thường, bỏ stopword & token ngắn ---
    auto t = BM25::terms("The Foldable Hat Rack, 2 arms.");
    CHECK(has(t, "foldable") && has(t, "rack"));
    CHECK(!has(t, "the"));   // stopword
    CHECK(!has(t, "2"));     // chữ số bị bỏ

    // --- xếp hạng: tài liệu liên quan phải lên đầu ---
    std::vector<std::string> docs = {
        "foldable hat rack with pivoting arms",         // 0
        "wireless bluetooth headphones noise cancelling",  // 1
        "collapsible coat rack with pivoting support arm",  // 2
    };
    BM25 bm;
    bm.build(docs);
    CHECK(bm.num_docs() == 3);

    auto hits = bm.search("pivoting rack", 5);
    CHECK(!hits.empty());
    CHECK(hits[0].doc == 0 || hits[0].doc == 2);  // 1 trong 2 doc về rack đứng đầu
    for (auto& h : hits) CHECK(h.doc != 1);        // doc headphones không khớp -> không xuất hiện
    CHECK(has(hits[0].matched, "rack") || has(hits[0].matched, "pivoting"));

    // truy vấn không khớp gì -> rỗng
    CHECK(bm.search("xyzzy quantum", 5).empty());
}
