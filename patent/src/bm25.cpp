#include "patent/bm25.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "mlcpp/data/tokenizer.hpp"

namespace patent {

// Stopword tiếng Anh + boilerplate patent (xuất hiện khắp nơi -> nhiễu).
static const std::unordered_set<std::string>& stopwords() {
    static const std::unordered_set<std::string> s = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "is", "are", "be", "as",
        "at", "by", "on", "from", "that", "this", "it", "its", "which", "such", "said", "claim",
        "claims", "comprising", "comprises", "wherein", "having", "including", "one", "least",
        "plurality", "first", "second", "configured", "based", "least", "may", "can", "each",
        "into", "between", "thereof", "via", "use", "used", "device", "method", "system",
    };
    return s;
}

std::vector<std::string> BM25::terms(const std::string& text) {
    std::vector<std::string> out;
    for (auto& tok : mlcpp::Tokenizer::split_words(text)) {
        // chỉ giữ token là "từ": chữ cái, dài >= 2
        if (tok.size() < 2) continue;
        unsigned char c0 = static_cast<unsigned char>(tok[0]);
        bool is_word = (c0 >= 'a' && c0 <= 'z') || (c0 >= 'A' && c0 <= 'Z') || c0 >= 0x80;
        if (!is_word) continue;
        std::string w;
        w.reserve(tok.size());
        for (char ch : tok) {
            unsigned char u = static_cast<unsigned char>(ch);
            w += (u >= 'A' && u <= 'Z') ? static_cast<char>(u - 'A' + 'a') : ch;  // hạ chữ thường ASCII
        }
        if (stopwords().count(w)) continue;
        out.push_back(std::move(w));
    }
    return out;
}

void BM25::build(const std::vector<std::string>& docs) {
    std::size_t N = docs.size();
    doclen_.assign(N, 0);
    postings_.clear();
    idf_.clear();

    long long total_len = 0;
    for (std::size_t d = 0; d < N; ++d) {
        std::unordered_map<std::string, int> tf;
        for (auto& t : terms(docs[d])) ++tf[t];
        doclen_[d] = 0;
        for (auto& kv : tf) {
            postings_[kv.first].push_back({static_cast<int>(d), kv.second});
            doclen_[d] += kv.second;
        }
        total_len += doclen_[d];
    }
    avgdl_ = N ? static_cast<double>(total_len) / N : 0.0;

    for (auto& kv : postings_) {
        double df = static_cast<double>(kv.second.size());
        idf_[kv.first] = std::log(1.0 + (N - df + 0.5) / (df + 0.5));
    }
}

std::vector<BM25::Hit> BM25::search(const std::string& query, int topk) const {
    // từ khóa truy vấn (duy nhất)
    std::vector<std::string> q;
    {
        std::unordered_set<std::string> seen;
        for (auto& t : terms(query))
            if (seen.insert(t).second) q.push_back(t);
    }

    std::unordered_map<int, double> score;
    std::unordered_map<int, std::vector<std::string>> hits;
    for (auto& t : q) {
        auto it = postings_.find(t);
        if (it == postings_.end()) continue;
        double idf = idf_.at(t);
        for (auto& p : it->second) {
            int d = p.first, tf = p.second;
            double denom = tf + k1_ * (1.0 - b_ + b_ * doclen_[d] / avgdl_);
            score[d] += idf * (tf * (k1_ + 1.0)) / denom;
            hits[d].push_back(t);
        }
    }

    std::vector<Hit> res;
    res.reserve(score.size());
    for (auto& kv : score) res.push_back({kv.first, kv.second, hits[kv.first]});
    std::sort(res.begin(), res.end(), [](const Hit& a, const Hit& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.doc < b.doc;
    });
    if (static_cast<int>(res.size()) > topk) res.resize(topk);
    return res;
}

}  // namespace patent
