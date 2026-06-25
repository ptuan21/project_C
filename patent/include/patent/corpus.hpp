#pragma once
// patent — nạp index sinh bởi patent/index/patent_index.py:
//   meta.tsv (metadata + hiệu lực), docs.tsv (text tìm kiếm), claims/<id>.txt.
#include <map>
#include <string>
#include <vector>

namespace patent {

struct PatentMeta {
    std::string id, kind, kindCode, status, owner, cpc, title;
    int filing = 0, grant = 0, expiry = 0, n_images = 0;

    bool is_utility() const { return kind == "utility"; }
    bool granted() const { return status == "con_hieu_luc"; }      // đã cấp & còn hiệu lực
    bool pending() const { return status == "don_cong_bo"; }       // đơn chưa cấp
};

class Corpus {
public:
    // dir = thư mục index (vd "data/patent_index"). Ném runtime_error nếu thiếu file.
    void load(const std::string& dir);

    std::size_t size() const { return metas_.size(); }
    const PatentMeta& meta(int doc) const { return metas_[doc]; }
    const std::string& text(int doc) const { return texts_[doc]; }
    int find(const std::string& id) const;  // -> doc index hoặc -1
    std::string claims_path(const std::string& id) const;

private:
    std::string dir_;
    std::vector<PatentMeta> metas_;
    std::vector<std::string> texts_;
    std::map<std::string, int> id2idx_;
};

}  // namespace patent
