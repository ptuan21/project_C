#include "patent/corpus.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace patent {

static std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(line);
    while (std::getline(ss, cur, '\t')) out.push_back(cur);
    return out;
}

static int to_int(const std::string& s) {
    try { return s.empty() ? 0 : std::stoi(s); } catch (...) { return 0; }
}

void Corpus::load(const std::string& dir) {
    dir_ = dir;
    metas_.clear();
    texts_.clear();
    id2idx_.clear();

    // 1) meta.tsv
    std::ifstream mf(dir + "/meta.tsv");
    if (!mf) throw std::runtime_error("Corpus: không mở được " + dir + "/meta.tsv (chạy patent_index.py trước)");
    std::string line;
    bool header = true;
    while (std::getline(mf, line)) {
        if (header) { header = false; continue; }  // bỏ dòng tiêu đề
        if (line.empty()) continue;
        auto f = split_tab(line);
        if (f.size() < 11) continue;
        PatentMeta m;
        m.id = f[0]; m.kind = f[1]; m.kindCode = f[2]; m.status = f[3];
        m.filing = to_int(f[4]); m.grant = to_int(f[5]); m.expiry = to_int(f[6]);
        m.n_images = to_int(f[7]); m.owner = f[8]; m.cpc = f[9]; m.title = f[10];
        id2idx_[m.id] = static_cast<int>(metas_.size());
        metas_.push_back(std::move(m));
    }

    // 2) docs.tsv (id <tab> text) — căn theo id
    texts_.assign(metas_.size(), std::string());
    std::ifstream df(dir + "/docs.tsv");
    if (!df) throw std::runtime_error("Corpus: không mở được " + dir + "/docs.tsv");
    while (std::getline(df, line)) {
        std::size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string id = line.substr(0, tab);
        auto it = id2idx_.find(id);
        if (it != id2idx_.end()) texts_[it->second] = line.substr(tab + 1);
    }
}

int Corpus::find(const std::string& id) const {
    auto it = id2idx_.find(id);
    return it == id2idx_.end() ? -1 : it->second;
}

std::string Corpus::claims_path(const std::string& id) const {
    return dir_ + "/claims/" + id + ".txt";
}

}  // namespace patent
