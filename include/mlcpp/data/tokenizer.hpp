#pragma once
// mlcpp — tokenizer mức ký tự UTF-8. Tách văn bản thành từng ký tự trọn vẹn
// (1–4 byte) thay vì từng byte, nên tiếng Việt có dấu được xử lý đúng và
// văn bản sinh ra luôn là UTF-8 hợp lệ.
#include <map>
#include <string>
#include <vector>

namespace mlcpp {

class CharTokenizer {
public:
    // Xây bộ từ vựng từ văn bản (theo thứ tự ký tự xuất hiện lần đầu).
    void build(const std::string& text);

    std::vector<int> encode(const std::string& text) const;  // bỏ qua ký tự lạ
    std::string decode(const std::vector<int>& ids) const;

    std::size_t vocab_size() const { return itos_.size(); }
    const std::string& token(int id) const { return itos_[id]; }
    bool contains(const std::string& ch) const { return stoi_.count(ch) > 0; }

    // Tiện ích: tách một chuỗi UTF-8 thành danh sách ký tự (mỗi phần tử 1 ký tự).
    static std::vector<std::string> split_utf8(const std::string& text);

private:
    std::map<std::string, int> stoi_;
    std::vector<std::string> itos_;
};

}  // namespace mlcpp
