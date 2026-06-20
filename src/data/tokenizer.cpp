#include "mlcpp/data/tokenizer.hpp"

namespace mlcpp {

// Số byte của một ký tự UTF-8 dựa vào byte dẫn đầu.
static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;            // 0xxxxxxx  (ASCII)
    if ((c >> 5) == 0x6) return 2;     // 110xxxxx
    if ((c >> 4) == 0xE) return 3;     // 1110xxxx
    if ((c >> 3) == 0x1E) return 4;    // 11110xxx
    return 1;                          // byte không hợp lệ -> coi như 1
}

std::vector<std::string> CharTokenizer::split_utf8(const std::string& text) {
    std::vector<std::string> chars;
    std::size_t i = 0;
    while (i < text.size()) {
        int len = utf8_len(static_cast<unsigned char>(text[i]));
        if (i + len > text.size()) len = 1;  // chuỗi cụt -> lấy 1 byte
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

void CharTokenizer::build(const std::string& text) {
    stoi_.clear();
    itos_.clear();
    for (const auto& ch : split_utf8(text)) {
        if (stoi_.find(ch) == stoi_.end()) {
            stoi_[ch] = static_cast<int>(itos_.size());
            itos_.push_back(ch);
        }
    }
}

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& ch : split_utf8(text)) {
        auto it = stoi_.find(ch);
        if (it != stoi_.end()) ids.push_back(it->second);  // bỏ qua ký tự không có trong vocab
    }
    return ids;
}

std::string CharTokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids)
        if (id >= 0 && id < static_cast<int>(itos_.size())) out += itos_[id];
    return out;
}

}  // namespace mlcpp
