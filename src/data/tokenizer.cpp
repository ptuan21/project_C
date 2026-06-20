#include "mlcpp/data/tokenizer.hpp"

#include <algorithm>
#include <unordered_map>

namespace mlcpp {

// Số byte của một ký tự UTF-8 dựa vào byte dẫn đầu.
static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;            // 0xxxxxxx  (ASCII)
    if ((c >> 5) == 0x6) return 2;     // 110xxxxx
    if ((c >> 4) == 0xE) return 3;     // 1110xxxx
    if ((c >> 3) == 0x1E) return 4;    // 11110xxx
    return 1;                          // byte không hợp lệ -> coi như 1
}

std::vector<std::string> Tokenizer::split_utf8(const std::string& text) {
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

// "Ký tự chữ": đa byte (chữ có dấu) hoặc chữ cái ASCII a-z/A-Z.
static bool is_word_char(const std::string& ch) {
    if (ch.size() > 1) return true;
    unsigned char c = static_cast<unsigned char>(ch[0]);
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

std::vector<std::string> Tokenizer::split_words(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (const auto& ch : split_utf8(text)) {
        if (is_word_char(ch)) {
            cur += ch;  // gộp vào từ đang dở
        } else {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            out.push_back(ch);  // dấu cách / dấu câu / chữ số = token riêng
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<std::string> Tokenizer::tokens_of(const std::string& text) const {
    return level_ == Level::Word ? split_words(text) : split_utf8(text);
}

void Tokenizer::build(const std::string& text, Level level, std::size_t max_vocab) {
    level_ = level;
    stoi_.clear();
    itos_.clear();
    unk_ = -1;

    std::vector<std::string> toks = tokens_of(text);

    if (level_ == Level::Char) {
        // Thứ tự xuất hiện lần đầu; mọi ký tự đều có trong vocab nên không cần <unk>.
        for (const auto& t : toks)
            if (stoi_.find(t) == stoi_.end()) {
                stoi_[t] = static_cast<int>(itos_.size());
                itos_.push_back(t);
            }
        return;
    }

    // Word: đếm tần suất, giữ token hay gặp nhất, phần còn lại -> <unk>.
    std::unordered_map<std::string, std::size_t> freq;
    for (const auto& t : toks) ++freq[t];

    std::vector<std::pair<std::string, std::size_t>> items(freq.begin(), freq.end());
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;  // tần suất giảm dần
        return a.first < b.first;                               // ổn định
    });

    unk_ = 0;
    stoi_["<unk>"] = 0;
    itos_.push_back("<unk>");
    for (const auto& it : items) {
        if (max_vocab > 0 && itos_.size() >= max_vocab) break;
        stoi_[it.first] = static_cast<int>(itos_.size());
        itos_.push_back(it.first);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& t : tokens_of(text)) {
        auto it = stoi_.find(t);
        if (it != stoi_.end())
            ids.push_back(it->second);
        else if (unk_ >= 0)
            ids.push_back(unk_);  // Word: token lạ -> <unk>; Char: bỏ qua
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(itos_.size())) continue;
        if (id == unk_) continue;  // không in <unk>
        out += itos_[id];
    }
    return out;
}

}  // namespace mlcpp
