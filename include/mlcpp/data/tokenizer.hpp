#pragma once
// mlcpp — tokenizer hỗ trợ 2 mức:
//   Char: mỗi ký tự UTF-8 là 1 token (đúng cho mọi văn bản, vocab nhỏ).
//   Word: mỗi âm tiết/từ là 1 token, dấu cách & dấu câu là token riêng
//         -> sinh ra từ tiếng Việt có thật, ngữ cảnh dài hơn, văn bản tự nhiên hơn.
#include <map>
#include <string>
#include <vector>

namespace mlcpp {

class Tokenizer {
public:
    enum class Level { Char, Word };

    // Xây từ vựng từ văn bản. Với Word, max_vocab>0 giới hạn còn các token hay gặp nhất
    // (phần còn lại quy về <unk>). max_vocab=0 = giữ tất cả.
    void build(const std::string& text, Level level = Level::Char, std::size_t max_vocab = 0);

    std::vector<int> encode(const std::string& text) const;  // token lạ -> <unk> (Word) hoặc bỏ (Char)
    std::string decode(const std::vector<int>& ids) const;   // nối token (tái lập đúng văn bản)

    std::size_t vocab_size() const { return itos_.size(); }
    const std::string& token(int id) const { return itos_[id]; }
    Level level() const { return level_; }
    int unk_id() const { return unk_; }  // -1 nếu không có

    // Tách chuỗi UTF-8 thành ký tự / thành từ (mỗi từ là run chữ cái; dấu cách & dấu câu riêng).
    static std::vector<std::string> split_utf8(const std::string& text);
    static std::vector<std::string> split_words(const std::string& text);

private:
    std::vector<std::string> tokens_of(const std::string& text) const;

    Level level_ = Level::Char;
    std::map<std::string, int> stoi_;
    std::vector<std::string> itos_;
    int unk_ = -1;
};

}  // namespace mlcpp
