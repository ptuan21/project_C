#include <string>
#include <vector>

#include "mlcpp/data/tokenizer.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_tokenizer() {
    std::printf("[test_tokenizer]\n");

    // --- Char level ---
    Tokenizer t;
    t.build("abcabc", Tokenizer::Level::Char);
    CHECK(t.vocab_size() == 3);
    CHECK(t.decode(t.encode("abcabc")) == "abcabc");

    std::string vi = "Bầu ơi thương lấy bí cùng";
    Tokenizer tv;
    tv.build(vi, Tokenizer::Level::Char);
    CHECK(tv.decode(tv.encode(vi)) == vi);  // round-trip UTF-8 hợp lệ

    Tokenizer ta;
    ta.build("áàá", Tokenizer::Level::Char);
    CHECK(ta.vocab_size() == 2);                 // 'á' và 'à'
    CHECK(ta.token(ta.encode("á")[0]) == "á");

    auto chars = Tokenizer::split_utf8("Đi");
    CHECK(chars.size() == 2 && chars[0] == "Đ");

    // --- Word level: tách âm tiết, dấu cách & dấu câu là token riêng ---
    auto words = Tokenizer::split_words("Y học là khoa học.");
    // "Y" " " "học" " " "là" " " "khoa" " " "học" "."  = 10 token
    CHECK(words.size() == 10);
    CHECK(words[0] == "Y" && words[1] == " " && words[2] == "học" && words[9] == ".");

    std::string sent = "công cha như núi, nghĩa mẹ như nước.";
    Tokenizer tw;
    tw.build(sent, Tokenizer::Level::Word);
    CHECK(tw.decode(tw.encode(sent)) == sent);   // tái lập đúng (kể cả dấu cách)
    CHECK(tw.unk_id() == 0);                       // Word luôn có <unk> ở id 0
    CHECK(tw.token(0) == "<unk>");
    // "như" xuất hiện 2 lần -> là 1 token duy nhất trong vocab
    CHECK(tw.encode("như").size() == 1);

    // Giới hạn vocab: chỉ giữ token hay gặp nhất, còn lại -> <unk>
    Tokenizer tcap;
    tcap.build("a a a b b c", Tokenizer::Level::Word, /*max_vocab=*/3);  // <unk> + 2 token
    CHECK(tcap.vocab_size() == 3);
    CHECK(tcap.encode("c")[0] == tcap.unk_id());   // 'c' hiếm -> <unk>
    CHECK(tcap.encode("a")[0] != tcap.unk_id());   // 'a' hay gặp -> giữ
}
