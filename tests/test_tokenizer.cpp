#include <string>
#include <vector>

#include "mlcpp/data/tokenizer.hpp"
#include "test_framework.hpp"

using namespace mlcpp;

void test_tokenizer() {
    std::printf("[test_tokenizer]\n");

    // ASCII cơ bản
    CharTokenizer t;
    t.build("abcabc");
    CHECK(t.vocab_size() == 3);
    std::vector<int> ids = t.encode("abcabc");
    CHECK(ids.size() == 6);
    CHECK(ids[0] == ids[3] && ids[1] == ids[4]);
    CHECK(t.decode(ids) == "abcabc");

    // Tiếng Việt: mỗi ký tự có dấu là 1 token (dù nhiều byte)
    std::string vi = "Bầu ơi thương lấy bí cùng";
    CharTokenizer tv;
    tv.build(vi);
    // round-trip phải khớp hoàn toàn (UTF-8 hợp lệ)
    CHECK(tv.decode(tv.encode(vi)) == vi);

    // "á" (2 byte) phải là MỘT token, không phải hai
    CharTokenizer ta;
    ta.build("áàá");
    CHECK(ta.vocab_size() == 2);                 // chỉ 'á' và 'à'
    CHECK(ta.encode("áàá").size() == 3);          // ba ký tự
    CHECK(ta.token(ta.encode("á")[0]) == "á");    // token là chuỗi UTF-8 trọn vẹn

    // split_utf8 đếm ký tự, không đếm byte
    auto chars = CharTokenizer::split_utf8("Đi");  // 'Đ' (2 byte) + 'i'
    CHECK(chars.size() == 2);
    CHECK(chars[0] == "Đ");

    // encode bỏ qua ký tự lạ (Q, Z, @, #, 9 không có trong câu trên)
    CHECK(tv.encode("QZ@#9").size() == 0);
}
