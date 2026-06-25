// patent — CLI tra cứu & sàng lọc rủi ro vi phạm patent (P1: tìm kiếm).
//   ./build/patent search "<mô tả sản phẩm>" [topk]
//   ./build/patent show <id>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "patent/bm25.hpp"
#include "patent/corpus.hpp"
#include "patent/risk.hpp"

using namespace patent;

static const char* INDEX_DIR = "data/patent_index";

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Lấy claim 1 (thường là claim độc lập — "bảo hộ cái gì") từ file claims.
// maxlen lớn để phân tích rủi ro (cần claim đầy đủ); nhỏ để hiển thị tóm tắt.
static std::string first_claim(const std::string& claims_file, std::size_t maxlen = 600) {
    std::size_t pos = claims_file.find("CLAIMS:");
    std::string body = pos == std::string::npos ? claims_file : claims_file.substr(pos + 7);
    std::istringstream ss(body);
    std::string line, out;
    bool started = false;
    while (std::getline(ss, line)) {
        // dòng bắt đầu một claim mới: "<số> ." hoặc "<số>."
        std::size_t i = 0;
        while (i < line.size() && line[i] == ' ') ++i;
        std::size_t j = i;
        while (j < line.size() && line[j] >= '0' && line[j] <= '9') ++j;
        bool is_claim_start = (j > i && j < line.size() && (line[j] == '.' || line[j] == ' '));
        std::string num = line.substr(i, j - i);
        if (is_claim_start && num == "1") { started = true; }
        else if (is_claim_start && started) break;  // sang claim 2 -> dừng
        if (started) { out += line; out += ' '; }
    }
    if (out.empty()) out = body.substr(0, maxlen);
    if (out.size() > maxlen) out = out.substr(0, maxlen) + "...";
    return out;
}

static std::string status_label(const PatentMeta& m) {
    if (m.status == "con_hieu_luc") return "CÒN HIỆU LỰC (hết hạn ~" + std::to_string(m.expiry) + ")";
    if (m.status == "don_cong_bo") return "ĐƠN công bố (CHƯA cấp)";
    if (m.status == "het_han") return "HẾT HẠN";
    return m.status;
}

static int cmd_search(Corpus& c, const std::string& query, int topk) {
    std::printf("Tìm patent utility liên quan: \"%s\"\n", query.c_str());

    std::vector<std::string> texts(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) texts[i] = c.text(static_cast<int>(i));
    BM25 bm;
    bm.build(texts);

    // lấy dư rồi lọc utility (design phân tích bằng ảnh, không qua text)
    auto hits = bm.search(query, topk * 4);
    int shown = 0;
    for (auto& h : hits) {
        const PatentMeta& m = c.meta(h.doc);
        if (!m.is_utility()) continue;
        ++shown;
        std::printf("\n#%d  %s  (điểm %.1f)\n", shown, m.id.c_str(), h.score);
        std::printf("    %s | nộp %d, cấp %d | %s\n", m.owner.empty() ? "(không rõ chủ)" : m.owner.c_str(),
                    m.filing, m.grant, status_label(m).c_str());
        std::printf("    \"%s\"\n", m.title.c_str());
        std::string matched;
        for (std::size_t k = 0; k < h.matched.size(); ++k)
            matched += (k ? ", " : "") + h.matched[k];
        std::printf("    Khớp: %s\n", matched.c_str());
        std::string c1 = first_claim(read_file(c.claims_path(m.id)));
        std::printf("    Claim 1 (bảo hộ): %s\n", c1.c_str());
        if (shown >= topk) break;
    }
    if (shown == 0) std::printf("\nKhông tìm thấy patent utility nào khớp.\n");
    std::printf("\n(Sàng lọc tự động — không phải tư vấn pháp lý.)\n");
    return 0;
}

static int cmd_show(Corpus& c, const std::string& id) {
    int d = c.find(id);
    if (d < 0) { std::printf("Không có patent %s trong index.\n", id.c_str()); return 1; }
    const PatentMeta& m = c.meta(d);
    std::printf("== %s ==\n", m.id.c_str());
    std::printf("Tiêu đề : %s\n", m.title.c_str());
    std::printf("Loại    : %s (%s)\n", m.kind.c_str(), m.kindCode.c_str());
    std::printf("Chủ sở hữu: %s\n", m.owner.c_str());
    std::printf("Ngày nộp: %d | cấp: %d | hiệu lực: %s\n", m.filing, m.grant, status_label(m).c_str());
    std::printf("CPC     : %s\n", m.cpc.c_str());
    std::printf("Bản vẽ  : %d ảnh\n\n", m.n_images);
    std::printf("%s\n", read_file(c.claims_path(m.id)).c_str());
    return 0;
}

static int cmd_risk(Corpus& c, const std::string& id, const std::string& desc) {
    int d = c.find(id);
    if (d < 0) { std::printf("Không có patent %s trong index.\n", id.c_str()); return 1; }
    const PatentMeta& m = c.meta(d);
    std::string claim = first_claim(read_file(c.claims_path(id)), 100000);  // claim đầy đủ

    std::printf("Phân tích rủi ro: %s  vs sản phẩm:\n  \"%s\"\n", id.c_str(), desc.c_str());
    std::printf("Patent: \"%s\" | %s | %s\n\n", m.title.c_str(), m.owner.c_str(),
                status_label(m).c_str());

    RiskReport r = assess_risk(claim, desc);
    std::printf("Claim 1 -> %zu đặc điểm:\n", r.elements.size());
    int i = 0;
    for (const auto& e : r.elements) {
        const char* tag = e.verdict == "khop" ? "KHỚP    " : e.verdict == "mot_phan" ? "MỘT PHẦN" : "KHÔNG   ";
        std::string mt;
        for (std::size_t k = 0; k < e.matched.size(); ++k) mt += (k ? ", " : "") + e.matched[k];
        std::string txt = e.text.size() > 90 ? e.text.substr(0, 90) + "..." : e.text;
        std::printf("  [%s] #%d %s\n", tag, ++i, txt.c_str());
        if (!mt.empty()) std::printf("             khớp: %s\n", mt.c_str());
    }
    std::printf("\n=> RỦI RO: %s  (khớp %d, một phần %d, thiếu %d / %zu đặc điểm; điểm %.2f)\n",
                risk_label(r.level), r.n_match, r.n_partial, r.n_none, r.elements.size(), r.score);
    if (r.any_missing)
        std::printf("   Lưu ý: có đặc điểm KHÔNG thấy -> khó vi phạm literal (vẫn nên xét doctrine of equivalents).\n");
    std::printf("\n(Sàng lọc bằng từ khóa — KHÔNG phải tư vấn pháp lý. Claim chart chuẩn cần luật sư.)\n");
    return 0;
}

// screen = quy trình 1 phát "check trước khi launch": tìm patent liên quan + chấm rủi ro từng cái.
static int cmd_screen(Corpus& c, const std::string& desc, int topk) {
    std::printf("SCREEN — kiểm tra rủi ro patent cho sản phẩm:\n  \"%s\"\n\n", desc.c_str());

    std::vector<std::string> texts(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) texts[i] = c.text(static_cast<int>(i));
    BM25 bm;
    bm.build(texts);

    struct Row { PatentMeta m; RiskReport r; double rel; };
    std::vector<Row> rows;
    for (auto& h : bm.search(desc, topk * 4)) {
        const PatentMeta& m = c.meta(h.doc);
        if (!m.is_utility()) continue;
        RiskReport r = assess_risk(first_claim(read_file(c.claims_path(m.id)), 100000), desc);
        rows.push_back({m, r, h.score});
        if (static_cast<int>(rows.size()) >= topk) break;
    }
    if (rows.empty()) { std::printf("Không tìm thấy patent utility liên quan.\n"); return 0; }

    auto print_group = [&](RiskLevel lvl, const char* head) {
        std::vector<const Row*> g;
        for (auto& r : rows) if (r.r.level == lvl) g.push_back(&r);
        if (g.empty()) return;
        std::sort(g.begin(), g.end(), [](const Row* a, const Row* b) { return a->r.score > b->r.score; });
        std::printf("%s (%zu):\n", head, g.size());
        for (auto* r : g) {
            std::printf("  %s | %s | %s\n", r->m.id.c_str(),
                        r->m.owner.empty() ? "(không rõ chủ)" : r->m.owner.c_str(),
                        status_label(r->m).c_str());
            std::printf("      \"%s\"\n", r->m.title.c_str());
            std::printf("      khớp %d/%zu đặc điểm claim 1 (điểm %.2f) — `patent risk %s ...` để xem chi tiết\n",
                        r->r.n_match, r->r.elements.size(), r->r.score, r->m.id.c_str());
        }
        std::printf("\n");
    };
    print_group(RiskLevel::High, "⚠️  RỦI RO CAO");
    print_group(RiskLevel::Medium, "🟡 RỦI RO TRUNG BÌNH");
    print_group(RiskLevel::Low, "✅ Liên quan nhưng rủi ro THẤP");

    std::printf("(Sàng lọc tự động trên %zu patent mẫu — KHÔNG phải tư vấn pháp lý / FTO.)\n", c.size());
    return 0;
}

static std::string jesc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n' || c == '\r' || c == '\t') o += ' ';
        else o += c;
    }
    return o;
}

// Xuất kết quả screen ở dạng JSON cho giao diện HTML.
static int cmd_screen_json(Corpus& c, const std::string& desc, int topk) {
    std::vector<std::string> texts(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) texts[i] = c.text(static_cast<int>(i));
    BM25 bm;
    bm.build(texts);
    std::printf("[");
    bool firstrow = true;
    int shown = 0;
    for (auto& h : bm.search(desc, topk * 4)) {
        const PatentMeta& m = c.meta(h.doc);
        if (!m.is_utility()) continue;
        RiskReport r = assess_risk(first_claim(read_file(c.claims_path(m.id)), 100000), desc);
        std::string c1 = first_claim(read_file(c.claims_path(m.id)), 240);
        std::printf("%s\n{\"id\":\"%s\",\"owner\":\"%s\",\"title\":\"%s\",\"status\":\"%s\","
                    "\"filing\":%d,\"grant\":%d,\"expiry\":%d,\"risk\":\"%s\",\"n_match\":%d,"
                    "\"n_elem\":%zu,\"score\":%.3f,\"claim1\":\"%s\"}",
                    firstrow ? "" : ",", m.id.c_str(), jesc(m.owner).c_str(), jesc(m.title).c_str(),
                    m.status.c_str(), m.filing, m.grant, m.expiry, risk_label(r.level), r.n_match,
                    r.elements.size(), r.score, jesc(c1).c_str());
        firstrow = false;
        if (++shown >= topk) break;
    }
    std::printf("\n]\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("Dùng:\n  ./build/patent screen \"<mô tả sản phẩm>\" [topk]   (tìm + chấm rủi ro)\n"
                    "  ./build/patent search \"<mô tả sản phẩm>\" [topk]\n"
                    "  ./build/patent show <id>\n"
                    "  ./build/patent risk <id> \"<mô tả sản phẩm>\"\n");
        return 1;
    }
    std::string cmd = argv[1];
    Corpus c;
    try {
        c.load(INDEX_DIR);
    } catch (const std::exception& e) {
        std::printf("Lỗi: %s\n", e.what());
        return 1;
    }

    if (cmd == "screen") {
        int topk = argc > 3 ? std::atoi(argv[3]) : 8;
        return cmd_screen(c, argv[2], topk);
    }
    if (cmd == "screen-json") {
        int topk = argc > 3 ? std::atoi(argv[3]) : 8;
        return cmd_screen_json(c, argv[2], topk);
    }
    if (cmd == "search") {
        int topk = argc > 3 ? std::atoi(argv[3]) : 8;
        return cmd_search(c, argv[2], topk);
    }
    if (cmd == "show") return cmd_show(c, argv[2]);
    if (cmd == "risk") {
        if (argc < 4) { std::printf("Dùng: ./build/patent risk <id> \"<mô tả>\"\n"); return 1; }
        return cmd_risk(c, argv[2], argv[3]);
    }
    std::printf("Lệnh không hợp lệ: %s\n", cmd.c_str());
    return 1;
}
