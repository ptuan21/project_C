#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mlcpp/core/quantize.hpp"
#include "mlcpp/core/serialize.hpp"
#include "mlcpp/dl/autograd.hpp"
#include "mlcpp/dl/nn.hpp"
#include "mlcpp/dl/sampling.hpp"
#include "mlcpp/data/tokenizer.hpp"
#include "mlcpp/models/transformer.hpp"
#include "mlcpp/prob/random.hpp"

using namespace mlcpp;

// ---------- trạng thái ----------
static std::unique_ptr<GPT> g_model;
static Tokenizer g_tok;
static std::vector<int> g_data;
static std::size_t g_T = 64;
static bool g_quantized = false;

// ---------- nhập liệu ----------
static std::string ask(const std::string& msg) {
    std::printf("%s", msg.c_str());
    std::string s;
    std::getline(std::cin, s);
    return s;
}
static int ask_int(const std::string& msg, int def) {
    std::string s = ask(msg);
    try { return s.empty() ? def : std::stoi(s); } catch (...) { return def; }
}
static double ask_double(const std::string& msg, double def) {
    std::string s = ask(msg);
    try { return s.empty() ? def : std::stod(s); } catch (...) { return def; }
}

static const char* kCorpus =
    "Bầu ơi thương lấy bí cùng, tuy rằng khác giống nhưng chung một giàn.\n"
    "Công cha như núi Thái Sơn, nghĩa mẹ như nước trong nguồn chảy ra.\n"
    "Có công mài sắt có ngày nên kim. Đi một ngày đàng học một sàng khôn.\n";

// ---------- 1) Train ----------
static void do_train() {
    std::string path = ask("Đường dẫn corpus (Enter = data/vietnamese_large.txt): ");
    if (path.empty()) path = "data/vietnamese_large.txt";
    std::string text;
    std::ifstream f(path);
    if (f) {
        std::stringstream ss; ss << f.rdbuf(); text = ss.str();
        std::printf("Dùng %s (%zu byte)\n", path.c_str(), text.size());
    } else {
        text = kCorpus;
        std::printf("Không mở được %s -> dùng corpus mẫu nhỏ.\n", path.c_str());
    }

    bool word = ask_int("Mức token: 1=Từ (tự nhiên hơn), 0=Ký tự  [mặc định 1]: ", 1) != 0;
    if (word) g_tok.build(text, Tokenizer::Level::Word, 12000);
    else g_tok.build(text, Tokenizer::Level::Char);
    g_data = g_tok.encode(text);
    std::size_t vocab = g_tok.vocab_size();
    std::size_t n_train = g_data.size() * 9 / 10;
    std::vector<int> train(g_data.begin(), g_data.begin() + n_train);
    std::vector<int> val(g_data.begin() + n_train, g_data.end());
    std::printf("mức %s | vocab=%zu | train=%zu | val=%zu token\n",
                word ? "TỪ" : "ký tự", vocab, train.size(), val.size());

    int steps = ask_int("Số bước (mặc định 1500): ", 1500);
    Rng rng(2026);
    GPTConfig cfg = word ? GPTConfig{vocab, 192, 6, 4, g_T}   // word: model lớn hơn
                         : GPTConfig{vocab, 128, 4, 3, g_T};
    g_model = std::make_unique<GPT>(cfg, rng);
    g_quantized = false;
    auto params = g_model->params();
    double base_lr = 3e-4;
    Adam opt(params, base_lr, 0.9, 0.999, 1e-8, 0.01);
    const int batch = 16, warmup = steps / 20 + 1;

    auto chunk = [&](const std::vector<int>& src, std::vector<int>& x, std::vector<int>& y) {
        std::size_t maxs = src.size() - g_T - 1;
        std::size_t s = (std::size_t)(rng.uniform(0, (double)maxs + 1));
        if (s > maxs) s = maxs;
        x.assign(src.begin() + s, src.begin() + s + g_T);
        y.assign(src.begin() + s + 1, src.begin() + s + 1 + g_T);
    };
    auto ppl = [&](const std::vector<int>& src, int c) {
        double tot = 0;
        for (int i = 0; i < c; ++i) { std::vector<int> x, y; chunk(src, x, y);
            tot += cross_entropy(g_model->forward(x), y).value()(0, 0); }
        return std::exp(tot / c);
    };

    std::printf("Huấn luyện (AdamW + clip + cosine LR)...\n");
    for (int step = 1; step <= steps; ++step) {
        opt.set_lr(cosine_lr(step, steps, base_lr, warmup));
        opt.zero_grad();
        double ls = 0;
        for (int b = 0; b < batch; ++b) {
            std::vector<int> x, y; chunk(train, x, y);
            Tensor loss = cross_entropy(g_model->forward(x), y);
            mul(loss, 1.0 / batch).backward();
            ls += loss.value()(0, 0);
        }
        clip_grad_norm(params, 1.0);
        opt.step();
        if (step % 200 == 0 || step == 1)
            std::printf("  bước %5d | loss=%.3f | val ppl=%.2f\n", step, ls / batch, ppl(val, 20));
    }
    std::printf("Xong. Train ppl=%.2f | Val ppl=%.2f\n", ppl(train, 40), ppl(val, 40));
}

// ---------- 2) Sinh văn bản ----------
static void do_generate() {
    if (!g_model) { std::printf("Chưa có model — hãy Train trước (mục 1).\n"); return; }
    std::string seed = ask("Văn bản mồi (Enter = đầu corpus): ");
    SampleConfig sc;
    sc.temperature = ask_double("temperature (0.8): ", 0.8);
    sc.top_k = ask_int("top_k (40, 0=tắt): ", 40);
    sc.top_p = ask_double("top_p (0.9): ", 0.9);
    int n = ask_int("Số token sinh (mặc định 250): ", 250);
    Rng rng(ask_int("seed ngẫu nhiên (mặc định 1): ", 1));

    std::vector<int> ctx;
    if (seed.empty())
        ctx.assign(g_data.begin(), g_data.begin() + std::min<std::size_t>(16, g_data.size()));
    else
        ctx = g_tok.encode(seed);
    if (ctx.empty()) { std::printf("Mồi không có ký tự nào trong vocab.\n"); return; }

    std::size_t vocab = g_tok.vocab_size();
    for (int i = 0; i < n; ++i) {
        std::vector<int> win = ctx;
        if (win.size() > g_T) win.erase(win.begin(), win.end() - g_T);
        Tensor logits = g_model->forward(win);
        std::size_t last = logits.rows() - 1;
        std::vector<double> row(vocab);
        for (std::size_t j = 0; j < vocab; ++j) row[j] = logits.value()(last, j);
        ctx.push_back(sample_logits(row, sc, rng));
    }
    std::printf("\n%s%s\n", g_quantized ? "[int8] " : "", g_tok.decode(ctx).c_str());
}

// ---------- 3) Lượng tử hóa int8 ----------
static void do_quantize() {
    if (!g_model) { std::printf("Chưa có model — hãy Train trước (mục 1).\n"); return; }
    if (g_quantized) { std::printf("Model đã ở int8 rồi.\n"); return; }
    QuantStats st = quantize_params_in_place(g_model->params());
    g_quantized = true;
    std::printf("Đã lượng tử hóa int8 toàn bộ tham số.\n");
    std::printf("Trọng số: %.1f KB (float32) -> %.1f KB (int8)  = nhẹ %.1fx\n",
                st.float_bytes / 1024.0, st.int8_bytes / 1024.0, st.ratio());
    std::printf("Hãy thử lại mục 2 để xem văn bản còn tự nhiên không.\n");
}

int main() {
    std::printf("==============================================\n");
    std::printf("   mlcpp — GPT nhỏ: train / sinh / lượng tử hóa\n");
    std::printf("==============================================\n");
    while (true) {
        std::printf(
            "\n  1) Train GPT (chọn corpus + số bước)\n"
            "  2) Sinh văn bản\n"
            "  3) Lượng tử hóa int8 (làm model nhẹ ~4x)\n"
            "  0) Thoát\n");
        std::string c = ask("Chọn: ");
        if (c == "0" || c == "q" || std::cin.eof()) break;
        if (c == "1") do_train();
        else if (c == "2") do_generate();
        else if (c == "3") do_quantize();
        else std::printf("Lựa chọn không hợp lệ.\n");
    }
    std::printf("Tạm biệt!\n");
    return 0;
}
