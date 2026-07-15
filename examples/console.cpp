#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mlcpp/dl/autograd.hpp"
#include "mlcpp/ml/linreg.hpp"
#include "mlcpp/ml/logreg.hpp"
#include "mlcpp/core/matrix.hpp"
#include "mlcpp/data/mnist.hpp"
#include "mlcpp/dl/nn.hpp"
#include "mlcpp/prob/random.hpp"
#include "mlcpp/dl/sampling.hpp"
#include "mlcpp/prob/stats.hpp"
#include "mlcpp/data/tokenizer.hpp"
#include "mlcpp/models/transformer.hpp"
#include <fstream>

using namespace mlcpp;

// ---------- tiện ích nhập liệu ----------
static std::string ask(const std::string& msg) {
    std::printf("%s", msg.c_str());
    std::string s;
    std::getline(std::cin, s);
    return s;
}
static int ask_int(const std::string& msg, int def) {
    std::string s = ask(msg);
    try {
        return s.empty() ? def : std::stoi(s);
    } catch (...) {
        return def;
    }
}
static double ask_double(const std::string& msg, double def) {
    std::string s = ask(msg);
    try {
        return s.empty() ? def : std::stod(s);
    } catch (...) {
        return def;
    }
}
static std::vector<double> parse_row(const std::string& line) {
    std::vector<double> v;
    std::istringstream ss(line);
    double x;
    while (ss >> x) v.push_back(x);
    return v;
}

// ---------- 1) Ma trận ----------
static void demo_matrix() {
    std::printf("\n-- Nhân ma trận A x B --\n");
    int ar = ask_int("Số hàng A (mặc định 2): ", 2);
    int ac = ask_int("Số cột A (mặc định 2): ", 2);
    Matrix A(ar, ac, 0.0);
    std::printf("Nhập %d hàng của A, mỗi hàng %d số cách nhau bởi dấu cách:\n", ar, ac);
    for (int i = 0; i < ar; ++i) {
        auto row = parse_row(ask("  > "));
        for (int j = 0; j < ac && j < (int)row.size(); ++j) A(i, j) = row[j];
    }
    int bc = ask_int("Số cột B (số hàng B = số cột A): ", 2);
    Matrix B(ac, bc, 0.0);
    std::printf("Nhập %d hàng của B, mỗi hàng %d số:\n", ac, bc);
    for (int i = 0; i < ac; ++i) {
        auto row = parse_row(ask("  > "));
        for (int j = 0; j < bc && j < (int)row.size(); ++j) B(i, j) = row[j];
    }
    try {
        std::printf("\nKết quả:\n%s", A.matmul(B).to_string().c_str());
    } catch (const std::exception& e) {
        std::printf("Lỗi: %s\n", e.what());
    }
}

// ---------- 2) Phân phối + thống kê ----------
static void demo_stats() {
    std::printf("\n-- Lấy mẫu phân phối + thống kê --\n");
    std::printf("  1=Gaussian  2=Uniform  3=Exponential  4=Poisson\n");
    int kind = ask_int("Chọn (mặc định 1): ", 1);
    int n = ask_int("Số mẫu (mặc định 5000): ", 5000);
    Rng rng(2026);
    std::vector<double> v(n);
    double lo = -4, hi = 4;
    if (kind == 1) {
        double m = ask_double("mean (0): ", 0), s = ask_double("stddev (1): ", 1);
        for (int i = 0; i < n; ++i) v[i] = rng.gaussian(m, s);
        lo = m - 4 * s;
        hi = m + 4 * s;
    } else if (kind == 2) {
        double a = ask_double("a (0): ", 0), b = ask_double("b (1): ", 1);
        for (int i = 0; i < n; ++i) v[i] = rng.uniform(a, b);
        lo = a;
        hi = b;
    } else if (kind == 3) {
        double l = ask_double("lambda (1): ", 1);
        for (int i = 0; i < n; ++i) v[i] = rng.exponential(l);
        lo = 0;
        hi = 5.0 / l;
    } else {
        double l = ask_double("lambda (4): ", 4);
        for (int i = 0; i < n; ++i) v[i] = rng.poisson(l);
        lo = 0;
        hi = 3 * l;
    }
    // histogram
    int bins = 16;
    std::vector<int> cnt(bins, 0);
    double w = (hi - lo) / bins;
    for (double x : v) {
        int b = (int)((x - lo) / w);
        if (b >= 0 && b < bins) ++cnt[b];
    }
    int mx = 1;
    for (int c : cnt) mx = c > mx ? c : mx;
    for (int i = 0; i < bins; ++i) {
        std::printf("  %7.2f | ", lo + (i + 0.5) * w);
        for (int k = 0; k < (int)(50.0 * cnt[i] / mx); ++k) std::putchar('#');
        std::printf("\n");
    }
    std::printf("mean=%.4f  stddev=%.4f  median=%.4f  min=%.4f  max=%.4f\n", mean(v), stddev(v),
                median(v), min(v), max(v));
}

// ---------- 3) Linear regression ----------
static void demo_linreg() {
    std::printf("\n-- Linear regression: y = a*x + b + nhiễu --\n");
    double a = ask_double("hệ số a thật (4): ", 4), b = ask_double("hệ số b thật (7): ", 7);
    double noise = ask_double("độ nhiễu (1.0): ", 1.0);
    Rng rng(1);
    int N = 200;
    Matrix X(N, 1, 0.0), y(N, 1, 0.0);
    for (int i = 0; i < N; ++i) {
        double x = rng.uniform(-5, 5);
        X(i, 0) = x;
        y(i, 0) = a * x + b + rng.gaussian(0, noise);
    }
    LinearRegression lr;
    lr.fit_normal_equation(X, y);
    std::printf("Học được: a=%.3f  b=%.3f  (mse=%.4f)\n", lr.weights()(1, 0), lr.weights()(0, 0),
                lr.mse(X, y));
}

// ---------- 4) Logistic regression ----------
static void demo_logreg() {
    std::printf("\n-- Logistic regression: 2 cụm Gaussian --\n");
    Rng rng(7);
    int N = 400;
    Matrix X(N, 2, 0.0), y(N, 1, 0.0);
    for (int i = 0; i < N; ++i) {
        int lab = i % 2;
        double c = lab ? 2.0 : -2.0;
        X(i, 0) = c + rng.gaussian(0, 0.8);
        X(i, 1) = c + rng.gaussian(0, 0.8);
        y(i, 0) = lab;
    }
    LogisticRegression clf;
    clf.fit(X, y, 0.1, 2000);
    std::printf("Accuracy = %.1f%%\n", 100.0 * clf.accuracy(X, y));
    double px = ask_double("Thử điểm x (2): ", 2), py = ask_double("Thử điểm y (2): ", 2);
    Matrix probe = Matrix::from_rows({{px, py}});
    std::printf("P(y=1 | (%.1f,%.1f)) = %.3f\n", px, py, clf.predict_proba(probe)(0, 0));
}

// ---------- 5) MLP (vòng tròn phi tuyến) ----------
static void demo_mlp() {
    std::printf("\n-- MLP học ranh giới phi tuyến (2 vòng tròn) --\n");
    Rng rng(2026);
    auto make = [&](int n, Matrix& X, std::vector<int>& yy) {
        X = Matrix(n, 2, 0.0);
        yy.resize(n);
        for (int i = 0; i < n; ++i) {
            int lab = i % 2;
            double r = lab ? rng.uniform(1.5, 2.5) : rng.uniform(0, 1);
            double ang = rng.uniform(0, 6.2831853);
            X(i, 0) = r * std::cos(ang);
            X(i, 1) = r * std::sin(ang);
            yy[i] = lab;
        }
    };
    Matrix Xtr, Xte;
    std::vector<int> ytr, yte;
    make(600, Xtr, ytr);
    make(200, Xte, yte);
    Linear l1(2, 16, rng), l2(16, 2, rng);
    std::vector<Tensor> ps;
    for (auto& p : l1.params()) ps.push_back(p);
    for (auto& p : l2.params()) ps.push_back(p);
    Adam opt(ps, 0.01);
    auto acc = [&](const Matrix& X, const std::vector<int>& yy) {
        Tensor out = l2.forward(relu(l1.forward(Tensor(X))));
        int ok = 0;
        for (std::size_t i = 0; i < X.rows(); ++i) {
            int pr = out.value()(i, 0) >= out.value()(i, 1) ? 0 : 1;
            if (pr == yy[i]) ++ok;
        }
        return 100.0 * ok / X.rows();
    };
    int epochs = ask_int("Số epoch (mặc định 300): ", 300);
    for (int e = 1; e <= epochs; ++e) {
        opt.zero_grad();
        Tensor loss = cross_entropy(l2.forward(relu(l1.forward(Tensor(Xtr)))), ytr);
        loss.backward();
        opt.step();
        if (e % 100 == 0 || e == epochs)
            std::printf("  epoch %d | train=%.1f%% test=%.1f%%\n", e, acc(Xtr, ytr), acc(Xte, yte));
    }
}

// ---------- 6) MNIST ----------
static void demo_mnist() {
    std::printf("\n-- MNIST (cần data/, chạy bash scripts/get_mnist.sh) --\n");
    Mnist tr, te;
    try {
        int lim = ask_int("Giới hạn số mẫu train (0 = tất cả 60k): ", 0);
        tr = load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", lim);
        te = load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::printf("Lỗi: %s\nHãy chạy: bash scripts/get_mnist.sh\n", e.what());
        return;
    }
    Rng rng(2026);
    Linear l1(784, 128, rng), l2(128, 10, rng);
    std::vector<Tensor> ps;
    for (auto& p : l1.params()) ps.push_back(p);
    for (auto& p : l2.params()) ps.push_back(p);
    Adam opt(ps, 1e-3);
    int epochs = ask_int("Số epoch (mặc định 3): ", 3);
    std::size_t N = tr.images.rows(), bs = 100;
    auto eval = [&](const Mnist& d) {
        Tensor out = l2.forward(relu(l1.forward(Tensor(d.images))));
        int ok = 0;
        for (std::size_t i = 0; i < d.images.rows(); ++i) {
            int best = 0;
            for (std::size_t j = 1; j < 10; ++j)
                if (out.value()(i, j) > out.value()(i, best)) best = (int)j;
            if (best == d.labels[i]) ++ok;
        }
        return 100.0 * ok / d.images.rows();
    };
    for (int e = 1; e <= epochs; ++e) {
        for (std::size_t s = 0; s + bs <= N; s += bs) {
            std::vector<int> yb(tr.labels.begin() + s, tr.labels.begin() + s + bs);
            Matrix xb(bs, 784, 0.0);
            for (std::size_t i = 0; i < bs; ++i)
                for (std::size_t j = 0; j < 784; ++j) xb(i, j) = tr.images(s + i, j);
            opt.zero_grad();
            cross_entropy(l2.forward(relu(l1.forward(Tensor(xb)))), yb).backward();
            opt.step();
        }
        std::printf("  epoch %d | test acc = %.2f%%\n", e, eval(te));
    }
}

// ---------- 7) GPT char-level (train + sinh tương tác) ----------
static const char* kCorpus =
    "to be or not to be that is the question.\n"
    "all that glitters is not gold.\n"
    "the quick brown fox jumps over the lazy dog.\n"
    "knowledge is power and power is knowledge.\n"
    "to learn is to grow and to grow is to live.\n";

static void demo_gpt() {
    static std::unique_ptr<GPT> model;
    static CharTokenizer tok;
    static std::size_t T = 48;

    if (!model) {
        // Chọn corpus: đường dẫn người dùng nhập, ngược lại dùng corpus nhúng sẵn.
        std::string path = ask("Đường dẫn văn bản (vd data/vietnamese.txt, Enter = corpus mẫu): ");
        std::string text;
        std::ifstream f(path);
        if (!path.empty() && f) {
            std::stringstream ss;
            ss << f.rdbuf();
            text = ss.str();
            std::printf("[GPT] Dùng %s (%zu byte)\n", path.c_str(), text.size());
        } else {
            text = kCorpus;
            if (!path.empty()) std::printf("[GPT] Không mở được %s -> ", path.c_str());
            std::printf("[GPT] Dùng corpus nhúng sẵn.\n");
        }
        tok.build(text);
        std::vector<int> data = tok.encode(text);

        int steps = ask_int("Số bước huấn luyện (mặc định 800): ", 800);
        const int batch = 8;
        const int warmup = steps / 20 + 1;
        Rng rng(2026);
        model = std::make_unique<GPT>(GPTConfig{tok.vocab_size(), 96, 4, 2, T}, rng);
        auto params = model->params();
        double base_lr = 3e-4;
        Adam opt(params, base_lr, 0.9, 0.999, 1e-8, 0.01);  // AdamW

        std::printf("[GPT] Huấn luyện (clip + cosine LR + AdamW)...\n");
        for (int step = 1; step <= steps; ++step) {
            opt.set_lr(cosine_lr(step, steps, base_lr, warmup));
            opt.zero_grad();
            double ls = 0;
            for (int b = 0; b < batch; ++b) {
                std::size_t maxs = data.size() - T - 1;
                std::size_t s = (std::size_t)(rng.uniform(0, (double)maxs + 1));
                if (s > maxs) s = maxs;
                std::vector<int> x(data.begin() + s, data.begin() + s + T);
                std::vector<int> y(data.begin() + s + 1, data.begin() + s + 1 + T);
                Tensor loss = cross_entropy(model->forward(x), y);
                mul(loss, 1.0 / batch).backward();
                ls += loss.value()(0, 0);
            }
            clip_grad_norm(params, 1.0);
            opt.step();
            if (step % 200 == 0) std::printf("  bước %d | loss=%.3f\n", step, ls / batch);
        }
        std::printf("[GPT] Sẵn sàng!\n");
    }

    Rng rng(ask_int("seed ngẫu nhiên cho sinh (mặc định 1): ", 1));
    std::string seed = ask("Nhập văn bản mồi (mặc định ký tự đầu corpus): ");
    if (seed.empty()) seed = tok.token(0);
    SampleConfig sc;
    sc.temperature = ask_double("temperature (0.8): ", 0.8);
    sc.top_k = ask_int("top_k (40, 0=tắt): ", 40);
    sc.top_p = ask_double("top_p (0.9): ", 0.9);
    int n = ask_int("Số ký tự sinh (mặc định 300): ", 300);

    std::vector<int> ctx = tok.encode(seed);
    if (ctx.empty()) {
        std::printf("Văn bản mồi không có ký tự nào trong vocab.\n");
        return;
    }
    std::size_t vocab = tok.vocab_size();
    for (int i = 0; i < n; ++i) {
        std::vector<int> win = ctx;
        if (win.size() > T) win.erase(win.begin(), win.end() - T);
        Tensor logits = model->forward(win);
        std::size_t last = logits.rows() - 1;
        std::vector<double> row(vocab);
        for (std::size_t j = 0; j < vocab; ++j) row[j] = logits.value()(last, j);
        ctx.push_back(sample_logits(row, sc, rng));
    }
    std::printf("\n%s\n", tok.decode(ctx).c_str());
}

int main() {
    std::printf("========================================\n");
    std::printf("   mlcpp — Console thử nghiệm thư viện\n");
    std::printf("========================================\n");
    while (true) {
        std::printf(
            "\n  1) Ma trận: nhân A x B\n"
            "  2) Phân phối + thống kê (histogram)\n"
            "  3) Linear regression\n"
            "  4) Logistic regression\n"
            "  5) MLP học ranh giới phi tuyến\n"
            "  6) MNIST (cần tải dữ liệu)\n"
            "  7) GPT char-level: train + sinh văn bản\n"
            "  0) Thoát\n");
        std::string c = ask("Chọn: ");
        if (c == "0" || c == "q" || std::cin.eof()) break;
        if (c == "1")
            demo_matrix();
        else if (c == "2")
            demo_stats();
        else if (c == "3")
            demo_linreg();
        else if (c == "4")
            demo_logreg();
        else if (c == "5")
            demo_mlp();
        else if (c == "6")
            demo_mnist();
        else if (c == "7")
            demo_gpt();
        else
            std::printf("Lựa chọn không hợp lệ.\n");
    }
    std::printf("Tạm biệt!\n");
    return 0;
}
