#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mlcpp/dl/autograd.hpp"
#include "mlcpp/dl/nn.hpp"
#include "mlcpp/prob/random.hpp"
#include "mlcpp/dl/sampling.hpp"
#include "mlcpp/data/tokenizer.hpp"
#include "mlcpp/models/transformer.hpp"

using namespace mlcpp;

static const char* kCorpus =
    "to be or not to be that is the question.\n"
    "all that glitters is not gold.\n"
    "the quick brown fox jumps over the lazy dog.\n"
    "knowledge is power and power is knowledge.\n"
    "to learn is to grow and to grow is to live.\n";

// Lấy hàng logits cuối của chuỗi từ tensor logits (T x vocab).
static std::vector<double> last_logits(const Tensor& logits, std::size_t vocab) {
    std::size_t last = logits.rows() - 1;
    std::vector<double> out(vocab);
    for (std::size_t j = 0; j < vocab; ++j) out[j] = logits.value()(last, j);
    return out;
}

int main(int argc, char** argv) {
    int steps = argc > 1 ? std::atoi(argv[1]) : 2000;
    const char* path = argc > 2 ? argv[2] : "data/input.txt";

    // 1) Nạp văn bản
    std::string text;
    std::ifstream f(path);
    if (f) {
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
        std::printf("Dùng %s (%zu byte)\n", path, text.size());
    } else {
        text = kCorpus;
        std::printf("Không thấy %s -> dùng corpus nhúng sẵn (%zu byte).\n", path, text.size());
        std::printf("Gợi ý: bash scripts/get_shakespeare.sh  hoặc  ./build/gpt_demo 2000 data/vietnamese.txt\n");
    }

    // 2) Tokenizer mức ký tự UTF-8 (đúng cho tiếng Việt)
    CharTokenizer tok;
    tok.build(text);
    std::size_t vocab = tok.vocab_size();
    std::vector<int> data = tok.encode(text);

    // 3) Chia train / validation (90 / 10)
    std::size_t n_train = data.size() * 9 / 10;
    std::vector<int> train(data.begin(), data.begin() + n_train);
    std::vector<int> val(data.begin() + n_train, data.end());
    std::printf("vocab=%zu | train=%zu | val=%zu token\n\n", vocab, train.size(), val.size());

    // 4) Mô hình + optimizer (AdamW)
    Rng rng(2026);
    GPTConfig cfg{vocab, 128, 4, 3, 64};  // n_embd=128, 4 đầu, 3 lớp, ngữ cảnh 64
    GPT model(cfg, rng);
    auto params = model.params();
    double base_lr = 3e-4;
    Adam opt(params, base_lr, 0.9, 0.999, 1e-8, /*weight_decay=*/0.01);

    std::size_t T = cfg.block_size;
    const int batch = 16;        // số chuỗi mỗi bước (gradient accumulation)
    const int warmup = steps / 20 + 1;
    const double grad_clip = 1.0;

    auto sample_chunk = [&](const std::vector<int>& src, std::vector<int>& x, std::vector<int>& y) {
        std::size_t max_start = src.size() - T - 1;
        std::size_t s = static_cast<std::size_t>(rng.uniform(0.0, static_cast<double>(max_start + 1)));
        if (s > max_start) s = max_start;
        x.assign(src.begin() + s, src.begin() + s + T);
        y.assign(src.begin() + s + 1, src.begin() + s + 1 + T);
    };

    auto perplexity = [&](const std::vector<int>& src, int chunks) {
        double total = 0.0;
        for (int c = 0; c < chunks; ++c) {
            std::vector<int> x, y;
            sample_chunk(src, x, y);
            total += cross_entropy(model.forward(x), y).value()(0, 0);
        }
        return std::exp(total / chunks);
    };

    // 5) Huấn luyện
    std::printf("Huấn luyện %d bước (batch=%d, clip=%.1f, AdamW + cosine LR)...\n", steps, batch,
                grad_clip);
    for (int step = 1; step <= steps; ++step) {
        opt.set_lr(cosine_lr(step, steps, base_lr, warmup));
        opt.zero_grad();
        double loss_sum = 0.0;
        for (int b = 0; b < batch; ++b) {
            std::vector<int> x, y;
            sample_chunk(train, x, y);
            Tensor loss = cross_entropy(model.forward(x), y);
            mul(loss, 1.0 / batch).backward();  // gradient trung bình theo batch
            loss_sum += loss.value()(0, 0);
        }
        double gnorm = clip_grad_norm(params, grad_clip);
        opt.step();

        if (step % 200 == 0 || step == 1) {
            std::printf("  bước %5d | loss=%.3f | lr=%.2e | |g|=%.2f | val ppl=%.2f\n", step,
                        loss_sum / batch, opt.lr(), gnorm, perplexity(val, 20));
        }
    }
    std::printf("\nTrain ppl=%.2f | Val ppl=%.2f  (gần nhau => không overfit)\n", perplexity(train, 40),
                perplexity(val, 40));

    // 6) Sinh văn bản với lấy mẫu top-k / top-p
    std::vector<int> ctx(data.begin(), data.begin() + std::min<std::size_t>(12, data.size()));

    SampleConfig sc;
    sc.temperature = 0.8;
    sc.top_k = 40;
    sc.top_p = 0.9;

    std::printf("\n--- Sinh 400 ký tự (temp=%.1f, top_k=%d, top_p=%.2f) ---\n", sc.temperature,
                sc.top_k, sc.top_p);
    for (int i = 0; i < 400; ++i) {
        std::vector<int> win = ctx;
        if (win.size() > T) win.erase(win.begin(), win.end() - T);
        int nx = sample_logits(last_logits(model.forward(win), vocab), sc, rng);
        ctx.push_back(nx);
    }
    std::printf("%s\n", tok.decode(ctx).c_str());
    return 0;
}
