#include "mlcpp/models/transformer.hpp"

#include <cmath>
#include <stdexcept>

namespace mlcpp {

// Khởi tạo ma trận trọng số (r x c) theo phân phối chuẩn std nhỏ (kiểu GPT-2: 0.02).
static Tensor init_weight(std::size_t r, std::size_t c, double std, Rng& rng) {
    Matrix m(r, c, 0.0);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) m(i, j) = rng.gaussian(0.0, std);
    return Tensor(std::move(m));
}

static Tensor zeros(std::size_t r, std::size_t c) { return Tensor(Matrix(r, c, 0.0)); }
static Tensor ones_row(std::size_t c) { return Tensor(Matrix(1, c, 1.0)); }

GPT::GPT(const GPTConfig& cfg, Rng& rng) : cfg_(cfg) {
    if (cfg.n_head == 0 || cfg.n_embd % cfg.n_head != 0)
        throw std::invalid_argument("GPT: n_embd phải chia hết cho n_head");
    const double s = 0.02;
    std::size_t E = cfg.n_embd;

    tok_emb_ = init_weight(cfg.vocab_size, E, s, rng);
    pos_emb_ = init_weight(cfg.block_size, E, s, rng);

    blocks_.reserve(cfg.n_layer);
    for (std::size_t l = 0; l < cfg.n_layer; ++l) {
        Block b{ones_row(E), zeros(1, E),
                init_weight(E, E, s, rng), zeros(1, E),  // Q
                init_weight(E, E, s, rng), zeros(1, E),  // K
                init_weight(E, E, s, rng), zeros(1, E),  // V
                init_weight(E, E, s, rng), zeros(1, E),  // O
                ones_row(E), zeros(1, E),
                init_weight(E, 4 * E, s, rng), zeros(1, 4 * E),
                init_weight(4 * E, E, s, rng), zeros(1, E)};
        blocks_.push_back(std::move(b));
    }

    lnf_g_ = ones_row(E);
    lnf_b_ = zeros(1, E);
    head_W_ = init_weight(E, cfg.vocab_size, s, rng);
    head_b_ = zeros(1, cfg.vocab_size);

    // Gom toàn bộ tham số để đưa vào optimizer.
    params_ = {tok_emb_, pos_emb_};
    for (auto& b : blocks_) {
        for (Tensor* p : {&b.ln1g, &b.ln1b, &b.Wq, &b.bq, &b.Wk, &b.bk, &b.Wv, &b.bv, &b.Wo, &b.bo,
                          &b.ln2g, &b.ln2b, &b.Wfc, &b.bfc, &b.Wproj, &b.bproj})
            params_.push_back(*p);
    }
    params_.push_back(lnf_g_);
    params_.push_back(lnf_b_);
    params_.push_back(head_W_);
    params_.push_back(head_b_);
}

Tensor GPT::block_forward(const Block& blk, const Tensor& x, const Tensor& mask) const {
    std::size_t hd = cfg_.n_embd / cfg_.n_head;  // số chiều mỗi đầu
    double scale = 1.0 / std::sqrt(static_cast<double>(hd));

    // --- Multi-head self-attention có mask nhân quả ---
    Tensor h = layernorm(x, blk.ln1g, blk.ln1b);
    Tensor Q = add(matmul(h, blk.Wq), blk.bq);  // T x n_embd
    Tensor K = add(matmul(h, blk.Wk), blk.bk);
    Tensor V = add(matmul(h, blk.Wv), blk.bv);

    std::vector<Tensor> heads;
    heads.reserve(cfg_.n_head);
    for (std::size_t hh = 0; hh < cfg_.n_head; ++hh) {
        std::size_t c0 = hh * hd;
        Tensor Qh = slice_cols(Q, c0, hd);  // T x hd
        Tensor Kh = slice_cols(K, c0, hd);
        Tensor Vh = slice_cols(V, c0, hd);
        Tensor scores = mul(matmul(Qh, transpose(Kh)), scale);  // T x T
        scores = add(scores, mask);                              // chặn nhìn về tương lai
        Tensor att = softmax_rows(scores);
        heads.push_back(matmul(att, Vh));                        // T x hd
    }
    Tensor ctx = concat_cols(heads);                             // T x n_embd
    Tensor attn_out = add(matmul(ctx, blk.Wo), blk.bo);
    Tensor x1 = add(x, attn_out);                                // residual

    // --- MLP ---
    Tensor h2 = layernorm(x1, blk.ln2g, blk.ln2b);
    Tensor m = add(matmul(gelu(add(matmul(h2, blk.Wfc), blk.bfc)), blk.Wproj), blk.bproj);
    return add(x1, m);                                     // residual
}

Tensor GPT::forward(const std::vector<int>& ids) const {
    std::size_t T = ids.size();
    if (T == 0 || T > cfg_.block_size)
        throw std::invalid_argument("GPT::forward: độ dài chuỗi phải trong [1, block_size]");

    // Mask nhân quả: 0 ở vị trí được phép, -1e9 ở phía trên đường chéo.
    Matrix mask_m(T, T, 0.0);
    for (std::size_t i = 0; i < T; ++i)
        for (std::size_t j = i + 1; j < T; ++j) mask_m(i, j) = -1e9;
    Tensor mask(std::move(mask_m));

    std::vector<int> pos(T);
    for (std::size_t t = 0; t < T; ++t) pos[t] = static_cast<int>(t);

    Tensor x = add(embedding(tok_emb_, ids), embedding(pos_emb_, pos));
    for (const auto& b : blocks_) x = block_forward(b, x, mask);
    x = layernorm(x, lnf_g_, lnf_b_);
    return add(matmul(x, head_W_), head_b_);  // logits T x vocab
}

}  // namespace mlcpp
