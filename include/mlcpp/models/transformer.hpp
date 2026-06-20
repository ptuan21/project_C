#pragma once
// mlcpp — GPT: kiến trúc Transformer decoder hoàn chỉnh, tái dùng autograd.
// Multi-head attention + weight tying (chia sẻ embedding đầu vào/đầu ra).
#include <vector>

#include "mlcpp/dl/autograd.hpp"
#include "mlcpp/prob/random.hpp"

namespace mlcpp {

struct GPTConfig {
    std::size_t vocab_size;
    std::size_t n_embd;      // số chiều embedding
    std::size_t n_head;      // số đầu attention (n_embd phải chia hết cho n_head)
    std::size_t n_layer;     // số khối Transformer
    std::size_t block_size;  // độ dài ngữ cảnh tối đa
};

class GPT {
public:
    GPT(const GPTConfig& cfg, Rng& rng);

    // Nhận chuỗi token ids (độ dài T <= block_size), trả về logits (T x vocab_size).
    Tensor forward(const std::vector<int>& ids) const;

    std::vector<Tensor> params() const { return params_; }
    const GPTConfig& config() const { return cfg_; }

private:
    struct Block {
        Tensor ln1g, ln1b;          // LayerNorm 1
        Tensor Wq, bq, Wk, bk, Wv, bv;  // chiếu Q, K, V
        Tensor Wo, bo;              // chiếu đầu ra attention
        Tensor ln2g, ln2b;          // LayerNorm 2
        Tensor Wfc, bfc, Wproj, bproj;  // MLP (n_embd -> 4n -> n_embd)
    };

    Tensor block_forward(const Block& blk, const Tensor& x, const Tensor& mask) const;

    GPTConfig cfg_;
    Tensor tok_emb_;       // vocab x n_embd
    Tensor pos_emb_;       // block_size x n_embd
    std::vector<Block> blocks_;
    Tensor lnf_g_, lnf_b_;  // LayerNorm cuối
    Tensor head_b_;         // bias đầu ra (trọng số đầu ra = tok_emb_ — weight tying)
    std::vector<Tensor> params_;
};

}  // namespace mlcpp
