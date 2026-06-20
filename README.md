# mlcpp — GPT nhỏ tiếng Việt + lượng tử hóa int8

Một mô hình ngôn ngữ **GPT char-level** tự viết bằng C++ thuần (không framework),
tập trung vào: **model nhỏ – nhẹ – chạy được trên phần cứng yếu** mà vẫn sinh văn bản
tiếng Việt tự nhiên. Tối ưu cho **MacBook Air M1** (dùng Accelerate, không cần GPU/CUDA).

## Yêu cầu
- macOS + `clang++` (Command Line Tools)
- `make`, `python3` (để gom corpus). CMake là tùy chọn.

## Build & chạy

```bash
make test     # biên dịch + chạy unit test (gồm gradient check, lượng tử hóa)
make demo     # biên dịch examples/ (console, gpt_demo)

# Console tương tác: train -> sinh văn bản -> lượng tử hóa int8
./build/console

# Hoặc chạy thẳng: train rồi sinh văn bản + so sánh float vs int8
./build/gpt_demo 3000 data/vietnamese_large.txt
```

### Tạo corpus tiếng Việt
```bash
python3 scripts/get_vietnamese_corpus.py
```
Gom **đa lĩnh vực** vào `data/vietnamese_large.txt` (~2.8MB):
- Wikipedia tiếng Việt: khoa học, y học, văn học, chính trị, lịch sử, công nghệ...
- Truyện Kiều (Nguyễn Du)
- Ca dao – tục ngữ (`data/vietnamese.txt`)

## Cấu trúc

```
include/mlcpp/  +  src/  (mirror nhau)
  core/   matrix.hpp serialize.hpp quantize.hpp   # ma trận, lưu/nạp, lượng tử hóa int8
  prob/   random.hpp stats.hpp                    # RNG (init + sampling), thống kê (kiểm thử)
  data/   tokenizer.hpp                           # CharTokenizer mức ký tự UTF-8
  dl/     autograd.hpp nn.hpp sampling.hpp        # autograd, Linear/AdamW, top-k/top-p
  models/ transformer.hpp                         # GPT: multi-head attention + LayerNorm + GELU
  mlcpp.hpp                                       # umbrella header
tests/          # khung test tự viết + unit test (gradient check, quantize, tokenizer...)
examples/       # console.cpp (tương tác), gpt_demo.cpp (train + lượng tử hóa)
scripts/        # get_vietnamese_corpus.py (gom dữ liệu)
data/           # vietnamese.txt (ca dao), vietnamese_large.txt (corpus lớn)
```

## Kiến trúc GPT
`models/transformer.hpp` — Transformer decoder hoàn chỉnh, tái dùng autograd tự viết:
- Token + positional embedding
- **Multi-head causal self-attention** (tách/gộp đầu qua `slice_cols`/`concat_cols`)
- LayerNorm + GELU + MLP + residual
- Huấn luyện: AdamW + gradient clipping + cosine LR schedule + mini-batch + perplexity
- Sinh văn bản: temperature + top-k + top-p (nucleus)

## Lượng tử hóa int8 (model nhẹ để deploy)
- `quantize_per_col` — int8 đối xứng theo cột, sai số mỗi phần tử ≤ ½·scale (có unit test).
- `qmatmul_i8` — GEMM **số nguyên int32** (lượng tử cả activation): nhẹ 4× **và nhanh hơn**
  trên phần cứng không có BLAS.
- `quantize_params_in_place` — nén toàn bộ tham số GPT xuống int8 (**~4× nhẹ**), kiểm tra
  văn bản còn tự nhiên không (perplexity giữ gần như nguyên). Thử mục 3 trong `console`.

## Ghi chú Apple Silicon (M1)
- Không cần CUDA/GPU. `Matrix::matmul` dùng `cblas_dgemm` của **Accelerate** (rất nhanh).
- `make portable` build bản inference **thuần C++** (bỏ `-DMLCPP_USE_ACCELERATE`) cho bo mạch yếu.

## Hạn chế (nói thẳng)
Model char-level nhỏ học được **chính tả, ngữ pháp, nhịp điệu** tiếng Việt và văn phong
theo lĩnh vực, nhưng **chưa "thông minh" về ngữ nghĩa** (không trả lời câu hỏi mạch lạc) —
điều đó cần model lớn hơn nhiều bậc. Mục tiêu ở đây là *nhỏ, nhẹ, tự nhiên trong giới hạn*.
