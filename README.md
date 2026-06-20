# mlcpp — Thư viện ML/DL từ con số 0 (C++)
Mỗi tầng tái sử dụng code của tầng trước — cuối cùng bạn có một thư viện ML/DL
hoàn chỉnh do chính mình viết.

## Yêu cầu
- macOS + `clang++` (đã có sẵn qua Command Line Tools)
- `make` (có sẵn). CMake là tùy chọn: `brew install cmake`

## Build & chạy

```bash
make test              # biên dịch và chạy unit test (gồm gradient check)
make demo              # biên dịch mọi file trong examples/
./build/console        # MENU TƯƠNG TÁC: thử mọi tầng, kể cả chat với GPT
./build/demo           # T1-2: ma trận, xác suất, thống kê
./build/regression_demo # T3: linear & logistic regression
./build/mlp_demo       # T4: MLP học ranh giới phi tuyến (chạy ngay)
make clean             # dọn dẹp
```

Huấn luyện MNIST (cần tải dữ liệu một lần):

```bash
bash scripts/get_mnist.sh   # tải ~12MB về data/
./build/mnist_demo          # ~3 giây/5 epoch trên M1, đạt ~97% test acc
```

GPT char-level — train Transformer rồi sinh văn bản:

```bash
bash scripts/get_shakespeare.sh         # tải ~1MB về data/input.txt (tùy chọn)
./build/gpt_demo                        # train 2000 bước (mặc định) trên data/input.txt nếu có
./build/gpt_demo 300                     # truyền số bước để thử nhanh
./build/gpt_demo 1500 data/vietnamese.txt   # train trên corpus TIẾNG VIỆT có sẵn
```

**Tiếng Việt:** dùng `CharTokenizer` mức **ký tự UTF-8** (tách mỗi ký tự có dấu
thành 1 token, không phải từng byte) nên văn bản sinh ra **luôn là UTF-8 hợp lệ**,
không còn byte hỏng.

- `data/vietnamese.txt` — ca dao – tục ngữ (~3KB, có sẵn). Corpus nhỏ → model dễ
  "học thuộc" (train ppl ≪ val ppl).
- `data/vietnamese_large.txt` — Truyện Kiều + ca dao (~143KB) sau khi chạy
  `bash scripts/get_vietnamese.sh`. Corpus lớn → **hết overfit** (train ppl ≈ val ppl)
  và văn bản sinh ra theo nhịp lục bát.

```bash
bash scripts/get_vietnamese.sh                  # gom corpus lớn (1 lần)
./build/gpt_demo 2000 data/vietnamese_large.txt # val perplexity ~5.6
```

Huấn luyện đã được tối ưu cho độ ổn định:
- **AdamW** (weight decay) + **cosine LR schedule** (warmup rồi giảm dần)
- **Gradient clipping** theo chuẩn L2 (chặn "nổ" gradient)
- **Mini-batch** qua gradient accumulation (gradient bớt nhiễu)
- **Perplexity** trên tập validation để đánh giá (train ≈ val => không overfit)
- Sinh văn bản: **temperature + top-k + top-p (nucleus)** cho câu trả lời mạch lạc, ổn định

Tham khảo kết quả: ~1200 bước trên Tiny Shakespeare đạt perplexity ~7.9.

Hoặc dùng CMake:

```bash
cmake -B build && cmake --build build
./build/run_tests && ./build/demo
```

## Cấu trúc

```
include/mlcpp/   # header công khai
  matrix.hpp     # ma trận dày đặc, nhân ma trận (qua Accelerate/BLAS)
  random.hpp     # RNG + lấy mẫu: uniform, Gaussian, exponential, Bernoulli, Poisson
  stats.hpp      # mean, variance, covariance, correlation, percentile, cov matrix
  csv.hpp        # đọc CSV số vào Matrix
  linalg.hpp     # giải hệ AX=B & nghịch đảo bằng phân rã LU (pivoting)
  linreg.hpp     # linear regression: normal equation + gradient descent
  logreg.hpp     # logistic regression: phân loại nhị phân bằng gradient descent
  autograd.hpp   # autograd engine: Tensor + reverse-mode autodiff
                 # matmul/add/relu/cross_entropy + softmax/layernorm/gelu/embedding/transpose
  nn.hpp         # Linear (He init), SGD & AdamW, gradient clipping, cosine LR schedule
  mnist.hpp      # đọc dữ liệu MNIST định dạng IDX
  transformer.hpp# GPT char-level: embedding + multi-head causal self-attention + LayerNorm + GELU + MLP
  sampling.hpp   # lấy mẫu xác suất: temperature, top-k, top-p (nucleus)
  tokenizer.hpp  # CharTokenizer mức ký tự UTF-8 (đúng cho tiếng Việt)
src/             # phần cài đặt tương ứng
tests/           # khung test tối giản tự viết + unit test (gồm gradient check)
scripts/get_mnist.sh        # tải bộ dữ liệu MNIST về data/
scripts/get_shakespeare.sh  # tải Tiny Shakespeare về data/input.txt
scripts/get_vietnamese.sh   # tải + làm sạch Truyện Kiều, gộp ca dao -> data/vietnamese_large.txt
examples/        # demo.cpp (T1-2), regression_demo.cpp (T3),
                 # mlp_demo.cpp + mnist_demo.cpp (T4), gpt_demo.cpp (T5)
```

