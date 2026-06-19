# mlcpp — Thư viện ML/DL từ con số 0 (C++)

Một dự án học tập tự xây dựng thư viện máy học / học sâu bằng C++ thuần,
tối ưu cho **MacBook Air M1** (dùng Accelerate framework của Apple thay cho CUDA).

Mỗi tầng tái sử dụng code của tầng trước — cuối cùng bạn có một thư viện ML/DL
hoàn chỉnh do chính mình viết.

## Yêu cầu
- macOS + `clang++` (đã có sẵn qua Command Line Tools)
- `make` (có sẵn). CMake là tùy chọn: `brew install cmake`

## Build & chạy

```bash
make test              # biên dịch và chạy unit test (gồm gradient check)
make demo              # biên dịch mọi file trong examples/
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
  autograd.hpp   # autograd engine: Tensor + reverse-mode autodiff (matmul/add/relu/cross_entropy)
  nn.hpp         # Linear layer (He init), optimizer SGD & Adam
  mnist.hpp      # đọc dữ liệu MNIST định dạng IDX
src/             # phần cài đặt tương ứng
tests/           # khung test tối giản tự viết + unit test (gồm gradient check)
scripts/get_mnist.sh   # tải bộ dữ liệu MNIST về data/
examples/        # demo.cpp (T1-2), regression_demo.cpp (T3),
                 # mlp_demo.cpp + mnist_demo.cpp (T4)
```

## Ghi chú về Apple Silicon (M1)
- **Không có CUDA** (đó là của NVIDIA). Không cần GPU để học ML/DL.
- `Matrix::matmul` dùng `cblas_dgemm` của **Accelerate** — rất nhanh trên M1.
  Tắt bằng cách bỏ `-DMLCPP_USE_ACCELERATE` (sẽ rơi về vòng lặp ngây thơ để so sánh).
- Bài tập hiệu năng tốt: tự viết matmul có blocking + ARM NEON, rồi benchmark
  với Accelerate.

## Roadmap (lộ trình kết hợp)

- [x] **Tầng 1–2** — Ma trận, xác suất, thống kê
- [x] **Tầng 3a** — Đại số tuyến tính (LU solve/inverse) + linear/logistic regression
- [ ] **Tầng 3b** — Còn lại của ML cổ điển: K-Means, PCA, regularization (Ridge/Lasso)
- [x] **Tầng 4** — Deep learning: Tensor + autograd + Linear + ReLU + SGD/Adam, train MNIST (~97% test acc) *(đang ở đây)*
- [ ] **Tầng 5** — (tùy chọn) Transformer inference: load GPT-2, sinh văn bản

### Gợi ý các bước tiếp theo (Tầng 3)
1. Thêm `solve()` / nghịch đảo ma trận (phân rã LU hoặc Cholesky) vào `Matrix`.
2. Viết module `linear_regression`: nghiệm chuẩn (normal equation) **và** gradient descent.
3. Viết `logistic_regression` (tái dùng sigmoid + gradient descent).
4. Tải một dataset nhỏ qua `read_csv` và huấn luyện thử.

## Ý tưởng bài tập mở rộng
- Thêm kiểm định chi-square cho RNG (so histogram với phân phối lý thuyết).
- Thêm Monte Carlo ước lượng π và tích phân số.
- Benchmark matmul: ngây thơ vs Accelerate ở nhiều kích thước.
