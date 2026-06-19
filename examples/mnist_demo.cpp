// Demo Tầng 4: huấn luyện MLP nhận diện chữ số viết tay MNIST.
// Cần dữ liệu trong data/ — chạy `bash scripts/get_mnist.sh` để tải.
#include <cstdio>
#include <vector>

#include "mlcpp/autograd.hpp"
#include "mlcpp/mnist.hpp"
#include "mlcpp/nn.hpp"
#include "mlcpp/random.hpp"

using namespace mlcpp;

// Lấy một batch [start, start+bs) từ tập dữ liệu.
static Tensor make_batch(const Matrix& X, std::size_t start, std::size_t bs) {
    Matrix b(bs, X.cols(), 0.0);
    for (std::size_t i = 0; i < bs; ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) b(i, j) = X(start + i, j);
    return Tensor(b);
}

static double evaluate(Linear& l1, Linear& l2, const Mnist& d) {
    Tensor x(d.images);
    Tensor out = l2.forward(relu(l1.forward(x)));
    int correct = 0;
    for (std::size_t i = 0; i < d.images.rows(); ++i) {
        int best = 0;
        for (std::size_t j = 1; j < out.cols(); ++j)
            if (out.value()(i, j) > out.value()(i, best)) best = static_cast<int>(j);
        if (best == d.labels[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(d.images.rows());
}

int main() {
    std::printf("=== mlcpp demo: MNIST ===\n\n");

    Mnist train, test;
    try {
        train = load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
        test = load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::printf("Không tải được dữ liệu MNIST: %s\n", e.what());
        std::printf("Hãy chạy:  bash scripts/get_mnist.sh\n");
        return 1;
    }
    std::printf("Train: %zu mẫu, Test: %zu mẫu\n\n", train.images.rows(), test.images.rows());

    Rng rng(2026);
    Linear l1(784, 128, rng), l2(128, 10, rng);
    std::vector<Tensor> params;
    for (auto& p : l1.params()) params.push_back(p);
    for (auto& p : l2.params()) params.push_back(p);
    Adam opt(params, 1e-3);

    const std::size_t N = train.images.rows();
    const std::size_t bs = 100;
    const int epochs = 5;

    for (int e = 1; e <= epochs; ++e) {
        double epoch_loss = 0.0;
        int steps = 0;
        for (std::size_t start = 0; start + bs <= N; start += bs) {
            std::vector<int> yb(train.labels.begin() + start, train.labels.begin() + start + bs);
            opt.zero_grad();
            Tensor x = make_batch(train.images, start, bs);
            Tensor out = l2.forward(relu(l1.forward(x)));
            Tensor loss = cross_entropy(out, yb);
            loss.backward();
            opt.step();
            epoch_loss += loss.value()(0, 0);
            ++steps;
        }
        std::printf("epoch %d | loss tb = %.4f | test acc = %.2f%%\n", e, epoch_loss / steps,
                    100.0 * evaluate(l1, l2, test));
    }
    return 0;
}
