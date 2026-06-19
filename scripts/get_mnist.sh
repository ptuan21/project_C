#!/usr/bin/env bash
# Tải bộ dữ liệu MNIST và giải nén vào thư mục data/.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DIR"
cd "$DIR"

BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
FILES=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)

for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    echo "Đã có: $f"
    continue
  fi
  echo "Tải $f.gz ..."
  curl -fL "$BASE/$f.gz" -o "$f.gz"
  gunzip -f "$f.gz"
done

echo "Xong. Dữ liệu nằm trong: $DIR"
echo "Giờ chạy:  make demo && ./build/mnist_demo"
