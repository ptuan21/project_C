#!/usr/bin/env bash
# Tải bộ dữ liệu Tiny Shakespeare (~1MB) về data/input.txt để huấn luyện GPT.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DIR"
OUT="$DIR/input.txt"

if [ -f "$OUT" ]; then
  echo "Đã có: $OUT ($(wc -c < "$OUT") bytes)"
  exit 0
fi

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
echo "Tải Tiny Shakespeare..."
curl -fL "$URL" -o "$OUT"
echo "Xong: $OUT ($(wc -c < "$OUT") bytes)"
echo "Giờ chạy:  make demo && ./build/gpt_demo"
