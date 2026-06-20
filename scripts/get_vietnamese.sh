#!/usr/bin/env bash
# Gom corpus tiếng Việt lớn hơn: tải Truyện Kiều (Nguyễn Du, phạm vi công cộng),
# làm sạch tiền tố số dòng, rồi gộp với bộ ca dao - tục ngữ có sẵn.
# Kết quả: data/vietnamese_large.txt
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DIR"
RAW="$DIR/truyen_kieu_raw.txt"
KIEU="$DIR/truyen_kieu.txt"
OUT="$DIR/vietnamese_large.txt"

URL="https://raw.githubusercontent.com/duyet/truyenkieu-word2vec/master/truyen_kieu_data.txt"

echo "Tải Truyện Kiều..."
curl -fL --max-time 60 "$URL" -o "$RAW"

# Bỏ tiền tố "<số>.." và khoảng trắng thừa đầu dòng.
sed -E 's/^[0-9]+\.\.[[:space:]]*//' "$RAW" > "$KIEU"
echo "Truyện Kiều đã làm sạch: $(wc -l < "$KIEU") dòng, $(wc -c < "$KIEU") byte"

# Gộp ca dao - tục ngữ (nếu có) + Truyện Kiều.
: > "$OUT"
if [ -f "$DIR/vietnamese.txt" ]; then
  cat "$DIR/vietnamese.txt" >> "$OUT"
  printf '\n' >> "$OUT"
fi
cat "$KIEU" >> "$OUT"

rm -f "$RAW"
echo "Xong: $OUT ($(wc -c < "$OUT") byte, $(wc -l < "$OUT") dòng)"
echo "Train:  ./build/gpt_demo 3000 data/vietnamese_large.txt"
