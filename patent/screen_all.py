#!/usr/bin/env python3
# Quy trình 1 phát: 1 input (mô tả + ảnh) -> báo cáo rủi ro CẢ utility (claims) LẪN design (kiểu dáng).
#   python3 patent/screen_all.py --text "mô tả sản phẩm" [--image ảnh.jpg] [--topk 5]
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "patent", "design"))

STATUS = {"con_hieu_luc": "CÒN HIỆU LỰC", "het_han": "HẾT HẠN", "don_cong_bo": "ĐƠN (chưa cấp)"}


def utility(text, topk):
    print("\n" + "─" * 64)
    print("①  UTILITY (công năng) — patent claims liên quan")
    print("─" * 64)
    binp = os.path.join(ROOT, "build", "patent")
    if not os.path.exists(binp):
        print("  (chưa build CLI utility: chạy `make patent`)")
        return
    out = subprocess.run([binp, "screen", text, str(topk)], cwd=ROOT,
                         capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())


def design(image, text, topk):
    print("\n" + "─" * 64)
    print("②  DESIGN (kiểu dáng) — thiết kế tương đồng")
    print("─" * 64)
    import match_core
    try:
        rows = match_core.rank(image=image, text=text, topk=topk)
    except Exception as e:
        print("  lỗi:", e)
        return
    for r in rows:
        st = STATUS.get(r["status"], r["status"])
        print(f'  {r["score"]*100:5.1f}%  {r["id"]} | {r["owner"]} | {st}')
        print(f'           "{r["title"]}"  (ảnh {r["img"]*100:.0f}% · text {r["txt"]*100:.0f}%)')


def main():
    ap = argparse.ArgumentParser(description="Sàng lọc rủi ro patent: utility + design")
    ap.add_argument("--text", default="", help="mô tả / keyword sản phẩm")
    ap.add_argument("--image", default="", help="ảnh sản phẩm (cho design)")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    if not a.text and not a.image:
        ap.error("cần --text và/hoặc --image")

    print("=" * 64)
    print("  BÁO CÁO SÀNG LỌC PATENT")
    if a.text:
        print(f'  Sản phẩm: "{a.text}"')
    if a.image:
        print(f"  Ảnh: {a.image}")
    print("=" * 64)

    if a.text:
        utility(a.text, a.topk)
    if a.image:
        design(a.image, a.text, a.topk)

    print("\n" + "=" * 64)
    print("  ⚠️  Sàng lọc tự động trên dữ liệu mẫu — KHÔNG phải tư vấn pháp lý / FTO.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
