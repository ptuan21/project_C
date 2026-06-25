#!/usr/bin/env python3
# So 1 ảnh sản phẩm với design patent bằng CLIP (ảnh) + BM25 tiêu đề (text).
#   python3 patent/design/match_design.py <ảnh> [topk] ["mô tả/keyword sản phẩm"]
import os
import sys

import match_core

STATUS = {"con_hieu_luc": "CÒN HIỆU LỰC", "het_han": "HẾT HẠN", "don_cong_bo": "ĐƠN (chưa cấp)"}


def main():
    if len(sys.argv) < 2:
        print('Dùng: match_design.py <ảnh> [topk] ["mô tả sản phẩm"]')
        return 1
    img = sys.argv[1]
    rest = sys.argv[2:]
    topk = 5
    if rest and rest[0].isdigit():
        topk = int(rest[0])
        rest = rest[1:]
    text = " ".join(rest)
    if not os.path.exists(img):
        print("Không thấy ảnh:", img)
        return 1

    print(f"Ảnh: {img}" + (f'  + text: "{text}"' if text else "") + "\n")
    for r in match_core.rank(image=img, text=text, topk=topk):
        st = STATUS.get(r["status"], r["status"])
        extra = f' (ảnh {r["img"]*100:.0f}% + text {r["txt"]*100:.0f}%)' if text else ""
        print(f'  {r["score"]*100:5.1f}%  {r["id"]}  ({r["fig"]}) | {r["owner"]} | {st}{extra}')
        print(f'           "{r["title"]}"')
    print("\n(CLIP + BM25, offline — KHÔNG phải tư vấn pháp lý.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
