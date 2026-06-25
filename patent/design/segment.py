#!/usr/bin/env python3
# patent/design — tách trang bản vẽ patent thành FIGURE riêng và TEXT riêng (offline).
# Thuật toán: (1) tách dòng text (nối ký tự ngang -> khối thấp&rộng), loại ra;
#             (2) bỏ nhiễu nhỏ; (3) closing mạnh để nối trọn nét figure (không cắt vụn).
#
#   python3 segment.py <patent_id>          # tách 1 patent -> figures/ + text/
#   python3 segment.py --debug <patent_id>  # vẽ overlay khung (xanh=figure, đỏ=text) để kiểm
#   python3 segment.py --all [limit]
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES = os.path.join(ROOT, "data", "uspto_out", "images")
META = os.path.join(ROOT, "data", "patent_index", "meta.tsv")
OUT = os.path.join(ROOT, "data", "patent_figures")


def _bands(content, H, gap_tol):
    """Gom các hàng 'có nội dung' thành dải dọc, cho phép khe trống nhỏ <= gap_tol."""
    bands = []
    i = 0
    while i < H:
        if not content[i]:
            i += 1
            continue
        start = last = i
        j = i + 1
        while j < H:
            if content[j]:
                last = j
                j += 1
            else:
                k = j
                while k < H and not content[k]:
                    k += 1
                if k < H and (k - j) <= gap_tol:
                    j = k  # bắc qua khe nhỏ -> cùng dải
                else:
                    break
        bands.append((start, last + 1))
        i = last + 1
    return bands


def segment_page(path):
    """Tách theo CHIẾU: dải ink lớn ở giữa = figure (lấy trọn); dải mỏng trên/dưới = text.
    Trả về (figures, texts) bbox (x0,y0,x1,y1) ở toạ độ ảnh GỐC."""
    arr = np.asarray(Image.open(path).convert("L"))
    H, W = arr.shape
    ink = arr < 200  # rộng tay để bắt cả nét mảnh/chống răng cưa

    row = ink.sum(axis=1)
    content = row > max(3, int(W * 0.002))
    bands = _bands(content, H, gap_tol=int(H * 0.022))

    figs, texts = [], []
    for y0, y1 in bands:
        col = ink[y0:y1].sum(axis=0)
        xs = np.where(col > max(2, int((y1 - y0) * 0.01)))[0]
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        box = (x0, int(y0), x1, int(y1))
        h, w = y1 - y0, x1 - x0
        if h >= H * 0.10 and w >= W * 0.15:
            figs.append(box)        # dải lớn -> figure (trọn vẹn)
        elif h <= H * 0.06:
            texts.append(box)       # dải mỏng -> header/caption/dòng chữ
        else:
            figs.append(box)        # trung bình -> coi là figure
    return figs, texts, (W, H)


def save_crops(pid, pad_frac=0.015, skip_existing=False):
    src = os.path.join(IMAGES, pid)
    if not os.path.isdir(src):
        return 0, 0
    fig_dir = os.path.join(OUT, "figures", pid)
    txt_dir = os.path.join(OUT, "text", pid)
    if skip_existing and os.path.isdir(fig_dir) and any(
            f.endswith(".png") for f in os.listdir(fig_dir)):
        return -1, -1  # đã tách trước đó -> bỏ qua
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    nf = nt = 0
    for page in sorted(f for f in os.listdir(src) if f.lower().endswith(".png")):
        path = os.path.join(src, page)
        figs, texts, (W, H) = segment_page(path)
        im = Image.open(path).convert("L")
        pad = int(min(W, H) * pad_frac)
        for b in figs:
            im.crop((max(0, b[0] - pad), max(0, b[1] - pad), min(W, b[2] + pad),
                     min(H, b[3] + pad))).save(os.path.join(fig_dir, f"{nf:03d}.png"))
            nf += 1
        for b in texts:
            im.crop((max(0, b[0] - 3), max(0, b[1] - 3), min(W, b[2] + 3),
                     min(H, b[3] + 3))).save(os.path.join(txt_dir, f"{nt:03d}.png"))
            nt += 1
    return nf, nt


def save_debug(pid):
    src = os.path.join(IMAGES, pid)
    out = os.path.join(OUT, "debug", pid)
    os.makedirs(out, exist_ok=True)
    for page in sorted(f for f in os.listdir(src) if f.lower().endswith(".png")):
        path = os.path.join(src, page)
        figs, texts, _ = segment_page(path)
        im = Image.open(path).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in texts:
            d.rectangle(b, outline=(230, 0, 0), width=5)
        for b in figs:
            d.rectangle(b, outline=(0, 170, 0), width=8)
        im.save(os.path.join(out, page))
    print(f"Overlay debug -> {out}/  (xanh=figure, đỏ=text)")


def all_ids():
    ids = []
    with open(META, encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.split("\t")
            if len(c) >= 8 and int(c[7] or 0) > 0:
                ids.append(c[0])
    return ids


def main():
    if len(sys.argv) < 2:
        print("Dùng: segment.py <id> | --debug <id> | --all [limit]")
        return 1
    if sys.argv[1] == "--debug":
        save_debug(sys.argv[2])
    elif sys.argv[1] == "--all":
        ids = all_ids()
        if len(sys.argv) > 2:
            ids = ids[: int(sys.argv[2])]
        tf = tt = new = 0
        for i, pid in enumerate(ids, 1):
            nf, nt = save_crops(pid, skip_existing=True)
            if nf < 0:
                continue  # đã có
            tf += nf
            tt += nt
            new += 1
            print(f"  [{i}/{len(ids)}] {pid}: {nf} figure, {nt} text")
        print(f"\nXong: {new} patent mới, {tf} figure, {tt} text -> {OUT}")
    else:
        pid = sys.argv[1]
        nf, nt = save_crops(pid)
        print(f"{pid}: {nf} figure, {nt} text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
