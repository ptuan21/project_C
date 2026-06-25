#!/usr/bin/env python3
# patent/design — tiện ích ảnh: tiền xử lý + descriptor (pHash + Hu moments) + độ tương đồng.
# Offline, chỉ dùng Pillow + numpy (không model/API).
import numpy as np
from PIL import Image


def preprocess(path, size=256):
    """Đọc ảnh -> grayscale -> crop về bounding box phần 'mực' (nét) -> resize size x size."""
    im = Image.open(path).convert("L")
    arr = np.asarray(im)
    ink = arr < 128  # pixel tối = nét vẽ
    ys, xs = np.where(ink)
    if len(xs) > 0:
        im = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    im = im.resize((size, size))
    return np.asarray(im).astype(np.float64)


def _dct_matrix(n):
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * x + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] /= np.sqrt(2.0)
    return m


_D32 = _dct_matrix(32)


def phash(gray, hash_size=8):
    """pHash dựa trên DCT: trả về số nguyên 64-bit."""
    small = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((32, 32))).astype(np.float64)
    d = _D32 @ small @ _D32.T
    low = d[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])  # bỏ hệ số DC
    bits = low > med
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hu_moments(gray):
    """7 bất biến Hu trên silhouette nhị phân (đã lấy log để cùng thang)."""
    img = (gray < 128).astype(np.float64)
    m00 = img.sum()
    if m00 == 0:
        return np.zeros(7)
    y, x = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    xb = (x * img).sum() / m00
    yb = (y * img).sum() / m00

    def mu(p, q):
        return (((x - xb) ** p) * ((y - yb) ** q) * img).sum()

    def eta(p, q):
        return mu(p, q) / (m00 ** (1 + (p + q) / 2.0))

    e20, e02, e11 = eta(2, 0), eta(0, 2), eta(1, 1)
    e30, e12, e21, e03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    h = np.zeros(7)
    h[0] = e20 + e02
    h[1] = (e20 - e02) ** 2 + 4 * e11 ** 2
    h[2] = (e30 - 3 * e12) ** 2 + (3 * e21 - e03) ** 2
    h[3] = (e30 + e12) ** 2 + (e21 + e03) ** 2
    h[4] = (e30 - 3 * e12) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2) + (
        3 * e21 - e03
    ) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2)
    h[5] = (e20 - e02) * ((e30 + e12) ** 2 - (e21 + e03) ** 2) + 4 * e11 * (e30 + e12) * (e21 + e03)
    h[6] = (3 * e21 - e03) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2) - (
        e30 - 3 * e12
    ) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2)
    # log transform giữ dấu
    return -np.sign(h) * np.log10(np.abs(h) + 1e-30)


def hog(gray, cell=16, bins=9, block=2, eps=1e-6):
    """HOG chuẩn (Dalal–Triggs) có chuẩn hóa theo khối (L2-Hys) -> phân biệt mạnh.
    Trả về vector L2-norm (cosine = tích vô hướng)."""
    g = gray.astype(np.float64)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx) % np.pi
    H, W = g.shape
    ncy, ncx = H // cell, W // cell
    bidx = np.minimum((ang / np.pi * bins).astype(int), bins - 1)

    cells = np.zeros((ncy, ncx, bins))
    for cy in range(ncy):
        for cx in range(ncx):
            m = mag[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell]
            b = bidx[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell]
            cells[cy, cx] = np.bincount(b.ravel(), weights=m.ravel(), minlength=bins)

    feat = []
    for by in range(ncy - block + 1):
        for bx in range(ncx - block + 1):
            blk = cells[by:by + block, bx:bx + block].ravel()
            v = blk / np.sqrt((blk ** 2).sum() + eps ** 2)  # L2
            v = np.minimum(v, 0.2)                            # clip (Hys)
            v = v / np.sqrt((v ** 2).sum() + eps ** 2)        # L2 lại
            feat.append(v)
    feat = np.concatenate(feat) if feat else np.zeros(1)
    n = np.linalg.norm(feat)
    return feat / n if n > 0 else feat


def phash_sim(a, b):
    """1 - hamming/64, trong [0,1]."""
    return 1.0 - bin(a ^ b).count("1") / 64.0


def hu_sim(a, b):
    return float(np.exp(-np.linalg.norm(np.asarray(a) - np.asarray(b)) / 5.0))


def image_similarity(a, b):
    """Gộp HOG (hướng nét — trụ cột) + pHash (cấu trúc). Bỏ Hu vì gần như vô dụng."""
    hg = float(np.dot(a["hog"], b["hog"]))  # 2 vector đã chuẩn L2 -> cosine
    return 0.85 * hg + 0.15 * phash_sim(a["phash"], b["phash"])


def describe(path):
    g = preprocess(path)
    return {"phash": phash(g), "hu": hu_moments(g).tolist(), "hog": hog(g).tolist()}
