#!/usr/bin/env python3
# patent/design — nhúng ảnh bằng CLIP ViT-B/32 (ONNX, chạy CPU/CoreML offline).
import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, label, find_objects

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "data", "models", "clip_vision.onnx")

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
_sess = None


def session():
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    return _sess


def remove_text_noise(im, max_size=15):
    """Xóa các nhãn số chỉ dẫn nhỏ (ví dụ '10', '12') trên bản vẽ kỹ thuật."""
    gray = np.asarray(im.convert("L")).copy()
    binary = gray < 128  # nét vẽ là True
    labeled, num_features = label(binary)
    slices = find_objects(labeled)
    for i, sl in enumerate(slices, 1):
        if sl is None:
            continue
        ys, xs = sl
        w = xs.stop - xs.start
        h = ys.stop - ys.start
        if w <= max_size and h <= max_size:
            patch = labeled[sl]
            gray[sl][patch == i] = 255
    return Image.fromarray(gray).convert("RGB")


def _to_edges(im):
    """Ảnh chụp thật -> edge map chất lượng cao sử dụng bộ lọc Gaussian và Sobel."""
    a = np.asarray(im.convert("L")).astype(np.float64)
    # Lọc nhiễu Gaussian
    blurred = gaussian_filter(a, sigma=1.0)
    # Tính đạo hàm gradient theo Sobel
    dx = sobel(blurred, axis=1)
    dy = sobel(blurred, axis=0)
    mag = np.hypot(dx, dy)
    mag_max = mag.max()
    if mag_max > 0:
        mag = mag / mag_max
    # Nét vẽ đen (0), nền trắng (255)
    edge = 255.0 - np.clip(mag * 255.0 * 3.0, 0, 255)
    return Image.fromarray(edge.astype("uint8")).convert("RGB")


def _is_photo(im):
    """Đoán ảnh chụp (nhiều mức xám trung gian) vs line-art (gần nhị phân)."""
    a = np.asarray(im.convert("L"))
    mid = np.mean((a > 40) & (a < 215))
    return mid > 0.15


def preprocess_whole_image(im, edge="auto"):
    """Tiền xử lý toàn bộ bức ảnh (phát hiện cạnh hoặc lọc nhiễu chữ số) một lần duy nhất."""
    is_ph = _is_photo(im)
    use = (edge is True) or (edge == "auto" and is_ph)
    if use:
        im = _to_edges(im)
    else:
        im = remove_text_noise(im)
    return im


def preprocess_patch(patch):
    """Tiền xử lý nhanh cho từng phân mảnh (patch) đã được lọc nhiễu trước đó."""
    patch = patch.resize((224, 224), Image.BICUBIC)
    a = np.asarray(patch).astype(np.float32) / 255.0
    a = a.transpose(2, 0, 1)
    a = (a - _MEAN) / _STD
    return a[None]


def preprocess_image(im, edge="auto"):
    im = preprocess_whole_image(im, edge)
    return preprocess_patch(im)


def preprocess(path, edge="auto"):
    im = Image.open(path).convert("RGB")
    return preprocess_image(im, edge)



def embed(path, edge="auto"):
    out = session().run(None, {"pixel_values": preprocess(path, edge)})[0][0]
    n = np.linalg.norm(out)
    return (out / n if n > 0 else out).astype(np.float32)


def embed_batch(paths, edge="auto"):
    """Nhúng nhiều ảnh 1 lần (nhanh hơn nhiều). Trả về mảng (N,512) đã L2-norm."""
    arr = np.concatenate([preprocess(p, edge) for p in paths], axis=0)
    out = session().run(None, {"pixel_values": arr})[0]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(n > 0, n, 1)).astype(np.float32)


def embed_images(images, edge="auto"):
    """Nhúng hàng loạt đối tượng PIL Image trong cùng một batch."""
    arr = np.concatenate([preprocess_image(img, edge) for img in images], axis=0)
    out = session().run(None, {"pixel_values": arr})[0]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(n > 0, n, 1)).astype(np.float32)


def embed_patches(patches):
    """Nhúng hàng loạt patch (đã được tiền xử lý toàn cục trước đó) trong cùng một batch."""
    arr = np.concatenate([preprocess_patch(p) for p in patches], axis=0)
    out = session().run(None, {"pixel_values": arr})[0]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(n > 0, n, 1)).astype(np.float32)
