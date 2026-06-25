#!/usr/bin/env python3
# Test descriptor ảnh (offline, tổng hợp — không cần dữ liệu thật).
import sys

import numpy as np

import imutil

P, F = 0, 0


def check(cond, msg):
    global P, F
    if cond:
        P += 1
    else:
        F += 1
        print("  [FAIL]", msg)


def main():
    # ảnh tổng hợp: hình vuông đặc ở giữa (giá trị thấp = "mực")
    a = np.full((256, 256), 255.0)
    a[80:176, 80:176] = 0.0
    # bản sao y hệt
    b = a.copy()
    # hình tròn (khác hẳn)
    yy, xx = np.mgrid[0:256, 0:256]
    circ = np.full((256, 256), 255.0)
    circ[(xx - 128) ** 2 + (yy - 128) ** 2 < 60 ** 2] = 0.0

    da = {"phash": imutil.phash(a), "hu": imutil.hu_moments(a), "hog": imutil.hog(a)}
    db = {"phash": imutil.phash(b), "hu": imutil.hu_moments(b), "hog": imutil.hog(b)}
    dc = {"phash": imutil.phash(circ), "hu": imutil.hu_moments(circ), "hog": imutil.hog(circ)}

    # pHash: y hệt -> tương đồng 1.0
    check(imutil.phash_sim(da["phash"], db["phash"]) == 1.0, "pHash ảnh giống hệt phải = 1.0")
    check(imutil.phash_sim(da["phash"], dc["phash"]) < 1.0, "pHash vuông vs tròn phải < 1.0")

    # Hu: giống hệt -> ~0; khác hình -> khác
    check(np.linalg.norm(da["hu"] - db["hu"]) < 1e-6, "Hu giống hệt phải ~0")
    check(np.linalg.norm(da["hu"] - dc["hu"]) > 1e-3, "Hu vuông vs tròn phải khác")

    # HOG: cosine giống hệt = 1; vuông vs tròn < 1 (phân biệt)
    check(abs(float(np.dot(da["hog"], db["hog"])) - 1.0) < 1e-9, "HOG cosine giống hệt = 1")
    check(float(np.dot(da["hog"], dc["hog"])) < 0.99, "HOG vuông vs tròn phải < 1")

    # similarity gộp: giống hệt = 1.0 và cao hơn khác hình
    sim_same = imutil.image_similarity(da, db)
    sim_diff = imutil.image_similarity(da, dc)
    check(sim_same > sim_diff, "image_similarity: giống hệt > khác hình")
    check(abs(sim_same - 1.0) < 1e-9, "image_similarity giống hệt phải = 1.0")

    # Hu bất biến tịnh tiến: dịch hình vuông -> Hu gần như không đổi
    a2 = np.full((256, 256), 255.0)
    a2[100:196, 100:196] = 0.0  # dịch +20,+20
    check(np.linalg.norm(imutil.hu_moments(a) - imutil.hu_moments(a2)) < 0.2,
          "Hu phải bất biến tịnh tiến")

    print(f"\n=== {P} passed, {F} failed ===")
    return 0 if F == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
