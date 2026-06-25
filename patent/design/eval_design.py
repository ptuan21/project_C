#!/usr/bin/env python3
# Đánh giá độ chính xác matching design: mỗi figure tìm figure gần nhất (khác chính nó),
# kiểm tra có CÙNG patent không (rank-1). Đo cả khả năng tách same/diff.
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX = os.path.join(ROOT, "data", "patent_index", "design_index.json")


def phash_sim_matrix(ph):
    N = len(ph)
    M = np.zeros((N, N))
    for i in range(N):
        a = ph[i]
        for j in range(i + 1, N):
            d = bin(a ^ ph[j]).count("1")
            M[i, j] = M[j, i] = 1 - d / 64.0
    return M


def main():
    recs = json.load(open(INDEX, encoding="utf-8"))
    ids = np.array([r["id"] for r in recs])
    N = len(recs)
    hog = np.array([r["hog"] for r in recs])          # (N,576) đã L2-norm
    hu = np.array([r["hu"] for r in recs])
    ph = [int(r["phash"]) for r in recs]

    HOG = hog @ hog.T                                  # cosine
    from scipy.spatial.distance import cdist
    HU = np.exp(-cdist(hu, hu) / 5.0)
    PH = phash_sim_matrix(ph)

    def rank1(S):
        S = S.copy()
        np.fill_diagonal(S, -1)
        j = S.argmax(axis=1)
        return float(np.mean(ids[j] == ids))

    combos = {
        "pHash": PH, "Hu": HU, "HOG": HOG,
        "pHash+Hu": 0.6 * PH + 0.4 * HU,
        "gộp (0.25/0.15/0.6)": 0.25 * PH + 0.15 * HU + 0.6 * HOG,
        "HOG mạnh (0.1/0.1/0.8)": 0.1 * PH + 0.1 * HU + 0.8 * HOG,
    }
    print(f"Số figure: {N} | số patent: {len(set(ids.tolist()))}\n")
    print("Rank-1 (hàng xóm gần nhất CÙNG patent):")
    for name, S in combos.items():
        print(f"  {name:24s}: {rank1(S)*100:5.1f}%")

    # tách same vs diff (dùng descriptor gộp)
    S = combos["gộp (0.25/0.15/0.6)"]
    same, diff = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (same if ids[i] == ids[j] else diff).append(S[i, j])
    print(f"\nĐiểm trung bình  same-patent={np.mean(same):.3f}  diff-patent={np.mean(diff):.3f}"
          f"  (cách biệt {np.mean(same)-np.mean(diff):.3f})")


if __name__ == "__main__":
    main()
