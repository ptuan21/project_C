#!/usr/bin/env python3
# Eval rank-1 cho CLIP index — MEMORY-SAFE (lấy mẫu query + chunk), chạy được trên 50k+ figure.
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLIP = os.path.join(ROOT, "data", "patent_index", "clip_index.json")


def main():
    recs = json.load(open(CLIP, encoding="utf-8"))
    ids = np.array([r["id"] for r in recs])
    E = np.asarray([r["emb"] for r in recs], dtype=np.float32)  # (N,512) đã L2-norm
    N = len(recs)
    npat = len(set(ids.tolist()))
    print(f"{N} figure / {npat} patent  (baseline ngẫu nhiên {100/npat:.3f}%)")

    rng = np.random.default_rng(0)
    K = min(3000, N)
    qidx = rng.choice(N, size=K, replace=False)

    ok = 0
    same_sum = same_n = diff_sum = diff_n = 0.0
    for s in range(0, K, 200):                       # chunk query để tiết kiệm RAM
        qi = qidx[s:s + 200]
        S = E[qi] @ E.T                              # (chunk, N)
        for r, i in enumerate(qi):
            row = S[r]
            row[i] = -1                              # loại chính nó
            j = int(row.argmax())
            ok += (ids[j] == ids[i])
            same = ids == ids[i]
            same[i] = False
            same_sum += row[same].sum(); same_n += same.sum()
            diff_sum += row[~same].sum(); diff_n += (~same).sum() - 1  # trừ phần tử -1 đã set? xấp xỉ
    print(f"\nCLIP rank-1 (figure cùng patent gần nhất): {ok/K*100:.1f}%  (mẫu {K} query)")
    print(f"Cosine TB  same-patent={same_sum/max(1,same_n):.3f}  diff-patent={diff_sum/max(1,diff_n):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
