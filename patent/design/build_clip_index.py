#!/usr/bin/env python3
# Nhúng CLIP cho mọi figure design đã tách -> data/patent_index/clip_index.json
# Kèm eval rank-1 để so với feature thủ công.
import json
import os
import sys

import numpy as np

import clip_embed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX = os.path.join(ROOT, "data", "patent_index")
FIGURES = os.path.join(ROOT, "data", "patent_figures", "figures")
OUT = os.path.join(INDEX, "clip_index.json")


def designs():
    out = {}
    with open(os.path.join(INDEX, "meta.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 11 and c[1] == "design":
                out[c[0]] = {"owner": c[8], "status": c[3], "title": c[10]}
    return out


def main():
    # tái dùng embedding cũ (chỉ nhúng figure mới)
    cache = {}
    if os.path.exists(OUT):
        for r in json.load(open(OUT, encoding="utf-8")):
            cache[(r["id"], r["fig"])] = r["emb"]

    d = designs()
    # liệt kê mọi figure design; tách cái đã cache vs cái mới
    recs = []
    todo = []  # (idx_in_recs, path)
    for pid, meta in d.items():
        fdir = os.path.join(FIGURES, pid)
        if not os.path.isdir(fdir):
            continue
        for fig in sorted(f for f in os.listdir(fdir) if f.endswith(".png")):
            rec = {"id": pid, "fig": fig, "emb": cache.get((pid, fig)), **meta}
            if rec["emb"] is None:
                todo.append((len(recs), os.path.join(fdir, fig)))
            recs.append(rec)

    # nhúng CLIP theo batch cho figure mới
    BATCH = 48
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        embs = clip_embed.embed_batch([p for _, p in chunk])
        for (idx, _), e in zip(chunk, embs):
            recs[idx]["emb"] = e.tolist()
        if (i // BATCH) % 20 == 0:
            print(f"  CLIP {min(i + BATCH, len(todo))}/{len(todo)} figure mới")
    print(f"  {len(todo)} figure mới nhúng CLIP (batch)")
    json.dump(recs, open(OUT, "w"))
    print(f"Xong: {OUT} — {len(recs)} figure / {len(set(r['id'] for r in recs))} patent")
    print("Đo độ chính xác:  python3 patent/design/eval_clip.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
