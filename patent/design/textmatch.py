#!/usr/bin/env python3
# patent/design — so text sản phẩm với TIÊU ĐỀ design patent (BM25), để fuse với điểm ảnh.
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = os.path.join(ROOT, "data", "patent_index", "meta.tsv")

STOP = {"the", "a", "an", "of", "for", "and", "or", "with", "to", "in", "device", "apparatus"}


def toks(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 2 and w not in STOP]


class TitleIndex:
    """BM25 trên tiêu đề các design patent. Trả về điểm chuẩn hóa [0,1] mỗi patent."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = {}  # id -> tokens
        with open(META, encoding="utf-8") as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) >= 11 and c[1] == "design":
                    self.docs[c[0]] = toks(c[10])
        self.N = len(self.docs)
        self.avgdl = (sum(len(t) for t in self.docs.values()) / self.N) if self.N else 0
        self.df = {}
        for t in self.docs.values():
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for w, df in self.df.items()}

    def scores(self, text):
        q = set(toks(text))
        raw = {}
        for pid, toklist in self.docs.items():
            tf = {}
            for w in toklist:
                tf[w] = tf.get(w, 0) + 1
            dl = len(toklist)
            s = 0.0
            for w in q:
                if w in tf:
                    s += self.idf[w] * tf[w] * (self.k1 + 1) / (
                        tf[w] + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl)))
            raw[pid] = s
        mx = max(raw.values()) if raw else 0.0
        return {pid: (s / mx if mx > 0 else 0.0) for pid, s in raw.items()}  # chuẩn hóa [0,1]
