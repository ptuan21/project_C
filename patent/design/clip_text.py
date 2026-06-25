#!/usr/bin/env python3
# patent/design — nhúng TEXT bằng CLIP ViT-B/32 (cùng không gian với ảnh) + BPE tokenizer.
import gzip
import html
import os
import re

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "data", "models", "clip_text.onnx")
BPE = os.path.join(ROOT, "data", "models", "bpe_simple_vocab_16e6.txt.gz")
CTX = 77


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word):
    return set(zip(word[:-1], word[1:]))


class SimpleTokenizer:
    def __init__(self):
        self.byte_encoder = bytes_to_unicode()
        merges = gzip.open(BPE).read().decode("utf-8").split("\n")
        merges = [tuple(m.split()) for m in merges[1:49152 - 256 - 2 + 1]]
        vocab = list(self.byte_encoder.values())
        vocab += [v + "</w>" for v in vocab]
        vocab += ["".join(m) for m in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {t: i for i, t in enumerate(vocab)}
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.cache = {}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+""",
            re.I)
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, 1e9))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new.extend(word[i:j])
                    i = j
                except ValueError:
                    new.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text):
        text = re.sub(r"\s+", " ", html.unescape(text).strip().lower())
        ids = []
        for tok in self.pat.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[bt] for bt in self.bpe(tok).split(" "))
        return ids


_tok = None
_sess = None


def _setup():
    global _tok, _sess
    if _tok is None:
        _tok = SimpleTokenizer()
        _sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    return _tok, _sess


def embed(text):
    tok, sess = _setup()
    ids = [tok.sot] + tok.encode(text)[:CTX - 2] + [tok.eot]
    mask = [1] * len(ids)
    while len(ids) < CTX:
        ids.append(0)
        mask.append(0)
    out = sess.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                          "attention_mask": np.array([mask], dtype=np.int64)})[0][0]
    n = np.linalg.norm(out)
    return (out / n if n > 0 else out).astype(np.float32)
