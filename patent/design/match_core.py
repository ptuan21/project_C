#!/usr/bin/env python3
# Lõi matching design (CLIP) — có CACHE .npy để truy vấn NHANH trên 50k+ figure.
#   - đại diện patent = MEAN embedding các view.
#   - ảnh & text trong cùng không gian CLIP -> fuse cross-modal.
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

import clip_embed
import clip_text


def get_patches(image_input):
    if isinstance(image_input, str):
        im = Image.open(image_input).convert("RGB")
    else:
        im = image_input.convert("RGB")
    w, h = im.size
    w_half, h_half = w // 2, h // 2
    
    # 5 patches: Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center
    patches = [
        im.crop((0, 0, w_half, h_half)),
        im.crop((w_half, 0, w, h_half)),
        im.crop((0, h_half, w_half, h)),
        im.crop((w_half, h_half, w, h)),
        im.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    ]
    return patches

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX = os.path.join(ROOT, "data", "patent_index")
CLIP_JSON = os.path.join(IDX, "clip_index.json")
EMB = os.path.join(IDX, "clip_emb.npy")
ROWS = os.path.join(IDX, "clip_rows.json")
PMEAN = os.path.join(IDX, "clip_pmeans.npy")
PIDS = os.path.join(IDX, "clip_pids.json")
PROWS = os.path.join(IDX, "clip_patrows.json")


def _ci(c):  # cosine ảnh-ảnh (~0.6..0.95) -> [0,1]
    return max(0.0, min(1.0, (c - 0.6) / 0.35))


def _ct(c):  # cosine text-ảnh (~0.18..0.33) -> [0,1]
    return max(0.0, min(1.0, (c - 0.18) / 0.15))


def _build_cache():
    """Chuyển clip_index.json -> .npy + patent means (chạy 1 lần, tự làm lại khi index đổi)."""
    recs = json.load(open(CLIP_JSON, encoding="utf-8"))
    E = np.asarray([r["emb"] for r in recs], dtype=np.float32)
    np.save(EMB, E)
    json.dump([{k: r[k] for k in ("id", "fig", "owner", "status", "title")} for r in recs],
              open(ROWS, "w", encoding="utf-8"))
    pr = defaultdict(list)
    for i, r in enumerate(recs):
        pr[r["id"]].append(i)
    pids = list(pr.keys())
    M = np.zeros((len(pids), E.shape[1]), dtype=np.float32)
    for k, p in enumerate(pids):
        m = E[pr[p]].mean(0)
        M[k] = m / (np.linalg.norm(m) + 1e-9)
    np.save(PMEAN, M)
    json.dump(pids, open(PIDS, "w"))
    json.dump({p: pr[p] for p in pids}, open(PROWS, "w"))


def _ensure():
    if not os.path.exists(CLIP_JSON):
        raise FileNotFoundError("Chưa có clip_index.json — chạy build_clip_index.py.")
    if (not os.path.exists(EMB)) or os.path.getmtime(EMB) < os.path.getmtime(CLIP_JSON):
        _build_cache()


_cache = None
_title_index = None


def _load():
    global _cache, _title_index
    if _cache is None:
        _ensure()
        E = np.load(EMB)
        rows = json.load(open(ROWS, encoding="utf-8"))
        M = np.load(PMEAN)
        pids = json.load(open(PIDS, encoding="utf-8"))
        pr = json.load(open(PROWS, encoding="utf-8"))
        _cache = (E, rows, M, pids, pr)
    if _title_index is None:
        import textmatch
        _title_index = textmatch.TitleIndex()
    return _cache


def rank(image=None, text="", topk=6, w_img=0.6, w_txt=0.4, use_local_rerank=True):
    E, rows, M, pids, pr = _load()
    qi = clip_embed.embed(image) if image else None
    qt = clip_text.embed(text) if text.strip() else None
    if qi is None and qt is None:
        raise ValueError("Cần ít nhất ảnh hoặc text.")

    score_img = None
    img_p = None
    img_max = None
    if qi is not None:
        img_p = M @ qi
        img_all = E @ qi
        # Max-pooling: tìm ảnh khớp nhất trong mỗi patent
        img_max = np.array([img_all[pr[pid]].max() for pid in pids])
        # Raw hybrid pooling (30% Mean + 70% Max) để giữ nguyên độ phân giải rank tốt nhất
        img_raw = 0.3 * img_p + 0.7 * img_max
        score_img = np.clip((img_raw - 0.6) / 0.35, 0, 1)

    score_txt = None
    if qt is not None:
        txt_p = M @ qt
        # CLIP text similarity
        clip_txt_score = np.clip((txt_p - 0.18) / 0.15, 0, 1)
        # BM25 title similarity
        bm25_scores = _title_index.scores(text)
        bm25_arr = np.array([bm25_scores.get(pid, 0.0) for pid in pids])
        # Gộp lại (50% CLIP + 50% BM25)
        score_txt = 0.5 * clip_txt_score + 0.5 * bm25_arr

    # Điểm thô ban đầu (Coarse search score)
    if score_img is not None and score_txt is not None:
        coarse_score = w_img * score_img + w_txt * score_txt
    elif score_img is not None:
        coarse_score = score_img
    else:
        coarse_score = score_txt

    # Khởi tạo mảng điểm ảnh cuối cùng (mặc định là score_img hoặc zero)
    final_img_score = score_img.copy() if score_img is not None else None

    # Tái xếp hạng cục bộ (Part-Based Reranking) khi có ảnh truy vấn
    score = coarse_score
    if qi is not None and image is not None and use_local_rerank:
        C = max(30, topk)
        top_coarse = np.argsort(-coarse_score)[:C]
        
        try:
            # 1) Tiền xử lý toàn cục ảnh Query và cắt patch
            q_im = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
            q_im_pre = clip_embed.preprocess_whole_image(q_im)
            q_patches = get_patches(q_im_pre)
            q_patches_emb = clip_embed.embed_patches(q_patches)  # (5, 512)
            
            # Tính toán Salience weights (mật độ nét vẽ) cho các patch của Query
            q_weights = []
            for p in q_patches:
                p_gray = np.asarray(p.convert("L"))
                ink_pixels = np.sum(p_gray < 220)
                q_weights.append(float(ink_pixels))
            q_weights = np.array(q_weights, dtype=np.float32)
            if q_weights.sum() > 0:
                q_weights = q_weights / q_weights.sum()
            else:
                q_weights = np.full(5, 0.2, dtype=np.float32)
            
            # 2) Chuẩn bị đường dẫn và tiền xử lý toàn cục + cắt patch cho các ứng viên
            cand_paths = []
            q = qi if qi is not None else qt
            for k in top_coarse:
                pid = pids[k]
                idxs = pr[pid]
                best = idxs[int(np.argmax(E[idxs] @ q))]
                r = rows[best]
                fig_path = os.path.join(ROOT, "data", "patent_figures", "figures", pid, r["fig"])
                cand_paths.append(fig_path)
                
            all_cand_patches = []
            for path in cand_paths:
                try:
                    cand_im = Image.open(path).convert("RGB")
                    cand_im_pre = clip_embed.preprocess_whole_image(cand_im)
                    all_cand_patches.extend(get_patches(cand_im_pre))
                except Exception:
                    all_cand_patches.extend([Image.new("RGB", (224, 224), (255, 255, 255))] * 5)
                    
            # 3) Nhúng hàng loạt các patch của các ứng viên (rất nhanh nhờ batching)
            c_patches_emb = clip_embed.embed_patches(all_cand_patches)  # (C * 5, 512)
            
            # 4) Tính toán độ tương đồng chi tiết cục bộ (Local Similarity)
            refined_scores = coarse_score.copy()
            for idx_in_top, k in enumerate(top_coarse):
                emb_i = c_patches_emb[idx_in_top * 5 : (idx_in_top + 1) * 5]  # (5, 512)
                S = q_patches_emb @ emb_i.T                                 # (5, 5)
                
                # So khớp 2 chiều (Bi-directional alignment matching):
                # Chiều 1: Query -> Candidate (có trọng số Salience)
                max_q_to_c = S.max(axis=1)
                sim_q_to_c = np.sum(max_q_to_c * q_weights)
                
                # Chiều 2: Candidate -> Query (trọng số đều)
                max_c_to_q = S.max(axis=0)
                sim_c_to_q = max_c_to_q.mean()
                
                local_sim = float(0.5 * sim_q_to_c + 0.5 * sim_c_to_q)
                score_img_local = max(0.0, min(1.0, (local_sim - 0.6) / 0.35))
                
                # Gộp điểm toàn cục và điểm cục bộ cho điểm ảnh
                img_coarse = score_img[k]
                img_refined = 0.6 * img_coarse + 0.4 * score_img_local
                final_img_score[k] = img_refined
                
                # Cập nhật điểm tổng hợp cuối cùng
                if score_txt is not None:
                    refined_scores[k] = w_img * img_refined + w_txt * score_txt[k]
                else:
                    refined_scores[k] = img_refined
            score = refined_scores
        except Exception:
            # Fallback nếu có lỗi xử lý ảnh
            pass

    top = np.argsort(-score)[:topk]
    q = qi if qi is not None else qt
    out = []
    for k in top:
        pid = pids[k]
        idxs = pr[pid]
        best = idxs[int(np.argmax(E[idxs] @ q))]   # view khớp nhất để hiển thị
        r = rows[best]
        out.append({"id": pid, "fig": r["fig"], "owner": r["owner"], "status": r["status"],
                    "title": r["title"], "score": float(score[k]),
                    "img": float(final_img_score[k]) if final_img_score is not None else 0.0,
                    "txt": float(_ct(M[k] @ qt)) if qt is not None else 0.0})
    return out

