#!/usr/bin/env python3
# Eval ở MỨC PATENT (đúng use case: cho 1 view -> tìm đúng patent).
# So 2 cách đại diện patent: max-over-views vs mean-embedding.
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLIP = os.path.join(ROOT, "data", "patent_index", "clip_index.json")


def main():
    recs = json.load(open(CLIP, encoding="utf-8"))
    ids = np.array([r["id"] for r in recs])
    E = np.asarray([r["emb"] for r in recs], dtype=np.float32)  # (N,512) L2-norm
    N = len(recs)
    pats = sorted(set(ids.tolist()))
    pat_to_idx = {p: i for i, p in enumerate(pats)}
    ids_idx = np.array([pat_to_idx[r] for r in ids])
    idx_of = {p: np.where(ids == p)[0] for p in pats}
    mean_emb = {}
    for p in pats:
        m = E[idx_of[p]].mean(0)
        mean_emb[p] = m / (np.linalg.norm(m) + 1e-9)
    MEAN = np.array([mean_emb[p] for p in pats], dtype=np.float32)

    rng = np.random.default_rng(0)
    K = min(1500, N)
    qidx = rng.choice(N, size=K, replace=False)

    ok_max = ok_mean = ok_hybrid = ok_hybrid2 = ok_hybrid3 = 0
    
    # Đánh giá tuần tự từng truy vấn (OOM-safe và cực nhanh bằng np.maximum.at)
    for step, idx in enumerate(qidx, 1):
        true = ids[idx]
        true_idx = pat_to_idx[true]
        qi = E[idx]
        
        # 1) Max-over-views: tìm ảnh tương đồng nhất (loại chính nó)
        sim_all = E @ qi
        sim_all_ex = sim_all.copy()
        sim_all_ex[idx] = -1.0
        
        patent_max = np.full(len(pats), -1.0, dtype=np.float32)
        np.maximum.at(patent_max, ids_idx, sim_all_ex)
        
        pred_max_idx = int(patent_max.argmax())
        ok_max += (pred_max_idx == true_idx)
        
        # 2) Mean-embedding
        sim_mean = MEAN @ qi
        pred_mean_idx = int(sim_mean.argmax())
        ok_mean += (pred_mean_idx == true_idx)
        
        # Hybrid-pooling với nhiều cấu hình trọng số
        m_score = np.clip((sim_mean - 0.6) / 0.35, 0, 1)
        mx_score = np.clip((patent_max - 0.6) / 0.35, 0, 1)
        
        # Thử nghiệm cấu hình raw (không clip) để giữ nguyên độ phân giải của ranking
        raw_h_1 = 0.1 * sim_mean + 0.9 * patent_max
        raw_h_2 = 0.2 * sim_mean + 0.8 * patent_max
        raw_h_3 = 0.3 * sim_mean + 0.7 * patent_max
        raw_h_5 = 0.5 * sim_mean + 0.5 * patent_max
        
        # Thử nghiệm cấu hình clipped
        clip_h_1 = 0.1 * m_score + 0.9 * mx_score
        clip_h_2 = 0.2 * m_score + 0.8 * mx_score
        
        pred_rh1_idx = int(raw_h_1.argmax())
        pred_rh2_idx = int(raw_h_2.argmax())
        pred_rh3_idx = int(raw_h_3.argmax())
        pred_rh5_idx = int(raw_h_5.argmax())
        
        pred_ch1_idx = int(clip_h_1.argmax())
        pred_ch2_idx = int(clip_h_2.argmax())
        
        # Khai báo các biến đếm ở trên hàm main (sẽ được cập nhật tự động dưới đây)
        if step == 1:
            global ok_rh1, ok_rh2, ok_rh3, ok_rh5, ok_ch1, ok_ch2
            ok_rh1 = ok_rh2 = ok_rh3 = ok_rh5 = ok_ch1 = ok_ch2 = 0
            
        ok_rh1 += (pred_rh1_idx == true_idx)
        ok_rh2 += (pred_rh2_idx == true_idx)
        ok_rh3 += (pred_rh3_idx == true_idx)
        ok_rh5 += (pred_rh5_idx == true_idx)
        ok_ch1 += (pred_ch1_idx == true_idx)
        ok_ch2 += (pred_ch2_idx == true_idx)

    print(f"{N} figure / {len(pats)} patent (baseline ngẫu nhiên {100/len(pats):.4f}%)\n")
    print(f"Rank-1 patent-level — max-over-views (raw) : {ok_max/K*100:.1f}%")
    print(f"Rank-1 patent-level — mean-embedding (raw) : {ok_mean/K*100:.1f}%")
    print(f"Rank-1 patent-level — raw hybrid (0.1 Mean + 0.9 Max) : {ok_rh1/K*100:.1f}%")
    print(f"Rank-1 patent-level — raw hybrid (0.2 Mean + 0.8 Max) : {ok_rh2/K*100:.1f}%")
    print(f"Rank-1 patent-level — raw hybrid (0.3 Mean + 0.7 Max) : {ok_rh3/K*100:.1f}%")
    print(f"Rank-1 patent-level — raw hybrid (0.5 Mean + 0.5 Max) : {ok_rh5/K*100:.1f}%")
    print(f"Rank-1 patent-level — clip hybrid (0.1 Mean + 0.9 Max) : {ok_ch1/K*100:.1f}%")
    print(f"Rank-1 patent-level — clip hybrid (0.2 Mean + 0.8 Max) : {ok_ch2/K*100:.1f}%")
    
    # Đánh giá so khớp chi tiết cục bộ (Part-Based Reranking)
    eval_local_rerank(recs, qidx, pats, pat_to_idx)
    return 0


def eval_local_rerank(recs, qidx, pats, pat_to_idx):
    import match_core
    # Lấy mẫu 100 query để đánh giá (vì nhúng patch on-the-fly tốn CPU)
    K_patch = min(100, len(qidx))
    print(f"\n--- Đánh giá Part-Based Reranking (Mẫu {K_patch} query) ---")
    
    ok_coarse = 0
    ok_refined = 0
    
    for idx_run, idx in enumerate(qidx[:K_patch], 1):
        true_pat = recs[idx]["id"]
        
        # Đường dẫn ảnh truy vấn
        q_path = os.path.join(ROOT, "data", "patent_figures", "figures", true_pat, recs[idx]["fig"])
        
        # 1) Search không có local rerank (chỉ coarse)
        res_coarse = match_core.rank(image=q_path, text="", topk=1, use_local_rerank=False)
        pred_coarse = res_coarse[0]["id"] if len(res_coarse) else None
        ok_coarse += (pred_coarse == true_pat)
        
        # 2) Search có local rerank (coarse + fine-grained)
        res_refined = match_core.rank(image=q_path, text="", topk=1, use_local_rerank=True)
        pred_refined = res_refined[0]["id"] if len(res_refined) else None
        ok_refined += (pred_refined == true_pat)
        
        if idx_run % 20 == 0:
            print(f"  Đã chạy {idx_run}/{K_patch} query...")
            
    print(f"\nRank-1 patent-level — Chỉ dùng Coarse Search (30% Mean + 70% Max) : {ok_coarse/K_patch*100:.1f}%")
    print(f"Rank-1 patent-level — Coarse + Part-Based Reranking (Tối ưu) : {ok_refined/K_patch*100:.1f}%")


if __name__ == "__main__":
    main()
