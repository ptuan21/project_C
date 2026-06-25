# Patent — hệ thống phân tích rủi ro vi phạm patent (US)

Mục tiêu: với 1 sản phẩm (của mình hoặc đối thủ), trả lời:
**ở US có patent nào liên quan không, nó bảo hộ gì, mình có nguy cơ vi phạm không?**

Dùng cho: (1) check rủi ro trước khi launch/source hàng Amazon, (2) tra cứu & phân tích patent.

> ⚠️ Đây là công cụ **sàng lọc**, không phải tư vấn pháp lý. Kết luận FTO cần luật sư.
> Dữ liệu mẫu chỉ ~1510 patent (prototype), không phải toàn bộ patent US.

## Cấu trúc
```
patent/
  index/patent_index.py   # P0 — ETL offline: USPTO json -> meta.tsv + docs.tsv + claims/<id>.txt
  include/patent/         # engine C++ (BM25, risk) — sẽ thêm
  src/
  cli/patent.cpp          # CLI — sẽ thêm
data/
  uspto_out/              # dữ liệu thô (5.4GB, không commit)
  patent_index/           # index sinh ra (không commit)
```

## Dữ liệu (data/uspto_out)
1510 patent USPTO (có metadata + claims text + bản vẽ):
- **utility**: 530 đã cấp (B1/B2) + 939 đơn công bố (A1) → phân tích bằng **claims (text)**
- **design**: 41 (kindCode S) → phân tích bằng **ảnh bản vẽ**

## Kế hoạch (MVP: thuần C++/offline, utility, input = keyword/mô tả)
- **P0 — Index** ✅ : parse json → claims sạch + metadata + tính hiệu lực (utility: nộp+20 năm; design: cấp+15 năm; A1 = đơn chưa cấp).
- **P1 — Retrieval** ✅ : mô tả sản phẩm → tokenize → **BM25** trên claims/abstract/title → top-K patent liên quan.
- **P2 — Chấm rủi ro** ✅ : tách claim độc lập thành đặc điểm → so phủ từ khóa → bảng đặc điểm + mức rủi ro (all-elements rule) + dẫn chứng.
- **screen** ✅ : gộp P1+P2 thành 1 báo cáo xếp nhóm rủi ro (CAO/TRUNG BÌNH/THẤP).
- **P3 (sau)**: design — nhận diện ảnh tương đồng (pHash + Hu moments trên silhouette; CNN embedding để nâng chất lượng); claim chart LLM cho luật sư.

## Chạy
```bash
python3 patent/index/patent_index.py    # P0: tạo index (cần data/uspto_out)
make patent && make patent-test          # build CLI + chạy test (19/19)

./build/patent screen "<mô tả sản phẩm>" [topk]   # tìm + chấm rủi ro (dùng chính)
./build/patent search "<mô tả sản phẩm>" [topk]   # chỉ tìm patent liên quan
./build/patent show <id>                          # metadata + claims đầy đủ
./build/patent risk <id> "<mô tả sản phẩm>"       # chấm rủi ro chi tiết từng đặc điểm
```

## Design (nhánh ảnh) — offline, Python (Pillow+numpy+scipy)
Nhận diện thiết kế tương đồng với design patent bằng **ảnh + text**:
```bash
python3 patent/design/segment.py <id>          # tách FIGURE riêng / TEXT riêng (projection)
python3 patent/design/segment.py --debug <id>  # overlay kiểm (xanh=figure, đỏ=text)
python3 patent/design/build_design_index.py    # descriptor pHash+Hu+HOG cho 277 figure
python3 patent/design/match_design.py <ảnh> 5 ["mô tả sản phẩm"]   # so + fuse text
python3 patent/design/demo_report.py <ảnh> 6 ["mô tả"] && open patent_demo.html  # demo HTML
python3 patent/design/test_design.py           # test descriptor (9/9)
```
- **Tách figure**: chiếu theo dải ngang → figure lấy trọn (không cắt), tách header/caption.
- **Descriptor**: pHash (cấu trúc) + Hu moments (hình, bất biến) + **HOG** (hướng nét — phân biệt mạnh).
- **Fuse ảnh + text**: BM25 trên tiêu đề design patent → điểm = 0.6·ảnh + 0.4·text.

### Quy mô hiện tại (sau khi scrape Google Patents)
- **5,171 design patent · 51,262 figure** (từ 41 patent ban đầu — ~125×).
- **CLIP rank-1: 61%** trên 5,171 patent (baseline ngẫu nhiên 0.019% → ~3200× tốt hơn).
- Truy vấn **<1s** nhờ cache `.npy` (`clip_emb.npy` + `clip_pmeans.npy`, tự build lại khi index đổi).
- Đo: `python3 patent/design/eval_clip.py` (memory-safe, lấy mẫu — chạy được trên 50k+ figure).

### CLIP embedding (độ chính xác cao) — đã tích hợp
Matcher chính giờ dùng **CLIP ViT-B/32 (ONNX, offline)** thay feature thủ công:
```bash
python3 -m pip install onnxruntime
# tải model 1 lần (~335MB):
curl -fL https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/main/model.onnx -o data/models/clip_vision.onnx
python3 patent/design/build_clip_index.py    # nhúng + tự eval rank-1
python3 patent/design/match_design.py <ảnh> 5 ["mô tả"]   # CLIP + BM25
```
**Kết quả đo (rank-1, hàng xóm gần nhất cùng patent):**
| Phương pháp | Rank-1 |
|---|---|
| Hu moments | 14.8% |
| pHash | 34.3% |
| HOG (đã tối ưu) | ~52% |
| Gộp thủ công | ~54% |
| **CLIP ViT-B/32** | **67.9%** |

CLIP còn xử lý được khoảng cách **ảnh chụp thật ↔ bản vẽ** mà feature thủ công không kham nổi.

**Nâng cấp thêm (đã tích hợp):**
- **Gộp theo patent (mean-embedding)** các view → rank-1 **69.7%** (so với 67.9% max-over-views).
- **CLIP text encoder** (model text 242M + BPE tokenizer tự cài): nhúng *mô tả/keyword* vào **cùng không gian CLIP** với ảnh → tìm bằng **text→ảnh** (cross-modal) và **fuse ảnh+text**.
```bash
curl -fL https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/main/model.onnx -o data/models/clip_text.onnx
curl -fL https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz -o data/models/bpe_simple_vocab_16e6.txt.gz
python3 patent/design/match_design.py <ảnh> 5 "mô tả sản phẩm"   # CLIP ảnh + CLIP text
```

## Gộp 1 lệnh: utility + design
```bash
python3 patent/screen_all.py  --text "..." --image ảnh.jpg   # báo cáo CONSOLE
python3 patent/screen_html.py --text "..." --image ảnh.jpg && open patent_report.html  # báo cáo HTML
# ① UTILITY (claims, C++ qua `patent screen-json`)  +  ② DESIGN (kiểu dáng, CLIP) trong 1 báo cáo
```

## Thu thập thêm dữ liệu (Google Patents, design)
```bash
python3 patent/scrape/scrape_google.py <start_num> <count> [workers]   # song song, resumable
#   vd: python3 patent/scrape/scrape_google.py 1000158 5000 5
bash patent/scrape/refresh.sh    # index + tách figure + nhúng CLIP (incremental)
```
Mẹo: design patent trên Google Patents dùng kind-code **S1** (vd `USD645299S1`); ảnh full-res ở
`patentimages.storage.googleapis.com` (CDN công khai). Cào *toàn bộ* ~1.1M design patent cần nhiều
ngày + ~1TB → làm theo từng đợt (resumable).

## Lưu ý độ chính xác design (đã kiểm với ảnh chụp thật)
- **Bản vẽ ↔ bản vẽ**: CLIP tốt (rank-1 69.7%).
- **Ảnh chụp thật ↔ bản vẽ, chỉ ảnh**: YẾU (domain gap) — vd ảnh ba lô thật bị xếp dưới "Lamp".
  Có tiền xử lý edge (`clip_embed.embed(edge="auto")`) nhưng chưa đủ.
- **Ảnh thật + keyword text**: ĐÚNG — ba lô+“backpack” → Backpack #1; umbrella+“umbrella” → umbrella #1.
  → Khi dùng ảnh Amazon thật, **luôn kèm mô tả/keyword**.

## Code (thuần C++, tái dùng mlcpp::Tokenizer)
```
patent/include/patent/  corpus.hpp  bm25.hpp  risk.hpp
patent/src/             corpus.cpp  bm25.cpp  risk.cpp
patent/cli/patent.cpp   # search / show / risk / screen
patent/tests/           test_bm25.cpp  test_risk.cpp  main.cpp
```
