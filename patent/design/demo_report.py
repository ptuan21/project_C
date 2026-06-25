#!/usr/bin/env python3
# Giao diện demo: so 1 ảnh sản phẩm với design patent (CLIP + text) -> trang HTML trực quan.
#   python3 patent/design/demo_report.py <ảnh> [topk] ["mô tả sản phẩm"]
import html
import json
import os
import sys

import match_core

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES = os.path.join(ROOT, "data", "patent_figures", "figures")
CLIP_INDEX = os.path.join(ROOT, "data", "patent_index", "clip_index.json")
OUTHTML = os.path.join(ROOT, "patent_demo.html")
STATUS = {"con_hieu_luc": "CÒN HIỆU LỰC", "het_han": "HẾT HẠN", "don_cong_bo": "ĐƠN (chưa cấp)"}


def uri(path):
    return "file://" + os.path.abspath(path)


def risk_color(s):
    if s >= 0.6:
        return "#c0392b", "Rủi ro CAO"
    if s >= 0.4:
        return "#e67e22", "Rủi ro TB"
    return "#27ae60", "Rủi ro thấp"


def main():
    if len(sys.argv) < 2:
        print("Dùng: demo_report.py <ảnh> [topk] [\"mô tả\"]")
        return 1
    img = sys.argv[1]
    rest = sys.argv[2:]
    topk = 6
    if rest and rest[0].isdigit():
        topk = int(rest[0])
        rest = rest[1:]
    text = " ".join(rest)

    rows = match_core.rank(image=img, text=text, topk=topk)
    nfig = len(json.load(open(CLIP_INDEX, encoding="utf-8")))

    cards = []
    for r in rows:
        color, label = risk_color(r["score"])
        figpath = os.path.join(FIGURES, r["id"], r["fig"])
        brk = (f'<div class="brk">ảnh {r["img"]*100:.0f}% · text {r["txt"]*100:.0f}%</div>'
               if text.strip() else "")
        cards.append(f"""
        <div class="card" style="border-color:{color}">
          <div class="bar" style="background:{color}">{r['score']*100:.0f}% — {label}</div>
          <img src="{uri(figpath)}">
          <div class="meta">
            <b>{html.escape(r['id'])}</b><br>
            <span class="t">{html.escape(r['title'])}</span><br>
            {html.escape(r['owner'])}<br>
            <span class="st">{STATUS.get(r['status'], r['status'])}</span>{brk}
          </div>
        </div>""")

    qtxt = (' + text: <i>' + html.escape(text) + '</i>') if text.strip() else ''
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Patent Design Match</title><style>
body{{font-family:-apple-system,Arial,sans-serif;margin:24px;background:#f4f5f7;color:#222}}
h1{{font-size:20px}} .sub{{color:#666;font-size:13px;margin-bottom:18px}}
.query{{background:#fff;padding:14px;border-radius:10px;display:inline-block;box-shadow:0 1px 4px #0002}}
.query img{{max-height:260px;border:1px solid #ddd}}
.grid{{display:flex;flex-wrap:wrap;gap:14px;margin-top:18px}}
.card{{background:#fff;border:3px solid;border-radius:10px;width:230px;overflow:hidden;box-shadow:0 1px 4px #0002}}
.card img{{width:100%;height:200px;object-fit:contain;background:#fff;border-bottom:1px solid #eee}}
.bar{{color:#fff;font-weight:bold;text-align:center;padding:6px;font-size:13px}}
.meta{{padding:10px;font-size:12px;line-height:1.5}} .t{{color:#444}} .st{{color:#0a7}}
.brk{{color:#999;margin-top:4px;font-size:11px}}
.note{{margin-top:22px;color:#888;font-size:12px}}
</style></head><body>
<h1>🔎 Nhận diện thiết kế tương đồng — Design Patent (CLIP)</h1>
<div class="sub">Ảnh so với {nfig} figure của 41 design patent — CLIP ViT-B/32{' + fuse text BM25' if text.strip() else ''}.</div>
<div class="query"><div><b>Ảnh truy vấn</b>{qtxt}</div><img src="{uri(img)}"></div>
<div class="grid">{''.join(cards)}</div>
<div class="note">Sàng lọc tự động (CLIP embedding + BM25) — KHÔNG phải tư vấn pháp lý.</div>
</body></html>"""

    open(OUTHTML, "w", encoding="utf-8").write(page)
    print(f"Đã tạo: {OUTHTML}\nMở bằng:  open patent_demo.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
