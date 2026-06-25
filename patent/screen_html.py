#!/usr/bin/env python3
# Báo cáo HTML trực quan: 1 input (mô tả + ảnh) -> utility (claims) + design (kiểu dáng).
#   python3 patent/screen_html.py --text "mô tả" [--image ảnh.jpg] [--topk 6]
import argparse
import html
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "patent", "design"))
FIGURES = os.path.join(ROOT, "data", "patent_figures", "figures")
OUTHTML = os.path.join(ROOT, "patent_report.html")
STATUS = {"con_hieu_luc": "CÒN HIỆU LỰC", "het_han": "HẾT HẠN", "don_cong_bo": "ĐƠN (chưa cấp)"}
RISK_COLOR = {"CAO": "#c0392b", "TRUNG BÌNH": "#e67e22", "THẤP": "#27ae60"}


def uri(p):
    return os.path.relpath(os.path.abspath(p), ROOT)


def esc(s):
    return html.escape(str(s))


def utility_cards(text, topk):
    binp = os.path.join(ROOT, "build", "patent")
    if not os.path.exists(binp):
        return '<p>(chưa build CLI utility: <code>make patent</code>)</p>'
    out = subprocess.run([binp, "screen-json", text, str(topk)], cwd=ROOT,
                         capture_output=True, text=True)
    try:
        rows = json.loads(out.stdout)
    except Exception:
        return f'<pre>{esc(out.stdout[:500])}</pre>'
    if not rows:
        return '<p>Không tìm thấy patent utility liên quan.</p>'
    cards = []
    for r in rows:
        col = RISK_COLOR.get(r["risk"], "#777")
        st = STATUS.get(r["status"], r["status"])
        exp = f' (hết hạn ~{r["expiry"]})' if r["expiry"] else ''
        cards.append(f"""
        <div class="ucard" style="border-left:5px solid {col}">
          <div class="urisk" style="color:{col}">RỦI RO {esc(r['risk'])} · khớp {r['n_match']}/{r['n_elem']} đặc điểm</div>
          <div class="utitle">{esc(r['title'])}</div>
          <div class="umeta">{esc(r['id'])} · {esc(r['owner']) or '(không rõ chủ)'} · nộp {r['filing']} · {esc(st)}{exp}</div>
          <div class="uclaim">Claim 1: {esc(r['claim1'])}</div>
        </div>""")
    return "".join(cards)


def design_cards(image, text, topk):
    import match_core
    try:
        rows = match_core.rank(image=image, text=text, topk=topk)
    except Exception as e:
        return f'<p>lỗi design: {esc(e)}</p>'
    cards = []
    for r in rows:
        col = "#c0392b" if r["score"] >= 0.6 else "#e67e22" if r["score"] >= 0.4 else "#27ae60"
        st = STATUS.get(r["status"], r["status"])
        fig = os.path.join(FIGURES, r["id"], r["fig"])
        brk = f'ảnh {r["img"]*100:.0f}% · text {r["txt"]*100:.0f}%' if text.strip() else ''
        cards.append(f"""
        <div class="dcard" style="border-color:{col}">
          <div class="dbar" style="background:{col}">{r['score']*100:.0f}%</div>
          <img src="{uri(fig)}">
          <div class="dmeta"><b>{esc(r['id'])}</b><br>{esc(r['title'])}<br>
            <span class="dsub">{esc(r['owner'])}<br>{esc(st)} · {brk}</span></div>
        </div>""")
    return f'<div class="dgrid">{"".join(cards)}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--topk", type=int, default=6)
    a = ap.parse_args()
    if not a.text and not a.image:
        ap.error("cần --text và/hoặc --image")

    usec = utility_cards(a.text, a.topk) if a.text else '<p>(không có mô tả text)</p>'
    desec = design_cards(a.image, a.text, a.topk) if a.image else '<p>(không có ảnh)</p>'
    
    qimg_header = '· có ảnh' if a.image else ''
    qimg_preview = ""
    if a.image:
        qimg_preview = f"""
        <div class="qimg-container">
          <div class="qimg-label">Ảnh truy vấn của bạn:</div>
          <img class="qimg" src="{uri(a.image)}">
        </div>
        """

    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Patent Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
:root {{
  --bg-color: #0b0f19;
  --panel-bg: rgba(22, 28, 45, 0.6);
  --border-color: rgba(255, 255, 255, 0.08);
  --text-primary: #f3f4f6;
  --text-secondary: #9ca3af;
  --primary-accent: #3b82f6;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
}}
body{{
  font-family: 'Outfit', sans-serif;
  margin: 0;
  background-color: var(--bg-color);
  background-image: radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.1) 0, transparent 50%),
                    radial-gradient(at 50% 0%, rgba(16, 185, 129, 0.05) 0, transparent 50%);
  color: var(--text-primary);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}}
header{{
  background: rgba(17, 24, 39, 0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-color);
  padding: 24px 40px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}}
header h1{{
  margin: 0;
  font-size: 24px;
  font-weight: 700;
  background: linear-gradient(135deg, #60a5fa, #34d399);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: flex;
  align-items: center;
  gap: 10px;
}}
header .q{{
  color: var(--text-secondary);
  font-size: 14px;
  background: rgba(255, 255, 255, 0.03);
  padding: 8px 16px;
  border-radius: 99px;
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  gap: 8px;
}}
.wrap{{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
  gap: 32px;
  padding: 40px;
}}
.col{{
  background: var(--panel-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border-color);
  border-radius: 20px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}}
h2{{
  font-size: 18px;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: 24px;
  color: #fff;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.qimg-container{{
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 24px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  padding: 20px;
  border-radius: 16px;
  align-items: center;
}}
.qimg-label{{
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 500;
  align-self: flex-start;
}}
.qimg{{
  max-height: 240px;
  max-width: 100%;
  object-fit: contain;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: #111827;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  transition: transform 0.3s ease;
}}
.qimg:hover{{
  transform: scale(1.02);
}}
.ucard{{
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 16px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}}
.ucard:hover{{
  transform: translateY(-2px);
  background: rgba(255, 255, 255, 0.04);
  border-color: rgba(255, 255, 255, 0.15);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}}
.urisk{{
  font-weight: 700;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}}
.utitle{{
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #fff;
  line-height: 1.4;
}}
.umeta{{
  color: var(--text-secondary);
  font-size: 13px;
  margin-bottom: 12px;
}}
.uclaim{{
  color: #d1d5db;
  font-size: 13px;
  line-height: 1.6;
  background: rgba(0, 0, 0, 0.2);
  padding: 12px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.03);
}}
.dgrid{{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 16px;
}}
.dcard{{
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  overflow: hidden;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
}}
.dcard:hover{{
  transform: translateY(-4px);
  background: rgba(255, 255, 255, 0.04);
  border-color: rgba(255, 255, 255, 0.15);
  box-shadow: 0 12px 24px rgba(0,0,0,0.3);
}}
.dcard img{{
  width: 100%;
  height: 160px;
  object-fit: contain;
  background: #111827;
  border-bottom: 1px solid var(--border-color);
  transition: transform 0.5s ease;
}}
.dcard:hover img{{
  transform: scale(1.05);
}}
.dbar{{
  color: #fff;
  font-weight: 700;
  text-align: center;
  padding: 6px 12px;
  font-size: 14px;
  letter-spacing: 0.05em;
  background: rgba(255, 255, 255, 0.05);
  border-bottom: 1px solid var(--border-color);
}}
.dmeta{{
  padding: 14px;
  font-size: 13px;
  line-height: 1.4;
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}}
.dmeta b{{
  color: #fff;
  font-size: 14px;
}}
.dsub{{
  color: var(--text-secondary);
  font-size: 12px;
  margin-top: 8px;
  line-height: 1.5;
}}
.foot{{
  padding: 30px 40px;
  color: var(--text-secondary);
  font-size: 13px;
  text-align: center;
  border-top: 1px solid var(--border-color);
  background: rgba(17, 24, 39, 0.4);
  margin-top: 40px;
}}
</style></head><body>
<header><h1>🛡️ Báo cáo sàng lọc rủi ro patent</h1>
<div class="q">Sản phẩm: "{esc(a.text)}" {qimg_header}</div></header>
<div class="wrap">
  <div class="col"><h2>① UTILITY — công năng (claims)</h2>{usec}</div>
  <div class="col"><h2>② DESIGN — kiểu dáng</h2>{qimg_preview}{desec}</div>
</div>
<div class="foot">⚠️ Sàng lọc tự động — KHÔNG phải tư vấn pháp lý / FTO. Ảnh thật nên kèm mô tả.</div>
</body></html>"""

    open(OUTHTML, "w", encoding="utf-8").write(page)
    print(f"Đã tạo: {OUTHTML}\nMở:  open patent_report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
