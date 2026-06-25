#!/usr/bin/env python3
# P0 — Index bộ patent USPTO (offline, không API) cho engine C++.
# Quét data/uspto_out/text/*.json -> trích metadata + claims sạch + tính hiệu lực.
# Xuất:
#   data/patent_index/meta.tsv          (1 dòng / patent: metadata + hiệu lực)
#   data/patent_index/docs.tsv          (id <tab> text tìm kiếm 1 dòng, cho BM25)
#   data/patent_index/claims/<id>.txt   (claims sạch nhiều dòng, để hiển thị)
import glob
import html
import json
import os
import re

# patent/index/patent_index.py -> ROOT là project_C (lên 3 cấp)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "uspto_out")
OUT = os.path.join(ROOT, "data", "patent_index")
TODAY_YEAR = 2026

os.makedirs(os.path.join(OUT, "claims"), exist_ok=True)


def first(v):
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def year(v):
    s = str(first(v))
    m = re.match(r"(\d{4})", s)
    return int(m.group(1)) if m else 0


def clean_html(s, keep_lines=False):
    if not s:
        return ""
    s = str(s)
    if keep_lines:
        s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
        s = re.sub(r"</(p|div|li)>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    if keep_lines:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
        return "\n".join(ln for ln in lines if ln)
    return re.sub(r"\s+", " ", s).strip()


def legal_status(kind_code, type_, filing_y, grant_y):
    # A1 / US-PGPUB = đơn công bố, chưa cấp -> chưa thực thi được.
    if kind_code == "A1" or type_ == "US-PGPUB":
        return "don_cong_bo", 0
    if kind_code == "S":  # design: hết hạn 15 năm sau khi cấp
        exp = (grant_y or filing_y) + 15
    else:                 # utility (B1/B2/E...): ~20 năm từ ngày nộp
        exp = filing_y + 20
    status = "con_hieu_luc" if exp >= TODAY_YEAR else "het_han"
    return status, exp


def main():
    files = sorted(glob.glob(os.path.join(SRC, "text", "*.json")))
    
    meta_path = os.path.join(OUT, "meta.tsv")
    docs_path = os.path.join(OUT, "docs.tsv")
    
    meta_entries = {}
    header = "id\tkind\tkindCode\tstatus\tfiling\tgrant\texpiry\tn_images\towner\tcpc\ttitle\n"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                header = lines[0]
                for line in lines[1:]:
                    parts = line.rstrip("\n").split("\t")
                    if parts:
                        meta_entries[parts[0]] = parts
                        
    docs_entries = {}
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    docs_entries[parts[0]] = parts[1]
                elif len(parts) == 1:
                    docs_entries[parts[0]] = ""

    n = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        pid = os.path.basename(f)[:-5]  # bỏ '.json' -> US-12635681-B1
        kc = first(d.get("kindCode"))
        type_ = d.get("type") or ""
        kind = "design" if kc == "S" else "utility"
        title = clean_html(d.get("inventionTitle"))
        owner = clean_html(first(d.get("assigneeName")) or first(d.get("applicantName")))
        filing_y = year(d.get("applicationFilingDate"))
        grant_y = year(d.get("datePublished"))
        status, exp = legal_status(kc, type_, filing_y, grant_y)
        cpc = clean_html(d.get("cpcInventiveFlattened"))

        img_dir = os.path.join(SRC, "images", pid)
        n_img = 0
        if os.path.isdir(img_dir):
            n_img = sum(1 for x in os.listdir(img_dir) if not x.startswith("."))

        abstract = clean_html(d.get("abstractHtml"))
        claims = clean_html(d.get("claimsHtml"), keep_lines=True)

        # metadata
        row = [pid, kind, kc, status, str(filing_y), str(grant_y), str(exp),
               str(n_img), owner, cpc, title]
        meta_entries[pid] = row

        # text tìm kiếm (1 dòng)
        search = " ".join([title, abstract, claims.replace("\n", " ")])
        search = re.sub(r"\s+", " ", search).strip()
        docs_entries[pid] = search

        # claims đầy đủ để hiển thị
        with open(os.path.join(OUT, "claims", pid + ".txt"), "w", encoding="utf-8") as cf:
            cf.write("TITLE: " + title + "\n\n")
            if abstract:
                cf.write("ABSTRACT: " + abstract + "\n\n")
            cf.write("CLAIMS:\n" + claims + "\n")
        n += 1

    # Ghi lại toàn bộ
    with open(meta_path, "w", encoding="utf-8") as meta_f:
        meta_f.write(header)
        for pid in sorted(meta_entries.keys()):
            meta_f.write("\t".join(c.replace("\t", " ") for c in meta_entries[pid]) + "\n")

    with open(docs_path, "w", encoding="utf-8") as docs_f:
        for pid in sorted(docs_entries.keys()):
            docs_f.write(pid + "\t" + docs_entries[pid] + "\n")

    print(f"Đã cập nhật/index {n} patent mới -> {OUT}")
    print(f"Tổng số patent trong index hiện tại: {len(meta_entries)}")
    print("  meta.tsv, docs.tsv, claims/<id>.txt")


if __name__ == "__main__":
    main()

