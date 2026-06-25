import json
import os
import sys
import imutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX = os.path.join(ROOT, "data", "patent_index")
# Index trên FIGURE đã tách (sạch hơn cả trang) — chạy segment.py trước.
FIGURES = os.path.join(ROOT, "data", "patent_figures", "figures")
OUT = os.path.join(INDEX, "design_index.json")

def design_patents():
    """Đọc meta.tsv -> dict id->(owner,status) cho các patent design."""
    out = {}
    with open(os.path.join(INDEX, "meta.tsv"), encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 11 and c[1] == "design":
                out[c[0]] = {"owner": c[8], "status": c[3], "title": c[10]}
    return out

def main():
    designs = design_patents()
    records = []
    done = 0
    for pid, meta in designs.items():
        d = os.path.join(FIGURES, pid)
        if not os.path.isdir(d):
            continue
        figs = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
        for fig in figs:
            try:
                desc = imutil.describe(os.path.join(d, fig))
            except Exception as e:
                print(f"  bỏ {pid}/{fig}: {e}")
                continue
            records.append({"id": pid, "fig": fig, **desc, **meta})
        done += 1
        print(f"  [{done}/{len(designs)}] {pid}: {len(figs)} figure")

    json.dump(records, open(OUT, "w", encoding="utf-8"))
    print(f"\nXong: {OUT} — {len(records)} figure từ {done} design patent")

if __name__ == "__main__":
    sys.exit(main())
