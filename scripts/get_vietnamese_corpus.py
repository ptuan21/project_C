import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
OUT = os.path.join(DATA, "vietnamese_large.txt")
UA = {"User-Agent": "mlcpp-corpus/1.0 (educational)"}

TITLES = [
    # Khoa học & công nghệ
    "Khoa học", "Vật lý học", "Hóa học", "Sinh học", "Toán học", "Thiên văn học",
    "Vũ trụ", "Trái Đất", "Tiến hóa", "Di truyền học", "Năng lượng", "Nguyên tử",
    "Ánh sáng", "Điện", "Trí tuệ nhân tạo", "Máy tính", "Internet", "Robot",
    "Học máy", "Thuật toán", "Cơ sở dữ liệu", "Hệ điều hành", "Mạng máy tính",
    "Kỹ thuật phần mềm", "Mật mã học", "Điện tử học", "Vật liệu học",
    # Y học & Sức khỏe
    "Y học", "Giải phẫu người", "Tim", "Não người", "Máu", "Vi khuẩn", "Virus",
    "Vắc-xin", "Kháng sinh", "Hệ miễn dịch", "Ung thư", "Đái tháo đường",
    "Dinh dưỡng", "COVID-19", "Tế bào", "DNA", "Tế bào gốc", "Hệ thần kinh",
    "Hệ tuần hoàn", "Hệ tiêu hóa", "Hệ hô hấp", "Dược lý học", "Sức khỏe tâm thần",
    # Văn học & Nghệ thuật
    "Văn học", "Văn học Việt Nam", "Thơ", "Tiểu thuyết", "Truyện Kiều", "Nguyễn Du",
    "Nam Cao", "Ngô Tất Tố", "Tố Hữu", "Xuân Diệu", "Hồ Xuân Hương", "Văn học dân gian",
    "Nghệ thuật", "Điện ảnh", "Hội họa", "Âm nhạc Việt Nam", "Kiến trúc", "Điêu khắc",
    "Chèo", "Cải lương", "Trịnh Công Sơn",
    # Chính trị, Luật & Xã hội
    "Chính trị", "Nhà nước", "Dân chủ", "Hiến pháp", "Quốc hội", "Pháp luật",
    "Việt Nam", "Hồ Chí Minh", "Liên Hợp Quốc", "Kinh tế", "Đảng Cộng sản Việt Nam",
    "Toàn cầu hóa", "Xã hội học", "Nhân học", "Luật dân sự", "Luật hình sự",
    "Luật pháp Việt Nam", "Tổ chức Thương mại Thế giới", "Hiệp hội các quốc gia Đông Nam Á",
    # Lịch sử, địa lý, văn hóa
    "Lịch sử Việt Nam", "Hà Nội", "Thành phố Hồ Chí Minh", "Địa lý Việt Nam",
    "Triết học", "Văn hóa Việt Nam", "Giáo dục", "Âm nhạc", "Phật giáo", "Ngôn ngữ",
    "Lịch sử thế giới", "Cách mạng công nghiệp", "Thế chiến thứ nhất", "Thế chiến thứ hai",
    "Bản đồ học", "Đông Nam Á", "Biển Đông", "Sông Mê Kông", "Sông Hồng",
    "Hải Phòng", "Đà Nẵng", "Cần Thơ", "Huế", "Sa Pa", "Vịnh Hạ Long",
    "Khổng giáo", "Đạo giáo", "Kitô giáo", "Hồi giáo", "Phong tục tập quán Việt Nam",
    "Tết Nguyên Đán", "Ẩm thực Việt Nam",
    # Thể thao
    "Bóng đá", "Thể thao Việt Nam", "Thế vận hội",
]


API = "https://vi.wikipedia.org/w/api.php"


def fetch_extract(title):
    params = {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "redirects": "1", "format": "json", "titles": title,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):  # retry với backoff khi bị 429
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            for p in d["query"]["pages"].values():
                return p.get("extract", "") or ""
            return ""
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    return ""


def clean(text):
    text = re.sub(r"\n{3,}", "\n\n", text)        # gộp dòng trống
    text = re.sub(r"==+[^=]+==+", "", text)        # bỏ tiêu đề mục "== ... =="
    return text.strip()


def fetch_truyen_kieu():
    url = "https://raw.githubusercontent.com/duyet/truyenkieu-word2vec/master/truyen_kieu_data.txt"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8")
    except Exception as e:
        print("  (bỏ qua Truyện Kiều:", e, ")")
        return ""
    # bỏ tiền tố "<số>.."
    lines = [re.sub(r"^[0-9]+\.\.\s*", "", ln) for ln in raw.splitlines()]
    return "\n".join(lines)


def main():
    parts = []

    # 1) Ca dao - tục ngữ có sẵn
    cadao = os.path.join(DATA, "vietnamese.txt")
    if os.path.exists(cadao):
        parts.append(open(cadao, encoding="utf-8").read())
        print("Ca dao - tục ngữ: OK")

    # 2) Wikipedia đa lĩnh vực
    ok, total = 0, len(TITLES)
    for i, t in enumerate(TITLES, 1):
        try:
            ext = clean(fetch_extract(t))
            if len(ext) > 400:
                parts.append(ext)
                ok += 1
                print(f"  [{i}/{total}] {t}: {len(ext)} ký tự")
            else:
                print(f"  [{i}/{total}] {t}: quá ngắn, bỏ")
        except Exception as e:
            print(f"  [{i}/{total}] {t}: lỗi ({e})")
        time.sleep(0.8)  # lịch sự với máy chủ Wikipedia
    print(f"Wikipedia: {ok}/{total} bài")

    # 3) Truyện Kiều
    kieu = fetch_truyen_kieu()
    if kieu:
        parts.append(kieu)
        print("Truyện Kiều: OK")

    text = "\n\n".join(parts)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nXong: {OUT} ({len(text.encode('utf-8'))} byte, {len(text)} ký tự)")
    print("Train:  ./build/gpt_demo 3000 data/vietnamese_large.txt")


if __name__ == "__main__":
    sys.exit(main())
