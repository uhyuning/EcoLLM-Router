# [역할] raw/dataset.json 을 읽어 학습용 labeled.csv 와 엑셀 파일을 생성한다.
# 실행: python classifier/data/processed/make_csv.py

import json
import csv
from pathlib import Path

RAW = Path("classifier/data/raw/dataset.json")
CSV_OUT = Path("classifier/data/processed/labeled.csv")
XLSX_OUT = Path("classifier/data/processed/labeled.xlsx")


def main():
    data = json.loads(RAW.read_text(encoding="utf-8"))

    # CSV 생성 (학습 스크립트용)
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "label", "note"])
        writer.writeheader()
        writer.writerows(data)
    print(f"CSV 저장 완료: {CSV_OUT}")

    # 엑셀 생성 (검토·라벨링용)
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "dataset"
        ws.append(["id", "prompt", "label (0=Flash/쉬운, 1=Pro/어려운)", "note"])
        for row in data:
            ws.append([row["id"], row["prompt"], row["label"], row["note"]])
        ws.column_dimensions["B"].width = 80
        ws.column_dimensions["D"].width = 20
        wb.save(XLSX_OUT)
        print(f"엑셀 저장 완료: {XLSX_OUT}")
    except ImportError:
        print("openpyxl 미설치 — 엑셀 생성 건너뜀 (pip install openpyxl)")


if __name__ == "__main__":
    main()
