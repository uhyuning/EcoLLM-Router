# [역할] 학습된 분류기의 라우팅 정확도와 Flash 분기 비율을 측정한다.
# 입력: classifier/data/processed/labeled.csv
# 출력: 라우팅 정확도(%), Flash/Pro 분기 비율(%) 콘솔 출력
# 실행: python -m classifier.src.evaluate --data classifier/data/processed/labeled.csv

from __future__ import annotations

import argparse

import pandas as pd

from app.services.classifier import predict
from app.core.config import settings


def evaluate(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        score = predict(row["prompt"])
        routed_model = "pro" if score >= settings.complexity_threshold else "flash"
        true_model = "pro" if row["label"] == 1 else "flash"
        results.append({"true": true_model, "predicted": routed_model, "score": score})

    res_df = pd.DataFrame(results)
    accuracy = (res_df["true"] == res_df["predicted"]).mean()
    flash_rate = (res_df["predicted"] == "flash").mean()

    print(f"Routing accuracy : {accuracy:.2%}")
    print(f"Flash route rate : {flash_rate:.2%}  (cost-saving proxy)")
    print(f"Pro route rate   : {1 - flash_rate:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    evaluate(args.data)
