# [역할] 4가지 분류기를 동일한 데이터셋으로 비교 평가한다.
# 비교 모델: Logistic Regression / SVM / Decision Tree / Random Forest
# 평가 방식: Stratified K-Fold (k=5) 교차 검증
# 실행: python -m classifier.src.compare_classifiers

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classifier.src.features import extract_features

DATA_PATH = Path("classifier/data/processed/labeled.csv")

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}


def load_data(csv_path: Path) -> tuple[list, list]:
    prompts, labels = [], []
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            prompts.append(row["prompt"])
            labels.append(int(row["label"]))
    return prompts, labels


def main() -> None:
    print("데이터 로드 중...")
    prompts, labels = load_data(DATA_PATH)
    X = [extract_features(p) for p in prompts]
    y = np.array(labels)
    print(f"총 {len(y)}개 샘플 (Flash={sum(y==0)}, Pro={sum(y==1)})\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    results = {}
    for name, model in MODELS.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            "accuracy":  scores["test_accuracy"].mean(),
            "precision": scores["test_precision_macro"].mean(),
            "recall":    scores["test_recall_macro"].mean(),
            "f1":        scores["test_f1_macro"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
        }

    # ── 결과 출력 ───────────────────────────────────────────────
    print(f"{'모델':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Std':>8}")
    print("-" * 74)
    for name, r in results.items():
        print(
            f"{name:<22}"
            f"{r['accuracy']:>10.4f}"
            f"{r['precision']:>10.4f}"
            f"{r['recall']:>10.4f}"
            f"{r['f1']:>10.4f}"
            f"{r['accuracy_std']:>8.4f}"
        )

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\n최고 성능 모델: {best}  (F1 = {results[best]['f1']:.4f})")
    print("\n* Std(표준편차)가 낮을수록 데이터 분할에 관계없이 안정적으로 동작함")


if __name__ == "__main__":
    main()
