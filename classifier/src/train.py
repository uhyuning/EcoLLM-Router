# [역할] 라우팅 분류기를 학습하고 모델 파일을 저장한다.
# 입력: classifier/data/processed/labeled.csv (컬럼: prompt, label)
# 출력: classifier/artifacts/router_model.joblib
# 실행: python -m classifier.src.train --data classifier/data/processed/labeled.csv

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classifier.src.features import extract_features, FEATURE_NAMES

ARTIFACT_PATH = Path("classifier/artifacts/router_model.joblib")


def load_dataset(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    # Expected columns: 'prompt', 'label'  (label: 0=flash, 1=pro)
    X = [extract_features(p) for p in df["prompt"]]
    y = df["label"].tolist()
    return X, y


def train(data_path: str) -> None:
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    print(classification_report(y_test, clf.predict(X_test), target_names=["flash", "pro"]))

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ARTIFACT_PATH)
    print(f"Model saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    train(args.data)
