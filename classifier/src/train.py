# [역할] 라우팅 분류기를 학습하고 모델 파일을 저장한다.
# 입력: classifier/data/processed/labeled.csv (컬럼: prompt, label)
# 출력: classifier/artifacts/router_model.joblib
# 실행: python -m classifier.src.train

from __future__ import annotations

import csv
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classifier.src.features import extract_features

# ── 경로 설정 ──────────────────────────────────────────────────
DATA_PATH     = Path("classifier/data/processed/labeled.csv")
ARTIFACT_PATH = Path("classifier/artifacts/router_model.joblib")


# ── 1단계: 데이터 로드 ─────────────────────────────────────────
def load_data(csv_path: Path) -> tuple[list, list]:
    """CSV에서 프롬프트와 라벨을 읽어 반환한다."""
    prompts, labels = [], []
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            prompts.append(row["prompt"])
            labels.append(int(row["label"]))
    return prompts, labels


# ── 2단계: 피처 추출 ───────────────────────────────────────────
def build_feature_matrix(prompts: list[str]) -> list[list[float]]:
    """프롬프트 리스트를 수치형 피처 행렬로 변환한다."""
    return [extract_features(p) for p in prompts]


# ── 3단계: 학습 ────────────────────────────────────────────────
def train(X: list, y: list) -> LogisticRegression:
    """로지스틱 회귀 분류기를 학습한다."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


# ── 4단계: 평가 ────────────────────────────────────────────────
def evaluate(model: LogisticRegression, X_test: list, y_test: list) -> None:
    """테스트 셋으로 정확도와 분류 리포트를 출력한다."""
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["Flash(쉬운)", "Pro(어려운)"]))


# ── 5단계: 저장 ────────────────────────────────────────────────
def save(model: LogisticRegression) -> None:
    """학습된 모델을 joblib 파일로 저장한다."""
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)
    print(f"모델 저장 완료 → {ARTIFACT_PATH}")


# ── 실행 진입점 ────────────────────────────────────────────────
if __name__ == "__main__":
    print("[ 1 ] 데이터 로드 중...")
    prompts, labels = load_data(DATA_PATH)
    print(f"      총 {len(prompts)}개 프롬프트 로드 (Flash: {labels.count(0)}, Pro: {labels.count(1)})")

    print("[ 2 ] 피처 추출 중...")
    X = build_feature_matrix(prompts)

    print("[ 3 ] 학습/테스트 분할 (8:2)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    print("[ 4 ] 모델 학습 중...")
    model = train(X_train, y_train)

    print("[ 5 ] 평가 결과:")
    evaluate(model, X_test, y_test)

    print("[ 6 ] 모델 저장 중...")
    save(model)
