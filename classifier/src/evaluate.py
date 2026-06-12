# [역할] 분류기의 라우팅 정확도를 데이터셋 전체에 걸쳐 검증한다.
# 입력: classifier/data/processed/labeled.csv (컬럼: prompt, label)
# 출력: Accuracy, Precision, Recall, F1, 혼동 행렬, 오분류 사례
# 실행: python -m classifier.src.evaluate   (프로젝트 루트에서)
#
# 담당: 하윤

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# __file__ 기준으로 절대 경로 고정 — 실행 위치(cwd)에 무관하게 동작
_DATA_PATH  = Path(__file__).parents[1] / "data" / "processed" / "labeled.csv"
_MODEL_PATH = Path(__file__).parents[1] / "artifacts" / "router_model.joblib"

# app/config.py 의 complexity_threshold 기본값과 반드시 동일하게 유지
_THRESHOLD = 0.5

_LABEL_NAMES = ["flash (label=0)", "pro (label=1)"]


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"prompt", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필수 컬럼 없음: {missing}")
    return df


# ── 예측 함수 ─────────────────────────────────────────────────────────────────

def _predict_rule_based(prompts: list[str]) -> list[int]:
    """rule_based.classify() 결과(0/1)를 리스트로 반환."""
    from classifier.src.rule_based import classify
    return [classify(p)["label"] for p in prompts]


def _predict_ml(prompts: list[str]) -> list[int] | None:
    """ML 모델이 존재하면 예측값 반환, 없으면 None.

    None 반환 시 rule-based 결과만 출력한다.
    """
    if not _MODEL_PATH.exists():
        return None
    try:
        import joblib
        from classifier.src.features import extract_features
        model = joblib.load(_MODEL_PATH)
        X = [extract_features(p) for p in prompts]
        return list(model.predict(X))
    except Exception as exc:
        print(f"[WARN] ML 모델 로드 실패 ({exc}) — rule-based만 평가합니다.")
        return None


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def _print_section(
    name: str,
    y_true: list[int],
    y_pred: list[int],
    prompts: list[str],
) -> None:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'=' * 62}")
    print(f"  {name}")
    print(f"{'=' * 62}")
    print(f"\n  Accuracy : {acc:.2%}  ({correct} / {len(y_true)})")

    # sklearn의 classification_report로 Precision / Recall / F1 출력
    print("\n  Classification Report:")
    report = classification_report(y_true, y_pred, target_names=_LABEL_NAMES, digits=3)
    for line in report.splitlines():
        print("    " + line)

    # 혼동 행렬 — 어느 방향으로 오분류가 많은지 확인
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n  Confusion Matrix (행=실제 레이블, 열=예측 레이블):")
    print("                    Pred Flash   Pred Pro")
    print(f"  Actual Flash          {tn:>4}         {fp:>4}   ← Flash를 Pro로 잘못 분류(비용 낭비)")
    print(f"  Actual Pro            {fn:>4}         {tp:>4}   ← Pro를 Flash로 잘못 분류(품질 저하)")

    # 오분류된 프롬프트 — 규칙 개선 방향을 파악하기 위해 최대 5건 출력
    errors = [
        (prompts[i], y_true[i], y_pred[i])
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]
    if errors:
        print(f"\n  오분류 사례 (총 {len(errors)}건 중 최대 5건):")
        for prompt, true_label, pred_label in errors[:5]:
            true_name = "flash" if true_label == 0 else "pro"
            pred_name = "flash" if pred_label == 0 else "pro"
            preview = (prompt[:72] + "...") if len(prompt) > 75 else prompt
            print(f"    [실제 {true_name} → 예측 {pred_name}]  {preview}")
    else:
        print("\n  오분류 없음")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def evaluate(csv_path: Path) -> None:
    df = _load(csv_path)
    prompts = df["prompt"].tolist()
    y_true  = df["label"].tolist()

    n_flash = sum(l == 0 for l in y_true)
    n_pro   = sum(l == 1 for l in y_true)
    print(f"\n데이터셋 : {csv_path}")
    print(f"총 샘플  : {len(df)}건  (Flash={n_flash}, Pro={n_pro})")

    # Rule-based 평가는 항상 실행
    y_rule = _predict_rule_based(prompts)
    _print_section("Rule-based Classifier", y_true, y_rule, prompts)

    # ML 모델 평가 — 학습된 모델 파일이 있을 때만 실행
    y_ml = _predict_ml(prompts)
    if y_ml is not None:
        _print_section("ML Classifier (GBM)", y_true, y_ml, prompts)
    else:
        print(f"\n[INFO] 학습된 ML 모델 없음 ({_MODEL_PATH.relative_to(Path.cwd())})")
        print("       학습 후 비교하려면:")
        print("       python -m classifier.src.train --data classifier/data/processed/labeled.csv")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="라우팅 분류기 정확도 평가")
    parser.add_argument("--data", type=Path, default=_DATA_PATH, help="CSV 경로")
    args = parser.parse_args()
    evaluate(args.data)
