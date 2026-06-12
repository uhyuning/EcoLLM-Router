# [역할] 학습된 ML 모델을 로드하여 프롬프트 복잡도 점수(0.0~1.0)를 반환한다.
# classifier/artifacts/router_model.joblib 를 읽어온다.
# 모델 파일이 없으면 휴리스틱 fallback으로 동작한다.
#
# ML 모델 학습 코드는 classifier/src/ 폴더를 참고할 것.
#
# 담당: 하윤

from __future__ import annotations

from pathlib import Path

# 프로젝트 루트 기준 절대 경로로 고정 — 실행 위치(cwd)에 무관하게 동작하도록
_MODEL_PATH = Path(__file__).parents[3] / "classifier" / "artifacts" / "router_model.joblib"

_model = None
_use_ml = False

# joblib과 모델 파일 모두 선택적 의존성.
# 학습 전(개발 초기) 또는 CI 환경에서도 서버가 기동될 수 있도록 예외를 삼킨다.
try:
    import joblib
    if _MODEL_PATH.exists():
        _model = joblib.load(_MODEL_PATH)
        _use_ml = True
except Exception:
    pass


def score(prompt: str) -> float:
    """복잡도 점수를 0.0~1.0으로 반환한다. 높을수록 Pro가 적합."""
    if _use_ml and _model is not None:
        from classifier.src.features import extract_features
        features = [extract_features(prompt)]

        # 확률 예측이 가능한 모델(LogisticRegression 등)은 클래스 1의 확률을 직접 점수로 사용.
        # predict만 지원하는 모델(SVM 등)은 0/1 레이블을 점수로 근사한다.
        if hasattr(_model, "predict_proba"):
            return float(_model.predict_proba(features)[0][1])
        return float(_model.predict(features)[0])

    return _rule_based_score(prompt)


def _rule_based_score(prompt: str) -> float:
    """ML 모델 없을 때 사용하는 휴리스틱 점수."""
    from classifier.src.rule_based import classify
    result = classify(prompt)

    # rule_based는 이진 레이블(0/1)만 반환하므로 연속 점수로 변환.
    # 0.2 / 0.8로 설정한 이유: 임계값 0.5 기준 충분한 여유를 두어
    # 경계 케이스에서 임계값을 잘못 넘는 상황을 줄이기 위함.
    return 0.8 if result["label"] == 1 else 0.2
