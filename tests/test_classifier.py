# [역할] 분류기(app/services/classifier.py)의 복잡도 점수 예측 결과를 검증한다.
# 단순 질문은 낮은 점수, 고난도 질문은 높은 점수가 나와야 한다.
# 실행: pytest tests/test_classifier.py

import pytest
from app.services import classifier


def test_score_range():
    """점수는 항상 0.0~1.0 사이여야 한다."""
    score = classifier.score("테스트 질문입니다.")
    assert 0.0 <= score <= 1.0


def test_simple_prompt_routes_to_flash():
    """짧고 단순한 질문은 0.5 미만(Flash 구간)이어야 한다."""
    score = classifier.score("안녕하세요")
    assert score < 0.5, f"단순 질문 점수가 예상보다 높습니다: {score}"


def test_complex_prompt_routes_to_pro():
    """길고 복잡한 질문은 0.5 이상(Pro 구간)이어야 한다."""
    prompt = (
        "딥러닝 모델에서 배치 정규화(Batch Normalization)와 레이어 정규화(Layer Normalization)의 "
        "수학적 원리를 비교 분석하고, 각각 어떤 아키텍처에 더 적합한지 근거와 함께 자세히 설명해주세요."
    )
    score = classifier.score(prompt)
    assert score >= 0.5, f"복잡한 질문 점수가 예상보다 낮습니다: {score}"


def test_empty_string_does_not_crash():
    """빈 문자열이 들어와도 예외 없이 점수를 반환해야 한다."""
    try:
        score = classifier.score("")
        assert 0.0 <= score <= 1.0
    except Exception as e:
        pytest.fail(f"빈 문자열 처리 중 예외 발생: {e}")


def test_score_is_float():
    """반환값이 float 타입이어야 한다."""
    score = classifier.score("파이썬이란 무엇인가요?")
    assert isinstance(score, float)
