# [역할] 복잡도 점수를 바탕으로 라우팅 대상 모델을 결정한다.
# 점수 < COMPLEXITY_THRESHOLD  → Gemini Flash (저비용·고속)
# 점수 >= COMPLEXITY_THRESHOLD → Gemini Pro   (고정확)
# force_model 파라미터로 수동 override 가능.
#
# 담당: 하윤

from __future__ import annotations

from typing import Literal, Optional

from app.config import settings

ModelChoice = Literal["flash", "pro"]


def select_model(
    complexity_score: float,
    force_model: Optional[str] = None,
) -> ModelChoice:
    """복잡도 점수를 보고 'flash' 또는 'pro'를 반환한다."""

    # force_model은 디버깅·A/B 테스트·벤치마크 시 분류기를 우회해
    # 특정 모델로 고정 호출할 때 사용한다.
    if force_model in ("flash", "pro"):
        return force_model  # type: ignore[return-value]  # Literal 검증은 ChatRequest에서 이미 완료

    # 동점(score == threshold)은 Pro로 처리 — 애매한 경우 품질을 우선
    if complexity_score >= settings.complexity_threshold:
        return "pro"
    return "flash"
