# [역할] POST /chat/ 엔드포인트의 응답(Response) 데이터 스키마.
# Pydantic BaseModel 사용.
# 필드 예시: answer, model_used, complexity_score, input_tokens,
#           output_tokens, latency_ms, estimated_cost_usd
#
# 담당: 하윤

from pydantic import BaseModel


class ModelResult(BaseModel):
    answer: str
    latency_ms: float
    estimated_cost_usd: float


class QualityResult(BaseModel):
    score_flash: float
    score_pro: float
    quality_gap: float  # score_pro - score_flash; 양수 = Pro가 더 좋음


class CompareResponse(BaseModel):
    prompt: str
    flash: ModelResult
    pro: ModelResult
    quality: QualityResult


class ChatResponse(BaseModel):
    answer: str
    model_used: str          # 실제로 사용된 모델 ("flash" | "pro")
    complexity_score: float  # 분류기가 예측한 복잡도 점수 (0.0 ~ 1.0)
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_usd: float  # API 호출 한 건의 추정 비용 (실제 청구액과 다를 수 있음)
