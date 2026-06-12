# [역할] POST /chat/ 엔드포인트 구현.
# 흐름: 요청 수신 → router.select_model() → llm_client.call() → 응답 반환
# 비용 계산 및 metrics 기록도 이 파일에서 수행한다.
#
# 담당: 하윤

from fastapi import APIRouter

from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services import classifier, router as model_router
from app.services import llm_client

# Gemini 공식 가격표 기준 (USD / 1M tokens), 2025년 기준
# Flash: 저가 모델 — 빠른 응답이 필요한 단순 질문용
# Pro:   고가 모델 — 정확도가 중요한 복잡한 질문용
# 가격이 변경되면 이 딕셔너리만 수정하면 된다.
_COST = {
    "flash": {"input": 0.075, "output": 0.30},
    "pro":   {"input": 1.25,  "output": 10.0},
}

chat_router = APIRouter()


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # 1. 복잡도 점수 계산 (0.0 ~ 1.0)
    complexity_score = classifier.score(req.prompt)

    # 2. 점수와 임계값을 비교해 모델 선택
    #    force_model이 있으면 분류 결과와 무관하게 해당 모델로 고정
    choice = model_router.select_model(complexity_score, req.force_model)

    # 3. 선택된 모델로 LLM 호출
    result = await (
        llm_client.call_flash(req.prompt)
        if choice == "flash"
        else llm_client.call_pro(req.prompt)
    )

    # 4. 실제 사용된 토큰 수로 비용 역산
    #    실제 청구 금액과 다를 수 있으므로 estimated(추정)로 명시
    cost = (
        result["input_tokens"]  * _COST[choice]["input"]  / 1_000_000
        + result["output_tokens"] * _COST[choice]["output"] / 1_000_000
    )

    return ChatResponse(
        answer=result["answer"],
        model_used=choice,
        complexity_score=round(complexity_score, 4),
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        latency_ms=result["latency_ms"],
        estimated_cost_usd=round(cost, 8),  # 소액이므로 소수점 8자리까지 표기
    )
