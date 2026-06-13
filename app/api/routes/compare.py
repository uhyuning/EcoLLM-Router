import asyncio

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.models.response import CompareResponse, ModelResult, QualityResult
from app.services import llm_client, quality_scorer

_COST = {
    "flash": {"input": 0.075, "output": 0.30},
    "pro":   {"input": 1.25,  "output": 10.0},
}

compare_router = APIRouter()


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


@compare_router.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest) -> CompareResponse:
    # Flash와 Pro를 동시에 호출해 레이턴시를 최소화
    flash_res, pro_res = await asyncio.gather(
        llm_client.call_flash(req.prompt),
        llm_client.call_pro(req.prompt),
    )

    # 두 응답이 모두 준비된 뒤 judge 호출
    quality = await quality_scorer.score(
        req.prompt,
        flash_res["answer"],
        pro_res["answer"],
    )

    def _cost(res: dict, model: str) -> float:
        return round(
            (res["input_tokens"] * _COST[model]["input"]
             + res["output_tokens"] * _COST[model]["output"]) / 1_000_000,
            8,
        )

    return CompareResponse(
        prompt=req.prompt,
        flash=ModelResult(
            answer=flash_res["answer"],
            latency_ms=flash_res["latency_ms"],
            estimated_cost_usd=_cost(flash_res, "flash"),
        ),
        pro=ModelResult(
            answer=pro_res["answer"],
            latency_ms=pro_res["latency_ms"],
            estimated_cost_usd=_cost(pro_res, "pro"),
        ),
        quality=QualityResult(**quality),
    )
