# [역할] GET /metrics/ 엔드포인트 구현.
# 서버 실행 중 누적된 호출 수, 평균 비용(USD), 평균 지연 시간(ms) 통계를 반환한다.
# 현재는 인메모리 저장 방식 (서버 재시작 시 초기화).
#
# 담당: 하윤

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import APIRouter

metrics_router = APIRouter()


@dataclass
class _Store:
    total_calls: int = 0
    flash_calls: int = 0
    pro_calls: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0


_store = _Store()


def record(model: str, cost_usd: float, latency_ms: float) -> None:
    """chat.py에서 호출 완료 후 통계를 누적한다."""
    _store.total_calls += 1
    if model == "flash":
        _store.flash_calls += 1
    else:
        _store.pro_calls += 1
    _store.total_cost_usd += cost_usd
    _store.total_latency_ms += latency_ms


@metrics_router.get("/metrics")
def get_metrics() -> dict:
    n = _store.total_calls
    if n == 0:
        return {"total_calls": 0, "message": "아직 호출 기록이 없습니다."}

    return {
        "total_calls": n,
        "flash_calls": _store.flash_calls,
        "pro_calls": _store.pro_calls,
        "flash_rate_pct": round(_store.flash_calls / n * 100, 1),
        "avg_cost_usd": round(_store.total_cost_usd / n, 8),
        "avg_latency_ms": round(_store.total_latency_ms / n, 2),
        "total_cost_usd": round(_store.total_cost_usd, 6),
    }
