# [역할] GET /metrics/ 엔드포인트 구현.
# 서버 실행 중 누적된 호출 수, 평균 비용(USD), 평균 지연 시간(ms) 통계를 반환한다.
# 현재는 인메모리 저장 방식 (서버 재시작 시 초기화).
#
# 담당: 하윤

from fastapi import APIRouter

from app.services.metrics_store import MetricsSnapshot, store

metrics_router = APIRouter()


@metrics_router.get("/metrics/", response_model=MetricsSnapshot)
def metrics() -> MetricsSnapshot:
    return store.snapshot()
