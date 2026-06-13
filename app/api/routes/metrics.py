from fastapi import APIRouter

from app.services.metrics_store import MetricsSnapshot, store

metrics_router = APIRouter()


@metrics_router.get("/metrics/", response_model=MetricsSnapshot)
def metrics() -> MetricsSnapshot:
    return store.snapshot()
