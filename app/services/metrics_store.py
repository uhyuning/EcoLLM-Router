from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


class MetricsSnapshot(BaseModel):
    total_calls: int
    flash_calls: int
    pro_calls: int
    avg_cost_usd: float
    avg_latency_ms: float


@dataclass
class _MetricsStore:
    total_calls: int = 0
    flash_calls: int = 0
    pro_calls: int = 0
    _total_cost_usd: float = field(default=0.0, repr=False)
    _total_latency_ms: float = field(default=0.0, repr=False)

    def record(self, model: str, cost_usd: float, latency_ms: float) -> None:
        """호출 한 건의 결과를 누적한다. chat.py에서 매 응답 직전에 호출한다."""
        self.total_calls += 1
        if model == "flash":
            self.flash_calls += 1
        else:
            self.pro_calls += 1
        self._total_cost_usd += cost_usd
        self._total_latency_ms += latency_ms

    def snapshot(self) -> MetricsSnapshot:
        """현재까지의 누적 통계를 반환한다. 호출이 없으면 평균값은 0.0."""
        n = self.total_calls
        return MetricsSnapshot(
            total_calls=n,
            flash_calls=self.flash_calls,
            pro_calls=self.pro_calls,
            avg_cost_usd=round(self._total_cost_usd / n, 8) if n else 0.0,
            avg_latency_ms=round(self._total_latency_ms / n, 2) if n else 0.0,
        )  # type: ignore[call-arg]


# 프로세스 수명과 동일한 인메모리 싱글턴.
# 서버 재시작 시 초기화된다.
store = _MetricsStore()
