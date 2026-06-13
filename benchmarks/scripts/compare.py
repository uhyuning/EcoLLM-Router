# [역할] baseline.jsonl 과 routed.jsonl 을 비교하여 성능 지표를 출력한다.
# 출력 항목: 총 비용(USD), 평균 비용/호출, 평균 지연(ms),
#           비용 절감률(%), 지연 개선률(%), Flash 분기 비율(%)
# 실행: python benchmarks/scripts/compare.py

from __future__ import annotations

import json
from pathlib import Path

BASELINE_FILE = Path("benchmarks/results/baseline.jsonl")
ROUTED_FILE   = Path("benchmarks/results/routed.jsonl")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(records: list[dict]) -> dict:
    costs     = [r["cost_usd"]   for r in records]
    latencies = [r["latency_ms"] for r in records]
    n = len(records)
    return {
        "n":              n,
        "total_cost":     sum(costs),
        "avg_cost":       sum(costs) / n,
        "avg_latency_ms": sum(latencies) / n,
    }


def main() -> None:
    if not BASELINE_FILE.exists() or not ROUTED_FILE.exists():
        print("결과 파일이 없습니다. run_baseline.py와 run_routed.py를 먼저 실행하세요.")
        return

    baseline = load(BASELINE_FILE)
    routed   = load(ROUTED_FILE)

    b = summarize(baseline)
    r = summarize(routed)

    cost_saving_pct    = (b["total_cost"]     - r["total_cost"])     / b["total_cost"]     * 100
    latency_saving_pct = (b["avg_latency_ms"] - r["avg_latency_ms"]) / b["avg_latency_ms"] * 100
    flash_count        = sum(1 for rec in routed if "flash" in rec["model"].lower())
    flash_rate         = flash_count / len(routed) * 100

    print("=" * 56)
    print(f"  {'지표':<26} {'Baseline':>12} {'Routed':>12}")
    print("-" * 56)
    print(f"  {'총 비용 (USD)':<26} {b['total_cost']:>12.6f} {r['total_cost']:>12.6f}")
    print(f"  {'평균 비용/호출 (USD)':<24} {b['avg_cost']:>12.8f} {r['avg_cost']:>12.8f}")
    print(f"  {'평균 지연 (ms)':<26} {b['avg_latency_ms']:>12.2f} {r['avg_latency_ms']:>12.2f}")
    print("=" * 56)
    print(f"  비용 절감률     : {cost_saving_pct:>+.1f}%")
    print(f"  지연 개선률     : {latency_saving_pct:>+.1f}%")
    print(f"  Flash 분기 비율 : {flash_rate:.1f}%  ({flash_count}/{len(routed)}건)")
    print("=" * 56)


if __name__ == "__main__":
    main()
