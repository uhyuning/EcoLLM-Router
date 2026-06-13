# [역할] 기준선(Baseline) 측정 스크립트.
# 모든 프롬프트를 라우팅 없이 Gemini Pro에만 보내 비용·지연 시간을 기록한다.
# 입력: benchmarks/prompts.txt
# 출력: benchmarks/results/baseline.jsonl
# 실행: python benchmarks/scripts/run_baseline.py

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import google.generativeai as genai

from app.config import settings

genai.configure(api_key=settings.gemini_api_key)

PROMPTS_FILE = Path("benchmarks/prompts.txt")
RESULTS_FILE = Path("benchmarks/results/baseline.jsonl")


async def run() -> None:
    prompts = [p.strip() for p in PROMPTS_FILE.read_text(encoding="utf-8").splitlines() if p.strip()]
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    model = genai.GenerativeModel(settings.pro_model)

    print(f"총 {len(prompts)}개 프롬프트 — 전부 Pro({settings.pro_model})로 전송\n")

    with RESULTS_FILE.open("w", encoding="utf-8") as fout:
        for i, prompt in enumerate(prompts, 1):
            start = time.perf_counter()
            response = await asyncio.to_thread(model.generate_content, prompt)
            latency_ms = (time.perf_counter() - start) * 1000

            usage = response.usage_metadata
            cost = (
                usage.prompt_token_count    / 1_000_000 * settings.pro_input_cost_per_1m
                + usage.candidates_token_count / 1_000_000 * settings.pro_output_cost_per_1m
            )
            record = {
                "id": i,
                "prompt": prompt[:80],
                "model": settings.pro_model,
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "latency_ms": round(latency_ms, 2),
                "cost_usd": round(cost, 8),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i:>3}/{len(prompts)}] latency={record['latency_ms']:>8.1f}ms  cost=${record['cost_usd']:.8f}")

    print(f"\n결과 저장 완료 → {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run())
