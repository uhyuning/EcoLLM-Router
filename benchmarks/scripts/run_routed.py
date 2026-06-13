# [역할] EcoLLM Router 경유 측정 스크립트.
# 서버의 POST /chat 를 호출하여 실제 라우팅 결과(모델, 비용, 지연)를 기록한다.
# 사전 조건: python app/main.py 로 서버가 실행 중이어야 한다.
# 입력: benchmarks/prompts.txt
# 출력: benchmarks/results/routed.jsonl
# 실행: python benchmarks/scripts/run_routed.py

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
PROMPTS_FILE = Path("benchmarks/prompts.txt")
RESULTS_FILE = Path("benchmarks/results/routed.jsonl")


async def run() -> None:
    prompts = [p.strip() for p in PROMPTS_FILE.read_text(encoding="utf-8").splitlines() if p.strip()]
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"총 {len(prompts)}개 프롬프트 — EcoLLM Router 경유\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        with RESULTS_FILE.open("w", encoding="utf-8") as fout:
            for i, prompt in enumerate(prompts, 1):
                resp = await client.post("/chat", json={"prompt": prompt})
                resp.raise_for_status()
                data = resp.json()

                record = {
                    "id": i,
                    "prompt": prompt[:80],
                    "model": data["model_used"],
                    "complexity_score": data["complexity_score"],
                    "input_tokens": data["input_tokens"],
                    "output_tokens": data["output_tokens"],
                    "latency_ms": data["latency_ms"],
                    "cost_usd": data["estimated_cost_usd"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[{i:>3}/{len(prompts)}] "
                    f"model={data['model_used']:<6}  "
                    f"score={data['complexity_score']:.3f}  "
                    f"latency={data['latency_ms']:>8.1f}ms  "
                    f"cost=${data['estimated_cost_usd']:.8f}"
                )

    print(f"\n결과 저장 완료 → {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run())
