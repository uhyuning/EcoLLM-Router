# [역할] EcoLLM Router 경유 측정 스크립트.
# 서버의 POST /chat/ 를 호출하여 실제 라우팅 결과(모델, 비용, 지연)를 기록한다.
# 사전 조건: uvicorn app.main:app --reload 로 서버가 실행 중이어야 한다.
# 입력: benchmarks/prompts.txt
# 출력: benchmarks/results/routed.jsonl
# 실행: python benchmarks/scripts/run_routed.py
