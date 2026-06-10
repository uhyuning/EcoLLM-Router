# [역할] POST /chat/ 엔드포인트 구현.
# 흐름: 요청 수신 → router.select_model() → llm_client.call() → 응답 반환
# 비용 계산 및 metrics 기록도 이 파일에서 수행한다.
#
# 담당: 하윤
