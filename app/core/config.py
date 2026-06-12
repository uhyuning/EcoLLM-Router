# [역할] 환경 변수 및 전역 설정값 관리.
# Gemini API 키, 모델명, 비용 단가, 라우팅 임계값 등을 .env 파일에서 읽어온다.
# 사용법: from app.core.config import settings
#
# 담당: 하윤

from app.config import settings  # noqa: F401
