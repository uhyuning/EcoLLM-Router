# [역할] POST /chat/ 엔드포인트의 요청(Request) 데이터 스키마.
# Pydantic BaseModel 사용.
# 필드 예시: prompt(str), force_model(Optional[str] — 'flash' 또는 'pro' 강제 지정)
#
# 담당: 하윤

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # min_length=1: 빈 문자열 전송 시 422 에러 반환 — LLM에 빈 요청이 전달되는 것을 방지
    prompt: str = Field(..., min_length=1)

    # 분류기를 우회하고 싶을 때 사용 (벤치마크, A/B 테스트 등)
    # None이면 분류기 결과를 따른다.
    force_model: Optional[Literal["flash", "pro"]] = None
