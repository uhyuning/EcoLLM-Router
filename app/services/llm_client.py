# [역할] Google Generative AI SDK를 감싼 비동기 LLM 호출 클라이언트.
# 지수 백오프(exponential back-off) 재시도 로직과 에러 분류 처리를 포함한다.
# 반환값: answer, input_tokens, output_tokens, latency_ms
#
# 담당: 하윤

from __future__ import annotations

import asyncio
import time
from typing import TypedDict

import google.generativeai as genai
from google.api_core import exceptions as gapi_exc

from app.config import settings
from app.core.exceptions import (
    LLMAuthError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
)

# configure는 프로세스당 한 번만 호출하면 된다.
# 모듈 임포트 시점에 실행하여 요청마다 재설정되는 낭비를 막는다.
genai.configure(api_key=settings.gemini_api_key)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # 초 단위 — 재시도 간격: 1s → 2s → 4s

# ResourceExhausted(429), ServiceUnavailable(503), InternalServerError(500):
# 일시적 서버 과부하이므로 대기 후 재시도하면 해소될 가능성이 있다.
_RETRYABLE_ERRORS = (
    gapi_exc.ResourceExhausted,
    gapi_exc.ServiceUnavailable,
    gapi_exc.InternalServerError,
)

# Unauthenticated(401), PermissionDenied(403), InvalidArgument(400):
# API 키 문제나 잘못된 요청이므로 재시도해도 반드시 같은 오류가 반복된다.
_NON_RETRYABLE_ERRORS = (
    gapi_exc.Unauthenticated,
    gapi_exc.PermissionDenied,
    gapi_exc.InvalidArgument,
)


class LLMResponse(TypedDict):
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


async def call(model_name: str, prompt: str) -> LLMResponse:
    """지수 백오프로 재시도하며 비동기 LLM 호출을 수행한다.

    에러 분류:
      - 재시도 가능(429/5xx): 최대 _MAX_RETRIES번 시도 후 LLMRateLimitError / LLMServiceUnavailableError
      - 재시도 불가(401/403): 즉시 LLMAuthError
      - 응답 파싱 실패: 즉시 LLMInvalidResponseError
    """
    model = genai.GenerativeModel(model_name)
    # 타입 체커를 위한 sentinel — _MAX_RETRIES > 0이면 루프 진입 전에 이 값이 쓰일 일은 없다.
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(_MAX_RETRIES):
        try:
            # time.perf_counter: 단조 증가 고해상도 클록 — 시스템 시간 변경에 영향받지 않아
            # 레이턴시 측정에 적합 (time.time() 대신 사용)
            start = time.perf_counter()
            response = await model.generate_content_async(prompt)
            latency_ms = (time.perf_counter() - start) * 1000

            # response.text 접근은 콘텐츠 필터가 전체 응답을 차단했을 때 ValueError를 던진다.
            # 이 경우 재시도해도 같은 결과이므로 바로 502로 변환한다.
            try:
                answer = response.text
            except ValueError as exc:
                raise LLMInvalidResponseError(
                    f"LLM returned no usable text (blocked by content filter?): {exc}"
                ) from exc  # from exc: 원본 traceback을 보존해 디버깅 시 Gemini 응답 내용을 볼 수 있다.

            usage = response.usage_metadata
            return LLMResponse(
                answer=answer,
                input_tokens=usage.prompt_token_count,
                output_tokens=usage.candidates_token_count,
                latency_ms=round(latency_ms, 2),
            )

        except LLMInvalidResponseError:
            # LLMInvalidResponseError는 Exception의 하위 클래스이므로
            # 이 guard가 없으면 아래 `except Exception` 블록에 걸려 불필요한 재시도가 발생한다.
            raise

        except _NON_RETRYABLE_ERRORS as exc:
            # 인증·권한 오류는 재시도해도 해소되지 않으므로 즉시 포기
            raise LLMAuthError(str(exc)) from exc

        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))

        except Exception as exc:
            # 네트워크 단절 등 미분류 오류도 재시도 대상으로 처리
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))

    # 루프가 여기까지 도달했다 == 재시도를 모두 소진.
    # 각 except 블록에서 즉시 raise하지 않고 loop 종료 후 한 곳에서 변환하는 이유:
    # 루프 중간에 raise하면 "마지막 시도에서 대기 없이 바로 포기"하는 흐름을 한 곳에서 제어할 수 없다.
    if isinstance(last_exc, gapi_exc.ResourceExhausted):
        raise LLMRateLimitError(str(last_exc)) from last_exc
    raise LLMServiceUnavailableError(str(last_exc)) from last_exc


async def call_flash(prompt: str) -> LLMResponse:
    return await call(settings.flash_model, prompt)


async def call_pro(prompt: str) -> LLMResponse:
    return await call(settings.pro_model, prompt)
