import time
from typing import TypedDict

import google.generativeai as genai

from app.config import settings

genai.configure(api_key=settings.gemini_api_key)


class LLMResponse(TypedDict):
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def _call(model_name: str, prompt: str) -> LLMResponse:
    model = genai.GenerativeModel(model_name)

    start = time.perf_counter()
    response = model.generate_content(prompt)
    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.usage_metadata
    return LLMResponse(
        answer=response.text,
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
        latency_ms=round(latency_ms, 2),
    )


def call_flash(prompt: str) -> LLMResponse:
    return _call(settings.flash_model, prompt)


def call_pro(prompt: str) -> LLMResponse:
    return _call(settings.pro_model, prompt)
