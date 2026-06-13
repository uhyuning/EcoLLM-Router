from __future__ import annotations

import json
import re
from typing import TypedDict

from app.services import llm_client

# ── judge 프롬프트 ────────────────────────────────────────────────────────────
# 레이블을 "Flash / Pro"가 아닌 "A / B"로 익명화하는 이유:
# LLM judge는 모델 이름을 알면 선호도가 생길 수 있다 (GPT > 무명 등).
# 어떤 모델이 A인지는 코드가 알고 있으므로 judge 자체는 블라인드로 채점하게 한다.
_JUDGE_PROMPT = """\
You are a strict, impartial judge evaluating two AI responses to the same question.

[Question]
{prompt}

[Response A]
{answer_a}

[Response B]
{answer_b}

Score each response from 0 to 10 based on:
- Accuracy: Is the information correct?
- Completeness: Does it fully address the question?
- Clarity: Is it easy to understand?

Respond ONLY with a JSON object — no explanation, no markdown, no extra text:
{{"score_a": <0-10>, "score_b": <0-10>}}
"""


class QualityScore(TypedDict):
    score_flash: float
    score_pro: float
    quality_gap: float  # score_pro - score_flash; 양수 = Pro가 더 좋음


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

async def score(
    prompt: str,
    flash_answer: str,
    pro_answer: str,
) -> QualityScore:
    """Flash를 judge로 사용해 두 응답의 품질 점수를 반환한다.

    flash_answer → Response A, pro_answer → Response B (블라인드 채점).
    judge 응답 파싱 실패 시 ValueError를 발생시킨다.
    """
    judge_prompt = _JUDGE_PROMPT.format(
        prompt=prompt,
        answer_a=flash_answer,
        answer_b=pro_answer,
    )
    result = await llm_client.call_flash(judge_prompt)
    scores = _parse_scores(result["answer"])

    score_flash = round(scores["score_a"], 2)
    score_pro = round(scores["score_b"], 2)

    return QualityScore(
        score_flash=score_flash,
        score_pro=score_pro,
        quality_gap=round(score_pro - score_flash, 2),
    )


# ── 내부 파싱 헬퍼 ────────────────────────────────────────────────────────────

def _parse_scores(text: str) -> dict[str, float]:
    """judge 응답 텍스트에서 JSON을 추출해 score_a / score_b를 반환한다.

    judge가 마크다운 코드블록이나 설명을 붙이더라도 첫 번째 {...} 블록을 추출한다.
    """
    match = re.search(r'\{[^}]+\}', text)
    if not match:
        raise ValueError(f"Judge returned no JSON object: {text!r}")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned malformed JSON: {text!r}") from exc

    try:
        return {
            "score_a": float(data["score_a"]),
            "score_b": float(data["score_b"]),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Judge JSON missing expected keys: {data!r}") from exc
