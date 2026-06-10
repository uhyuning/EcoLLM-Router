# [역할] 훈련 데이터 없이 규칙만으로 동작하는 1차 분류기.
# ML 모델 학습 전에 라우팅 로직의 기준선(baseline)을 빠르게 확인하는 용도.
# 실행: python -m classifier.src.rule_based

from __future__ import annotations

from classifier.src.features import (
    avg_word_length,
    char_count,
    complex_keyword_count,
    simple_keyword_count,
    token_count,
)

# ── 임계값 ────────────────────────────────────────────────
TOKEN_THRESHOLD = 30        # 토큰 수가 이 이상이면 '어려운' 쪽으로 가중
CHAR_THRESHOLD = 150        # 글자 수
COMPLEX_KW_THRESHOLD = 1    # 복잡 키워드가 1개 이상이면 어려운 것으로 판정
AVG_WORD_LEN_THRESHOLD = 7  # 평균 단어 길이 (영어 기준 긴 단어 = 전문 용어)


def classify(prompt: str) -> dict:
    """
    반환: {"label": 0 or 1, "reason": str, "scores": dict}
      label 0 = Flash(쉬운), label 1 = Pro(어려운)
    """
    tok = token_count(prompt)
    chars = char_count(prompt)
    ckw = complex_keyword_count(prompt)
    skw = simple_keyword_count(prompt)
    awl = avg_word_length(prompt)

    # ── 규칙 적용 (우선순위 순) ──────────────────────────
    reasons = []

    if ckw >= COMPLEX_KW_THRESHOLD:
        reasons.append(f"복잡 키워드 {ckw}개 감지")

    if tok >= TOKEN_THRESHOLD:
        reasons.append(f"토큰 수 {tok}개 (기준: {TOKEN_THRESHOLD})")

    if chars >= CHAR_THRESHOLD:
        reasons.append(f"글자 수 {chars}자 (기준: {CHAR_THRESHOLD})")

    if awl >= AVG_WORD_LEN_THRESHOLD:
        reasons.append(f"평균 단어 길이 {awl:.1f} (전문 용어 추정)")

    # 단순 키워드가 있고 복잡 신호가 없으면 Flash
    if skw > 0 and not reasons:
        label = 0
        reason = f"단순 키워드 감지 ({skw}개), 복잡 신호 없음"
    elif reasons:
        label = 1
        reason = " | ".join(reasons)
    else:
        label = 0
        reason = "복잡 신호 없음 → 기본값 Flash"

    return {
        "label": label,
        "model": "Pro (Gemini 2.5)" if label == 1 else "Flash (Gemini 2.0)",
        "reason": reason,
        "scores": {
            "token_count": tok,
            "char_count": chars,
            "complex_keywords": ckw,
            "simple_keywords": skw,
            "avg_word_length": round(awl, 2),
        },
    }


# ── 테스트 실행 ───────────────────────────────────────────
if __name__ == "__main__":
    test_prompts = [
        "What is the capital of France?",
        "파리는 어느 나라의 수도야?",
        "Derive the closed-form solution for the time-optimal control of a double integrator system.",
        "트랜스포머 아키텍처에서 Self-Attention의 수식을 유도하고 RNN 대비 장단점을 분석하시오.",
        "머신러닝이 뭔지 한 줄로 요약해줘.",
        "NP-완전 문제의 정의를 설명하고 Cook-Levin 정리로 증명하시오.",
    ]

    print(f"{'프롬프트':<55} {'분류':^20} {'이유'}")
    print("-" * 110)
    for p in test_prompts:
        result = classify(p)
        preview = p[:52] + "..." if len(p) > 55 else p
        print(f"{preview:<55} {result['model']:^20} {result['reason']}")
