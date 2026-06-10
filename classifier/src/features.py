# [역할] 프롬프트에서 수치형 피처를 추출한다.
# extract_features(prompt) 가 반환하는 고정 길이 리스트를 분류기 학습·예측에 사용한다.
# FEATURE_NAMES 는 학습 시 컬럼명으로 활용한다.

from __future__ import annotations

import re

_COMPLEX_KW = re.compile(
    r"\b(proof|derive|solve|optimize|algorithm|complexity|theorem|"
    r"differential|integral|quantum|architecture|trade.?off|analyze|"
    r"compare|evaluate|논증|증명|풀어|설계|분석|비교)\b",
    re.IGNORECASE,
)

_SIMPLE_KW = re.compile(
    r"\b(what is|who is|when|where|define|translate|summarize|"
    r"뭐야|언제|어디|정의|번역|요약)\b",
    re.IGNORECASE,
)


def token_count(prompt: str) -> int:
    return len(prompt.split())


def char_count(prompt: str) -> int:
    return len(prompt)


def complex_keyword_count(prompt: str) -> int:
    return len(_COMPLEX_KW.findall(prompt))


def simple_keyword_count(prompt: str) -> int:
    return len(_SIMPLE_KW.findall(prompt))


def question_mark_count(prompt: str) -> int:
    return prompt.count("?")


def avg_word_length(prompt: str) -> float:
    words = prompt.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def extract_features(prompt: str) -> list[float]:
    """Returns a fixed-length feature vector for the classifier."""
    return [
        token_count(prompt),
        char_count(prompt),
        complex_keyword_count(prompt),
        simple_keyword_count(prompt),
        question_mark_count(prompt),
        avg_word_length(prompt),
    ]


FEATURE_NAMES = [
    "token_count",
    "char_count",
    "complex_keyword_count",
    "simple_keyword_count",
    "question_mark_count",
    "avg_word_length",
]
