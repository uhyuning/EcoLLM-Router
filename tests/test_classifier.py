import pytest

from app.services.classifier import (
    _avg_word_length,
    _complex_kw_count,
    _extract_features,
    _rule_based_label,
    _simple_kw_count,
    _token_count,
    score,
)

# ── 테스트 데이터 ──────────────────────────────────────────────────────────────

_SIMPLE = [
    "What is the capital of France?",
    "When was Python created?",
    "Who invented the telephone?",
    "Where is the Eiffel Tower located?",
    "What is machine learning?",
]

_COMPLEX = [
    "Derive the closed-form solution for the time-optimal control of a double integrator.",
    "Compare and evaluate the trade-offs of sorting algorithms in terms of complexity.",
    "Analyze the quantum computing architecture for fault-tolerant error correction.",
    "Design an algorithm to optimize distributed system throughput and evaluate its trade-offs.",
    "Prove the correctness of a recursive algorithm using mathematical induction.",
]


# ── score() 공개 인터페이스 ────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt", _SIMPLE)
def test_simple_prompt_scores_low(prompt):
    assert score(prompt) < 0.5, f"단순 질문인데 점수가 높음: {prompt!r}"


@pytest.mark.parametrize("prompt", _COMPLEX)
def test_complex_prompt_scores_high(prompt):
    assert score(prompt) >= 0.5, f"복잡한 질문인데 점수가 낮음: {prompt!r}"


@pytest.mark.parametrize("prompt", _SIMPLE + _COMPLEX)
def test_score_always_in_unit_range(prompt):
    s = score(prompt)
    assert 0.0 <= s <= 1.0, f"score가 [0, 1] 범위를 벗어남: {s}"


# ── 복잡도 키워드 감지 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt,expected_min", [
    ("derive the solution",                    1),
    ("compare and evaluate the algorithm",     3),
    ("analyze the architecture and trade-off", 3),
    ("what is the capital of France?",         0),
    ("when was it created?",                   0),
])
def test_complex_keyword_count(prompt, expected_min):
    assert _complex_kw_count(prompt) >= expected_min


# ── 단순 키워드 감지 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt,expected_min", [
    ("what is machine learning?",  1),
    ("where is the Eiffel Tower?", 1),
    ("who is the president?",      1),
    ("derive and solve this",      0),
    ("analyze the algorithm",      0),
])
def test_simple_keyword_count(prompt, expected_min):
    assert _simple_kw_count(prompt) >= expected_min


# ── 단순/복잡 신호가 동시에 있으면 복잡이 우선 ────────────────────────────────

def test_complex_signal_overrides_simple_keyword():
    # "what is"(단순)가 있어도 "algorithm"(복잡)이 있으면 Pro로 분류
    assert _rule_based_label("what is the most efficient algorithm?") == 1


# ── _rule_based_label ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt,expected", [
    ("What is the capital of France?",                               0),
    ("Derive the closed-form solution for a double integrator.",     1),
    ("Compare and evaluate sorting algorithms by complexity.",       1),
    ("",                                                             0),
])
def test_rule_based_label(prompt, expected):
    assert _rule_based_label(prompt) == expected


# ── 피처 벡터 형식 ────────────────────────────────────────────────────────────

def test_extract_features_returns_six_floats():
    feats = _extract_features("hello world")
    assert len(feats) == 6
    assert all(isinstance(f, (int, float)) for f in feats)


def test_extract_features_empty_prompt():
    feats = _extract_features("")
    assert feats[0] == 0   # token_count
    assert feats[1] == 0   # char_count
    assert feats[5] == 0.0 # avg_word_length


def test_token_count():
    assert _token_count("hello world foo") == 3
    assert _token_count("") == 0


def test_avg_word_length_empty():
    assert _avg_word_length("") == 0.0
