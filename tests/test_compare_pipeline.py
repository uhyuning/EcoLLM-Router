"""
POST /compare 전체 파이프라인 통합 테스트.

호출 흐름:
  1. call_flash(user_prompt)  → Flash 응답
  2. call_pro(user_prompt)    → Pro 응답        (1, 2는 asyncio.gather로 병렬)
  3. call_flash(judge_prompt) → 품질 점수 JSON

LLM 호출은 모두 mock으로 대체한다.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ── mock 응답 픽스처 ──────────────────────────────────────────────────────────

_FLASH_RESP = {
    "answer": "Flash: Paris is the capital of France.",
    "input_tokens": 10,
    "output_tokens": 15,
    "latency_ms": 450.0,
}
_PRO_RESP = {
    "answer": "Pro: Paris is the capital of France, famous for the Eiffel Tower.",
    "input_tokens": 10,
    "output_tokens": 40,
    "latency_ms": 1800.0,
}
_JUDGE_RESP = {
    "answer": '{"score_a": 6, "score_b": 9}',
    "input_tokens": 200,
    "output_tokens": 10,
    "latency_ms": 600.0,
}


def _flash_side_effect(prompt: str):
    # judge 프롬프트는 "impartial judge" 문구로 구분
    if "impartial judge" in prompt:
        return _JUDGE_RESP
    return _FLASH_RESP


@pytest.fixture
def patched_llm():
    with (
        patch(
            "app.services.llm_client.call_flash",
            new=AsyncMock(side_effect=_flash_side_effect),
        ),
        patch(
            "app.services.llm_client.call_pro",
            new=AsyncMock(return_value=_PRO_RESP),
        ),
    ):
        yield


# ── 응답 구조 검증 ────────────────────────────────────────────────────────────

def test_response_top_level_keys(patched_llm):
    resp = client.post("/compare", json={"prompt": "What is the capital of France?"})
    assert resp.status_code == 200
    assert set(resp.json().keys()) == {"prompt", "flash", "pro", "quality"}


def test_response_echoes_prompt(patched_llm):
    prompt = "What is the capital of France?"
    resp = client.post("/compare", json={"prompt": prompt})
    assert resp.json()["prompt"] == prompt


def test_flash_field_keys(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    assert set(resp.json()["flash"].keys()) == {"answer", "latency_ms", "estimated_cost_usd"}


def test_pro_field_keys(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    assert set(resp.json()["pro"].keys()) == {"answer", "latency_ms", "estimated_cost_usd"}


def test_quality_field_keys(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    assert set(resp.json()["quality"].keys()) == {"score_flash", "score_pro", "quality_gap"}


# ── LLM 응답이 올바르게 전달되는지 ────────────────────────────────────────────

def test_flash_answer_matches_mock(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    assert resp.json()["flash"]["answer"] == _FLASH_RESP["answer"]


def test_pro_answer_matches_mock(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    assert resp.json()["pro"]["answer"] == _PRO_RESP["answer"]


def test_latency_ms_matches_mock(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    data = resp.json()
    assert data["flash"]["latency_ms"] == _FLASH_RESP["latency_ms"]
    assert data["pro"]["latency_ms"] == _PRO_RESP["latency_ms"]


# ── 품질 점수 검증 ────────────────────────────────────────────────────────────

def test_quality_scores_from_judge(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    quality = resp.json()["quality"]
    assert quality["score_flash"] == 6.0
    assert quality["score_pro"] == 9.0


def test_quality_gap_equals_pro_minus_flash(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    quality = resp.json()["quality"]
    expected_gap = round(quality["score_pro"] - quality["score_flash"], 2)
    assert quality["quality_gap"] == expected_gap


def test_quality_gap_positive_when_pro_better(patched_llm):
    # judge mock: score_a(flash)=6, score_b(pro)=9 → gap > 0
    resp = client.post("/compare", json={"prompt": "test"})
    assert resp.json()["quality"]["quality_gap"] > 0


# ── 비용 계산 검증 ────────────────────────────────────────────────────────────

def test_flash_cheaper_than_pro(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    data = resp.json()
    assert data["flash"]["estimated_cost_usd"] < data["pro"]["estimated_cost_usd"]


def test_estimated_cost_positive(patched_llm):
    resp = client.post("/compare", json={"prompt": "test"})
    data = resp.json()
    assert data["flash"]["estimated_cost_usd"] > 0
    assert data["pro"]["estimated_cost_usd"] > 0


# ── 입력 유효성 검증 ──────────────────────────────────────────────────────────

def test_empty_prompt_returns_422():
    resp = client.post("/compare", json={"prompt": ""})
    assert resp.status_code == 422


def test_missing_prompt_field_returns_422():
    resp = client.post("/compare", json={})
    assert resp.status_code == 422
