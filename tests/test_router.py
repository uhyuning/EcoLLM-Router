import pytest

from app.services.router import select_model


# ── 임계값을 테스트마다 0.5로 고정 ────────────────────────────────────────────
# .env의 complexity_threshold 설정값에 무관하게 결과가 결정적이어야 한다.
@pytest.fixture(autouse=True)
def fixed_threshold(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "complexity_threshold", 0.5)


# ── 임계값 미만 → Flash ────────────────────────────────────────────────────────

@pytest.mark.parametrize("score", [0.0, 0.1, 0.49, 0.499])
def test_below_threshold_selects_flash(score):
    assert select_model(score) == "flash"


# ── 임계값 이상 → Pro ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("score", [0.5, 0.51, 0.8, 1.0])
def test_at_or_above_threshold_selects_pro(score):
    assert select_model(score) == "pro"


def test_exact_threshold_tie_goes_to_pro():
    # 동점(score == threshold)은 품질 우선으로 Pro 선택
    assert select_model(0.5) == "pro"


# ── force_model 오버라이드 ─────────────────────────────────────────────────────

def test_force_flash_overrides_high_score():
    # 복잡한 질문(0.9)이어도 force_model="flash"이면 Flash 강제
    assert select_model(0.9, force_model="flash") == "flash"


def test_force_pro_overrides_low_score():
    # 단순한 질문(0.1)이어도 force_model="pro"이면 Pro 강제
    assert select_model(0.1, force_model="pro") == "pro"


def test_force_flash_overrides_boundary():
    assert select_model(0.5, force_model="flash") == "flash"


def test_force_pro_overrides_boundary():
    assert select_model(0.499, force_model="pro") == "pro"


# ── force_model=None은 점수 기반으로 동작 ─────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (0.3, "flash"),
    (0.7, "pro"),
])
def test_force_model_none_uses_score(score, expected):
    assert select_model(score, force_model=None) == expected


# ── 유효하지 않은 force_model 값은 무시하고 점수로 판단 ─────────────────────

@pytest.mark.parametrize("invalid", ["turbo", "gpt4", "", "Flash", "Pro"])
def test_invalid_force_model_falls_through_to_score(invalid):
    # 낮은 점수 + 잘못된 force_model → score 기반으로 Flash 선택
    assert select_model(0.3, force_model=invalid) == "flash"
    # 높은 점수 + 잘못된 force_model → score 기반으로 Pro 선택
    assert select_model(0.7, force_model=invalid) == "pro"
