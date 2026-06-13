# [역할] 라우터(app/services/router.py)의 모델 선택 로직을 검증한다.
# 점수 임계값 경계, force_model 오버라이드 동작 등을 테스트한다.
# 실행: pytest tests/test_router.py

import pytest
from app.services.router import select_model


def test_low_score_routes_to_flash():
    """점수 0.1 → Flash (임계값 0.5 미만)"""
    assert select_model(0.1) == "flash"


def test_high_score_routes_to_pro():
    """점수 0.9 → Pro (임계값 0.5 이상)"""
    assert select_model(0.9) == "pro"


def test_score_at_threshold_routes_to_pro():
    """점수가 임계값(0.5)과 정확히 같으면 Pro로 처리한다 (동점 → 품질 우선)."""
    assert select_model(0.5) == "pro"


def test_force_flash_overrides_high_score():
    """force_model='flash'는 높은 점수를 무시하고 Flash를 강제한다."""
    assert select_model(0.9, force_model="flash") == "flash"


def test_force_pro_overrides_low_score():
    """force_model='pro'는 낮은 점수를 무시하고 Pro를 강제한다."""
    assert select_model(0.1, force_model="pro") == "pro"


def test_invalid_force_model_is_ignored():
    """force_model에 flash/pro 외의 값이 오면 점수 기반으로 정상 라우팅한다."""
    assert select_model(0.1, force_model="unknown") == "flash"
    assert select_model(0.9, force_model="unknown") == "pro"


def test_boundary_just_below_threshold():
    """0.4999는 Flash 구간이어야 한다."""
    assert select_model(0.4999) == "flash"


def test_boundary_just_above_threshold():
    """0.5001은 Pro 구간이어야 한다."""
    assert select_model(0.5001) == "pro"
