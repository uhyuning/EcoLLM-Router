# EcoLLM Router — 단위 테스트 및 통합 테스트 결과 보고서

**작성일:** 2026년 6월 13일
**테스트 실행 결과:** 74개 테스트 전체 통과 (0 실패, 0 오류)

---

## 1. 테스트 환경

| 항목 | 내용 |
|---|---|
| 운영체제 | macOS (Darwin 25.5.0, arm64) |
| Python | 3.11.0 |
| pytest | 9.0.3 |
| FastAPI | 0.136.3 |
| Pydantic | 2.13.4 |
| uvicorn | 0.49.0 |
| google-generativeai | 0.8.6 |
| scikit-learn | 1.9.0 |

---

## 2. 테스트 구조 개요

```
tests/
├── test_classifier.py       — 복잡도 분류기 단위 테스트        (39개)
├── test_router.py           — 모델 라우터 단위 테스트          (20개)
└── test_compare_pipeline.py — /compare 엔드포인트 통합 테스트  (15개)
```

---

## 3. 실행 결과 요약

| 테스트 파일 | 테스트 수 | 통과 | 실패 | 오류 |
|---|:---:|:---:|:---:|:---:|
| `test_classifier.py` | 39 | 39 | 0 | 0 |
| `test_router.py` | 20 | 20 | 0 | 0 |
| `test_compare_pipeline.py` | 15 | 15 | 0 | 0 |
| **합계** | **74** | **74** | **0** | **0** |

**전체 실행 시간:** 0.65초

---

## 4. 테스트 케이스 상세

### 4.1 `test_classifier.py` — 복잡도 분류기 (39개)

복잡도 분류기(`app/services/classifier.py`)의 동작을 검증한다.
ML 모델 파일이 없는 경우 자동으로 규칙 기반(rule-based) fallback으로 전환되며,
두 모드 모두에서 동일한 공개 인터페이스(`score()`)를 통해 0.0~1.0의 점수를 반환한다.

#### 4.1.1 `score()` — 복잡도 점수 범위 검증

| 구분 | 프롬프트 | 예상 | 결과 |
|---|---|:---:|:---:|
| 단순 | What is the capital of France? | < 0.5 | PASS |
| 단순 | When was Python created? | < 0.5 | PASS |
| 단순 | Who invented the telephone? | < 0.5 | PASS |
| 단순 | Where is the Eiffel Tower located? | < 0.5 | PASS |
| 단순 | What is machine learning? | < 0.5 | PASS |
| 복잡 | Derive the closed-form solution for the time-optimal control of a double integrator. | ≥ 0.5 | PASS |
| 복잡 | Compare and evaluate the trade-offs of sorting algorithms in terms of complexity. | ≥ 0.5 | PASS |
| 복잡 | Analyze the quantum computing architecture for fault-tolerant error correction. | ≥ 0.5 | PASS |
| 복잡 | Design an algorithm to optimize distributed system throughput and evaluate its trade-offs. | ≥ 0.5 | PASS |
| 복잡 | Prove the correctness of a recursive algorithm using mathematical induction. | ≥ 0.5 | PASS |

#### 4.1.2 `score()` — 출력 범위 보장 (10개)

위 10개 프롬프트 전체에 대해 `0.0 ≤ score(prompt) ≤ 1.0` 조건을 검증. 전체 PASS.

#### 4.1.3 복잡도 키워드 감지 (`_complex_kw_count`)

| 입력 | 예상 최소 감지 수 | 결과 |
|---|:---:|:---:|
| "derive the solution" | 1 | PASS |
| "compare and evaluate the algorithm" | 3 | PASS |
| "analyze the architecture and trade-off" | 3 | PASS |
| "what is the capital of France?" | 0 | PASS |
| "when was it created?" | 0 | PASS |

#### 4.1.4 단순 키워드 감지 (`_simple_kw_count`)

| 입력 | 예상 최소 감지 수 | 결과 |
|---|:---:|:---:|
| "what is machine learning?" | 1 | PASS |
| "where is the Eiffel Tower?" | 1 | PASS |
| "who is the president?" | 1 | PASS |
| "derive and solve this" | 0 | PASS |
| "analyze the algorithm" | 0 | PASS |

#### 4.1.5 기타 단위 테스트

| 테스트 | 검증 내용 | 결과 |
|---|---|:---:|
| `test_complex_signal_overrides_simple_keyword` | 단순 키워드("what is")와 복잡 키워드("algorithm")가 동시에 존재할 때 복잡이 우선 → label=1 | PASS |
| `test_rule_based_label` (4케이스) | 규칙 기반 레이블 정확성 (단순, 복잡, 빈 문자열) | PASS |
| `test_extract_features_returns_six_floats` | 피처 벡터 길이 6, 모두 수치형 | PASS |
| `test_extract_features_empty_prompt` | 빈 입력에서 token=0, char=0, avg_word_len=0.0 | PASS |
| `test_token_count` | 공백 분리 토큰 수 계산 정확성 | PASS |
| `test_avg_word_length_empty` | 빈 입력 시 0.0 반환 | PASS |

---

### 4.2 `test_router.py` — 모델 라우터 (20개)

모델 선택 로직(`app/services/router.py`)을 검증한다.
`complexity_threshold`는 `monkeypatch`로 0.5에 고정해 `.env` 설정값에 무관하게 결정적(deterministic)으로 실행된다.

#### 4.2.1 임계값 기반 분기

| 복잡도 점수 | 예상 모델 | 결과 |
|:---:|:---:|:---:|
| 0.0 | flash | PASS |
| 0.1 | flash | PASS |
| 0.49 | flash | PASS |
| 0.499 | flash | PASS |
| 0.5 (동점) | pro | PASS |
| 0.51 | pro | PASS |
| 0.8 | pro | PASS |
| 1.0 | pro | PASS |

> **동점 처리 정책:** `score == threshold`일 때 Pro를 선택한다. 품질 우선 원칙.

#### 4.2.2 `force_model` 오버라이드

| 복잡도 점수 | force_model | 예상 모델 | 결과 |
|:---:|:---:|:---:|:---:|
| 0.9 (복잡) | "flash" | flash | PASS |
| 0.1 (단순) | "pro" | pro | PASS |
| 0.5 (경계) | "flash" | flash | PASS |
| 0.499 (경계) | "pro" | pro | PASS |
| 0.3 | None | flash | PASS |
| 0.7 | None | pro | PASS |

#### 4.2.3 유효하지 않은 `force_model` 값 처리 (10개)

`"turbo"`, `"gpt4"`, `""`, `"Flash"` (대문자), `"Pro"` (대문자) 등 잘못된 값은 무시되고 복잡도 점수 기반으로 판단된다. 낮은 점수(0.3) → flash, 높은 점수(0.7) → pro. 전체 PASS.

---

### 4.3 `test_compare_pipeline.py` — `/compare` 통합 테스트 (15개)

`POST /compare` 엔드포인트의 전체 파이프라인을 검증한다.

**파이프라인 흐름:**
```
asyncio.gather(call_flash(prompt), call_pro(prompt))   ← 병렬 LLM 호출
            ↓
quality_scorer.score(flash_answer, pro_answer)          ← LLM-as-judge 채점
            ↓
CompareResponse { flash, pro, quality }                 ← 최종 응답
```

외부 LLM 호출은 `unittest.mock.AsyncMock`으로 대체하며, judge 호출은 프롬프트 내 `"impartial judge"` 문구로 구분한다.

**Mock 설정:**

| 호출 | 반환값 |
|---|---|
| `call_flash(user_prompt)` | answer: "Flash: Paris is the capital of France.", latency: 450ms |
| `call_pro(user_prompt)` | answer: "Pro: Paris is the capital...", latency: 1800ms |
| `call_flash(judge_prompt)` | answer: `{"score_a": 6, "score_b": 9}` |

#### 4.3.1 응답 구조 검증

| 테스트 | 검증 내용 | 결과 |
|---|---|:---:|
| `test_response_top_level_keys` | 응답 최상위 키: `prompt`, `flash`, `pro`, `quality` | PASS |
| `test_response_echoes_prompt` | 입력 프롬프트가 응답에 그대로 반환 | PASS |
| `test_flash_field_keys` | flash 필드: `answer`, `latency_ms`, `estimated_cost_usd` | PASS |
| `test_pro_field_keys` | pro 필드: `answer`, `latency_ms`, `estimated_cost_usd` | PASS |
| `test_quality_field_keys` | quality 필드: `score_flash`, `score_pro`, `quality_gap` | PASS |

#### 4.3.2 LLM 응답 전달 검증

| 테스트 | 검증 내용 | 결과 |
|---|---|:---:|
| `test_flash_answer_matches_mock` | Flash 응답 텍스트가 mock 반환값과 일치 | PASS |
| `test_pro_answer_matches_mock` | Pro 응답 텍스트가 mock 반환값과 일치 | PASS |
| `test_latency_ms_matches_mock` | Flash 450ms, Pro 1800ms 그대로 반환 | PASS |

#### 4.3.3 품질 점수 검증

| 테스트 | 검증 내용 | 결과 |
|---|---|:---:|
| `test_quality_scores_from_judge` | score_flash=6.0, score_pro=9.0 (judge JSON 파싱) | PASS |
| `test_quality_gap_equals_pro_minus_flash` | quality_gap = score_pro − score_flash = 3.0 | PASS |
| `test_quality_gap_positive_when_pro_better` | Pro 우위 시 quality_gap > 0 | PASS |

#### 4.3.4 비용 계산 검증

| 테스트 | 검증 내용 | 결과 |
|---|---|:---:|
| `test_flash_cheaper_than_pro` | Flash 비용 < Pro 비용 (토큰 단가 기준) | PASS |
| `test_estimated_cost_positive` | Flash, Pro 각각 비용 > 0 | PASS |

#### 4.3.5 입력 유효성 검사

| 테스트 | 입력 | 예상 HTTP 코드 | 결과 |
|---|---|:---:|:---:|
| `test_empty_prompt_returns_422` | `{"prompt": ""}` | 422 | PASS |
| `test_missing_prompt_field_returns_422` | `{}` | 422 | PASS |

---

## 5. 비용 계산 기준

테스트에서 사용한 Gemini 모델별 단가 (USD / 1M tokens, 2025년 기준):

| 모델 | 입력 단가 | 출력 단가 |
|---|:---:|:---:|
| Gemini 2.0 Flash | $0.075 | $0.30 |
| Gemini 2.5 Pro | $1.25 | $10.0 |

---

## 6. 설계 검증 요약

| 검증 항목 | 결과 |
|---|:---:|
| 분류기: 단순 질문 → score < 0.5 | 확인 |
| 분류기: 복잡 질문 → score ≥ 0.5 | 확인 |
| 분류기: 출력 범위 [0.0, 1.0] 보장 | 확인 |
| 분류기: ML 없을 때 rule-based fallback 동작 | 확인 |
| 라우터: 임계값 미만 → Flash 선택 | 확인 |
| 라우터: 임계값 이상 → Pro 선택 | 확인 |
| 라우터: 동점(score == threshold) → Pro 우선 | 확인 |
| 라우터: force_model 오버라이드 정상 동작 | 확인 |
| 라우터: 유효하지 않은 force_model 무시 | 확인 |
| 파이프라인: Flash·Pro 병렬 호출 | 확인 |
| 파이프라인: LLM-as-judge 품질 채점 연동 | 확인 |
| 파이프라인: quality_gap 계산 정확성 | 확인 |
| 파이프라인: Flash 비용 < Pro 비용 | 확인 |
| API: 빈 프롬프트 422 에러 반환 | 확인 |

---

## 7. 최종 실행 로그

```
============================= test session starts ==============================
platform darwin -- Python 3.11.0, pytest-9.0.3, pluggy-1.6.0
rootdir: /Users/kanghayun/Desktop/EcoLLM-Router
plugins: anyio-4.13.0
collected 74 items

tests/test_classifier.py .......................................         [ 52%]
tests/test_compare_pipeline.py ...............                          [ 72%]
tests/test_router.py ....................                                [100%]

======================== 74 passed, 1 warning in 0.65s =========================
```

> **경고 1건:** `google.generativeai` 패키지 deprecated 알림 (기능 동작에는 영향 없음).
> 향후 `google.genai` 패키지로 마이그레이션 예정.
