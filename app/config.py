from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str

    # .env에 없으면 Gemini 공식 최신 Flash/Pro 기본값 사용
    flash_model: str = "gemini-2.0-flash"
    pro_model: str = "gemini-2.5-pro"

    # 0.5는 Flash와 Pro를 동등하게 반반 나누는 중립 기준점.
    # 낮출수록 Pro 사용 빈도 증가 → 비용↑ 품질↑
    # 높일수록 Flash 사용 빈도 증가 → 비용↓ 품질↓
    complexity_threshold: float = 0.5

    # 비용 단가 (USD / 1M tokens) — 벤치마크 비용 계산에 사용
    flash_input_cost_per_1m:  float = 0.075
    flash_output_cost_per_1m: float = 0.30
    pro_input_cost_per_1m:    float = 1.25
    pro_output_cost_per_1m:   float = 10.00

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
