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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
