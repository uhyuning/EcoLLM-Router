from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str
    flash_model: str = "gemini-2.0-flash"
    pro_model: str = "gemini-2.5-pro"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
