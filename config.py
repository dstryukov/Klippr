from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENROUTER_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    
    OPENROUTER_MODEL: str = "openai/gpt-oss-120b:free"
    GROQ_MODEL: str = "llama3-70b-8192"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
