from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", _REPO_ROOT / ".env.local"),
        extra="ignore",
    )

    anthropic_api_key: str = ""
    gptzero_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    max_analyze_chars: int = 80_000
    max_chat_chars: int = 16_000
    max_memory_chars: int = 32_000
    max_theme_fit_section_chars: int = 12_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
