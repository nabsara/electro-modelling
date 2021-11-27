import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    MODELS_DIR: Optional[str] = Field(env="MODELS_DIR")


settings = Settings(
    _env_file=os.environ.get("ENV_FILE") or ".env", _env_file_encoding="utf-8"
)
