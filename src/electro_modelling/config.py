import os
import torch
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    MODELS_DIR: Optional[str] = Field(env="MODELS_DIR")
    DATA_DIR: Optional[str] = Field(env="DATA_DIR")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


settings = Settings(
    _env_file=os.environ.get("ENV_FILE") or ".env", _env_file_encoding="utf-8"
)
