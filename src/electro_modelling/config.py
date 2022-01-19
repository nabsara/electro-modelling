# -*- coding: utf-8 -*-

"""
Module that defines in a pydantic Settings class instance with global
variables.

if the ENV_FILE variable is set to provide a deployement ENV file
then we will use this ENV file (example : prod.env or stage.env) to build
the Setings configuration instance else, we will use the local .env file

NB : if the environment variable ENV_FILE doesn't existe and the .env
file doesn't exist, the Setting class will check if the var env exist in
the local context
"""

import os
import torch
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """
    Settings class to configure global variables from .env

    BaseSettings ([type]): pydantic's BaseSettings class allows pydantic
    to be used in both a "validate this request data" context and in a
    "load my system settings" context.

    """

    MODELS_DIR: Optional[str] = Field(env="MODELS_DIR")
    DATA_DIR: Optional[str] = Field(env="DATA_DIR")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


settings = Settings(
    _env_file=os.environ.get("ENV_FILE") or ".env", _env_file_encoding="utf-8"
)
