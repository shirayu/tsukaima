#!/usr/bin/env python3
from typing import Any

from pydantic import BaseModel


class ConfigModel(BaseModel):
    path: str
    names: list[str]
    forced_parameters: dict[str, Any]
    model_kwargs: dict[str, Any]
    tokenizer_kwargs: dict[str, Any]


class Config(BaseModel):
    version: int
    models: list[ConfigModel]
