#!/usr/bin/env python3

# This code comes from fastchat/protocol/openai_api_protocol.py licensed under Apache-2.0 license.
# https://github.com/lm-sys/FastChat

import time
from typing import Literal

import shortuuid
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class ChatCompletionRequest(BaseModel):
    model: str
    messages: str | list[dict[str, str]]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    n: int | None = 1
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    user: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] | None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length"] | None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
