#!/usr/bin/env python3

# This code comes from fastchat/serve/openai_api_server.py licensed under Apache-2.0 license.
# https://github.com/lm-sys/FastChat

import json
from typing import Any, AsyncGenerator, Iterator

import shortuuid

from tsukaima.model import Model
from tsukaima.schema.openai import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)


def chat_completion_stream(
    *,
    model: Model,
    request: ChatCompletionRequest,
) -> Iterator[dict[str, Any]]:
    full_text: str = ""
    for text in model.generate(
        request=request,
    ):
        full_text += text
        yield {
            "error_code": 0,
            "text": full_text,
        }


async def chat_completion_stream_generator(
    *,
    model: Model,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, Any]:
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    for i in range(1):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id,
            choices=[choice_data],
            model=request.model,
        )
        yield f"data: {chunk.json(exclude_unset=True)}\n\n"

        previous_text = ""
        for content in chat_completion_stream(
            model=model,
            request=request,
        ):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=content.get("finish_reason", None),
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=request.model
            )
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"
