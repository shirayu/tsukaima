#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from tsukaima.model import Model
from tsukaima.schema.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from tsukaima.schema.schema import Config
from tsukaima.streamer import chat_completion_stream_generator


def get_app(config: Config):
    model = Model(config=config)

    app = FastAPI(
        title="Tsukaima",
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
    ):
        if req.stream:
            generator = chat_completion_stream_generator(
                model=model,
                request=req,
                raw_request=request,
            )
            return StreamingResponse(generator, media_type="text/event-stream")

        choices = []

        full_text: str = "".join(
            [
                text
                for text in model.generate(
                    request=req,
                )
            ]
        )
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=full_text),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(
            model=req.model,
            choices=choices,
            usage=usage,
        )

    return app


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=6006, type=int)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root_path", default="")

    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")
    return parser.parse_args()


def main():
    opts = get_opts()

    config: Config = Config.parse_file(opts.config)
    app = get_app(config)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=opts.allowed_origins,
        allow_credentials=opts.allow_credentials,
        allow_methods=opts.allowed_methods,
        allow_headers=opts.allowed_headers,
    )

    uvicorn.run(
        app,  # type: ignore
        host=opts.host,
        port=opts.port,
        root_path=opts.root_path,
    )


if __name__ == "__main__":
    main()
