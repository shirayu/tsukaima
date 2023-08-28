#!/usr/bin/env python3
from threading import Thread
from typing import Final, Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from tsukaima.schema.openai import ChatCompletionRequest, ChatMessage
from tsukaima.schema.schema import Config, ConfigModel


class Model:
    supported_config_version: Final[int] = 2
    rinna_speaker_name_system: Final[str] = "システム"

    @staticmethod
    def get_rinna_prompt(
        *,
        messages: list[ChatMessage],
    ) -> str:
        prompt: str = ""
        for uttr in messages:
            if uttr.role == "system":
                continue
            my_role: str = {
                "user": "ユーザー",
                "assistant": Model.rinna_speaker_name_system,
            }[uttr.role]
            prompt += f"{my_role}: {uttr.content}<NL>"
        prompt += f"{Model.rinna_speaker_name_system}: "
        return prompt

    @staticmethod
    def get_prompt(
        *,
        model_name: str,
        messages: list[ChatMessage],
    ) -> str:
        if model_name.startswith("rinna/"):
            return Model.get_rinna_prompt(
                messages=messages,
            )
        raise KeyError

    def __init__(self, *, config: Config):
        assert (
            config.version == Model.supported_config_version
        ), f"Unsupported config version: {config.version}"

        self.name2model = {}
        self.name2tokenizer = {}
        self.name2config_model = {}

        for config_model in config.models:
            if not config_model.enabled:
                continue
            tokenizer = AutoTokenizer.from_pretrained(
                config_model.path,
                use_fast=False,
                **config_model.tokenizer_kwargs,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config_model.path,
                **config_model.model_kwargs,
            )
            for _alt_name in config_model.names:
                self.name2model[_alt_name] = model
                self.name2tokenizer[_alt_name] = tokenizer
                self.name2config_model[_alt_name] = config_model

    def generate(
        self,
        *,
        request: ChatCompletionRequest,
    ) -> Iterator[str]:
        model_name: str = request.model
        tokenizer = self.name2tokenizer[model_name]
        model = self.name2model[model_name]
        config_model: ConfigModel = self.name2config_model[model_name]

        messages = []
        assert isinstance(request.messages, list)
        for _msg in request.messages:
            messages.append(ChatMessage.parse_obj(_msg))
        prompt: str = Model.get_prompt(
            model_name=config_model.path,
            messages=messages,
        )

        token_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

        generation_args = [token_ids.to(model.device)]
        generation_kwargs = dict(
            streamer=streamer,
            do_sample=True,
            max_new_tokens=config_model.forced_parameters.get(
                "max_new_tokens", 256  # FIXME
            ),
            temperature=config_model.forced_parameters.get(
                "temperature", request.temperature
            ),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=config_model.forced_parameters.get(
                "repetition_penalty", request.presence_penalty
            ),
        )

        thread = Thread(
            target=model.generate,
            args=generation_args,
            kwargs=generation_kwargs,
        )
        thread.start()

        for next_text in streamer:
            if not next_text:
                continue
            t: str = next_text.replace("<NL>", "\n")
            t = t.rstrip("</s>")  # TODO: make this more general
            yield t
