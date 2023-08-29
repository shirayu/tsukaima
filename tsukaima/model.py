#!/usr/bin/env python3
from threading import Thread
from typing import Final, Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.models.mega.modeling_mega import MegaClassificationHead

from tsukaima.schema.openai import ChatCompletionRequest, ChatMessage
from tsukaima.schema.schema import Config, ConfigModel


class Model:
    supported_config_version: Final[int] = 2

    @staticmethod
    def get_rinna_prompt(
        *,
        messages: list[ChatMessage],
    ) -> str:
        rinna_speaker_name_system: Final[str] = "システム"
        prompt: str = ""
        for uttr in messages:
            if uttr.role == "system":
                continue
            my_role: str = {
                "user": "ユーザー",
                "assistant": rinna_speaker_name_system,
            }[uttr.role]
            prompt += f"{my_role}: {uttr.content}<NL>"
        prompt += f"{rinna_speaker_name_system}: "
        return prompt

    @staticmethod
    def get_line_prompt(
        *,
        messages: list[ChatMessage],
    ) -> str:
        line_speaker_name_system: Final[str] = "システム"
        prompt: str = ""
        for uttr in messages:
            if uttr.role == "system":
                continue
            my_role: str = {
                "user": "ユーザー",
                "assistant": line_speaker_name_system,
            }[uttr.role]
            prompt += f"{my_role}: {uttr.content}\n"
        prompt += f"{line_speaker_name_system}: "
        return prompt

    @staticmethod
    def get_elyza_prompt(
        *,
        messages: list[ChatMessage],
    ) -> str:
        prev_role: str = "system"
        user_contents: list[str] = []
        assistant_contents: list[str] = []

        for uttr in messages:
            if uttr.role == "system":
                pass
            elif uttr.role == "user":
                if prev_role == "user":
                    user_contents[-1] += f"\n{uttr.content}"
                elif prev_role in {"assistant", "system"}:
                    user_contents.append(uttr.content)
                else:
                    raise NotImplementedError(f"Unsupported prev_role: {prev_role}")

            elif uttr.role == "assistant":
                if prev_role in {"user", "system"}:
                    assistant_contents.append(uttr.content)
                elif prev_role == "assistant":
                    assistant_contents[-1] += f"\n{uttr.content}"
                else:
                    raise NotImplementedError(f"Unsupported prev_role: {prev_role}")

            else:
                raise KeyError(uttr.role)
            prev_role = uttr.role

        bos_token: Final[str] = "<s>"
        eos_token: Final[str] = "</s>"
        default_system_prompt: Final[str] = "あなたは誠実で優秀な日本人のアシスタントです。"
        prompt: str = (
            f"{bos_token}[INST] <<SYS>>\n{default_system_prompt}\n<</SYS>>\n\n"
        )
        assert len(user_contents) == len(assistant_contents) + 1
        for user_input, assistant_resp in zip(user_contents, assistant_contents):
            prompt += f"{user_input} [/INST] {assistant_resp.strip()} {eos_token}{bos_token}[INST] "

        prompt += f"{user_contents[-1]} [/INST]"
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
        elif model_name.startswith("line-corporation/"):
            return Model.get_line_prompt(
                messages=messages,
            )
        elif model_name.startswith("elyza/"):
            return Model.get_elyza_prompt(
                messages=messages,
            )

        raise KeyError(f"Unsupported model: {model_name}")

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
                "repetition_penalty", request.frequency_penalty
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
