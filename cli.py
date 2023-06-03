#!/usr/bin/env python3

import argparse
import re
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def operation(
    *,
    model_name: str,
) -> None:
    max_new_tokens = 256
    temperature = 0.9
    max_history = 16
    speaker_name_user = "ユーザー"
    speaker_name_system = "システム"

    initial_messages = [
        #         {"speaker": speaker_name_user, "text": "こんにちは。"},
        {
            "speaker": speaker_name_system,
            #             "text": f"こんにちは、私は{speaker_name_system}です。あなたの質問に適切な回答をします。どのようなご用件ですか？",
            "text": "あなたの質問に適切な回答をします。どのようなご用件ですか？",
        },
    ]
    messages = []
    messages.extend(initial_messages)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #         device_map="auto",
        device_map={"": 0},
        #         load_in_4bit=True,
        load_in_8bit=True,
    )

    output = None
    while True:
        input_text = input(f"{speaker_name_user}: ")

        # コマンドの処理
        if input_text == "clear":
            messages = initial_messages[:]
            output = None
            print("履歴が初期化されました")
            continue
        elif input_text == "exit":
            break
        elif input_text == "retry":
            pass
        else:
            if output:
                messages.append({"speaker": speaker_name_system, "text": output})
            messages.append({"speaker": speaker_name_user, "text": input_text})
            if len(messages) > max_history:
                del messages[0 : (len(messages) - max_history)]

        # プロンプトの作成
        prompt = [f"{uttr['speaker']}: {uttr['text']}" for uttr in messages]
        prompt = "<NL>".join(prompt)
        prompt = prompt + "<NL>" + f"{speaker_name_system}: "

        # テキスト生成の開始
        token_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

        generation_args = [token_ids.to(model.device)]
        generation_kwargs = dict(
            streamer=streamer,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

        thread = Thread(
            target=model.generate, args=generation_args, kwargs=generation_kwargs
        )
        thread.start()

        # TextIteratorStreamer を使った生成結果の受け取り
        print(f"{speaker_name_system}: ", end="")
        generated_text = ""
        for next_text in streamer:
            if not next_text:
                continue
            print(next_text.replace("<NL>", "\n"), end="", flush=True)
            generated_text += next_text
        print("")

        # 生成結果
        output = re.sub("</s>$", "", generated_text)


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument(
        "--model",
        "-m",
        default="rinna/japanese-gpt-neox-3.6b-instruction-ppo",
    )
    return oparser.parse_args()


def main() -> None:
    assert torch.cuda.is_available()
    opts = get_opts()
    operation(
        model_name=opts.model,
    )


if __name__ == "__main__":
    main()
