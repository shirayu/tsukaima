
# tsukaima

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://github.com/shirayu/tsukaima/blob/main/LICENSE.txt)
[![CI](https://github.com/shirayu/tsukaima/actions/workflows/ci.yml/badge.svg)](https://github.com/shirayu/tsukaima/actions/workflows/ci.yml)
[![CodeQL](https://github.com/shirayu/tsukaima/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/shirayu/tsukaima/actions/workflows/codeql-analysis.yml)
[![Typos](https://github.com/shirayu/tsukaima/actions/workflows/typos.yml/badge.svg)](https://github.com/shirayu/tsukaima/actions/workflows/typos.yml)

Tsukaima is a tool to call local large language models (LLMs) using the existing OpenAI ChatGPT clients.
Currently, the following models are supported.

- [rinna LLM](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)
- [line-corporation/japanese-large-lm-3.6b-instruction-sft](https://huggingface.co/line-corporation/japanese-large-lm-3.6b-instruction-sft)

## How to use

```console
$ python3 -m venv myvenv
$ source myvenv/bin/activate
(myvenv) $ pip install -U git+https://github.com/shirayu/tsukaima.git
(myvenv) $ wget https://raw.githubusercontent.com/shirayu/tsukaima/main/examples_config/rinna.json -O rinna.json
(myvenv) $ tsukaima --host 0.0.0.0 --port 6006 --config ./rinna.json
```

Set API endpoint to the address (Eg: ``http://0.0.0.0:6006/v1/chat/completions``) to use ChatGPT clients such as [BetterChatGPT](https://github.com/ztjhz/BetterChatGPT).

<img src="https://user-images.githubusercontent.com/963961/243087372-3fca7c13-4225-414f-9f72-e438f30bf661.png" alt="Screenshot" width="400">

Check [other config examples](https://github.com/shirayu/tsukaima/tree/main/examples_config).

## Specification

- Messages whose `role` is `system` will be ignored

## Setting file format

Please read [tsukaima.schema.schema](https://github.com/shirayu/tsukaima/blob/main/tsukaima/schema/schema.py)

## Tips

- You may need install [NCCL](https://developer.nvidia.com/nccl/nccl-download)
    - ``sudo apt install libnccl-dev libnccl2``

## Reference

- <https://gist.github.com/advanceboy/b9143aa9de23a6f9a60a07a862e0b4a8>
- <https://note.com/hamachi_jp/n/n70b1e48c09a0#816e2412-a3b6-4319-b056-3a7a1bcf7638>
- <https://note.com/hamachi_jp/n/n8e1cbf3314be>
- <https://github.com/lm-sys/FastChat>
- <https://gist.github.com/advanceboy/b9143aa9de23a6f9a60a07a862e0b4a8>
