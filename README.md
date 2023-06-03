
# tsukaima

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://github.com/shirayu/tsukaima/blob/main/LICENSE.txt)
[![CI](https://github.com/shirayu/tsukaima/actions/workflows/ci.yml/badge.svg)](https://github.com/shirayu/tsukaima/actions/workflows/ci.yml)
[![CodeQL](https://github.com/shirayu/tsukaima/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/shirayu/tsukaima/actions/workflows/codeql-analysis.yml)
[![Typos](https://github.com/shirayu/tsukaima/actions/workflows/typos.yml/badge.svg)](https://github.com/shirayu/tsukaima/actions/workflows/typos.yml)

既存のOpenAI ChatGPTクライアントを使って[rinna LLM](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)を呼ぶために作りました．

```bash
pip install -U git+https://github.com/shirayu/tsukaima.git

# 設定ファイルのダウンロード
wget https://github.com/shirayu/tsukaima/config.json

tsukaima --host 0.0.0.0 --port 6006 --config ./config.json    
```

[BetterChatGPT](https://github.com/ztjhz/BetterChatGPT)などのクライアントのAPIエンドポイントに``http://0.0.0.0:6006/v1/chat/completions``を指定してください

![Screenshot](https://github.com/shirayu/tsukaima/assets/963961/0e3ee7db-d570-40f7-867a-6c9ec708588b)

## Tips

- You may need install [NCCL](https://developer.nvidia.com/nccl/nccl-download)
    - ``sudo apt install libnccl-dev libnccl2``

## Reference

- <https://gist.github.com/advanceboy/b9143aa9de23a6f9a60a07a862e0b4a8>
- <https://note.com/hamachi_jp/n/n70b1e48c09a0#816e2412-a3b6-4319-b056-3a7a1bcf7638>
- <https://note.com/hamachi_jp/n/n8e1cbf3314be>
- <https://github.com/lm-sys/FastChat>
- <https://gist.github.com/advanceboy/b9143aa9de23a6f9a60a07a862e0b4a8>
