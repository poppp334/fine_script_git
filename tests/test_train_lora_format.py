import pytest
from transformers import AutoTokenizer

import train_lora

pytestmark = pytest.mark.slow

TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def test_format_chat_uses_chatml_template():
    tokenizer = load_tokenizer()

    text = train_lora.format_chat(
        [
            {"role": "user", "content": "สวัสดี"},
            {"role": "assistant", "content": "สวัสดีครับ"},
        ],
        tokenizer,
    )

    assert "<|im_start|>user\nสวัสดี<|im_end|>" in text
    assert "<|im_start|>assistant\nสวัสดีครับ<|im_end|>" in text


def test_format_chat_has_no_generation_prompt():
    tokenizer = load_tokenizer()

    text = train_lora.format_chat(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenizer,
    )

    assert not text.rstrip().endswith("assistant")
