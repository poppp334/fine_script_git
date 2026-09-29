import gc
import subprocess
import sys

import pytest
import torch

import merge_lora

MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.mark.slow
def test_merge_produces_plain_transformers_model(tmp_path):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=512, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0, target_modules=TARGET_MODULES
    )
    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    merged_dir = merge_lora.merge(adapter_dir, tmp_path / "merged")

    check_script = """
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

merged = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(merged, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(merged)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "สวัสดี"}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=16)
text = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
).strip()
assert text, "merged model generated empty output"
print("PLAIN_OK: " + text[:80])
"""
    result = subprocess.run(
        [sys.executable, "-c", check_script, str(merged_dir)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "PLAIN_OK:" in result.stdout
