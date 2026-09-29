import gc
import subprocess

import pytest
import torch

import export_gguf

MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.mark.slow
def test_export_gguf_and_ollama_answers(tmp_path):
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

    gguf_dir = tmp_path / "gguf"
    artifacts = export_gguf.export(adapter_dir, gguf_dir, quantization="q4_k_m")

    assert artifacts["gguf"].exists()
    assert artifacts["gguf"].stat().st_size > 0
    assert artifacts["modelfile"].exists()

    model_name = "smoke-qwen-test"
    try:
        create = subprocess.run(
            ["ollama", "create", model_name, "-f", str(artifacts["modelfile"])],
            capture_output=True,
            text=True,
            cwd=str(artifacts["gguf"].parent),
            timeout=600,
        )
        assert create.returncode == 0, create.stderr[-2000:]

        run = subprocess.run(
            ["ollama", "run", model_name, "สวัสดี"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert run.returncode == 0, run.stderr[-2000:]
        assert run.stdout.strip()
    finally:
        subprocess.run(["ollama", "rm", model_name], capture_output=True)
