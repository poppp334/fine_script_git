#!/usr/bin/env python3
"""End-to-end smoke test: train 10 steps on a tiny model, save, reload, generate."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent

DUMMY_DATA = [
    {"messages": [{"role": "user", "content": "สวัสดี"}, {"role": "assistant", "content": "สวัสดีครับ มีอะไรให้ช่วยไหมครับ"}]},
    {"messages": [{"role": "user", "content": "ร้านเปิดกี่โมง"}, {"role": "assistant", "content": "ร้านเปิด 9 โมงเช้าถึง 2 ทุ่มครับ"}]},
    {"messages": [{"role": "user", "content": "ค่าส่งเท่าไหร่"}, {"role": "assistant", "content": "ค่าส่งเริ่มต้น 50 บาทครับ"}]},
    {"messages": [{"role": "user", "content": "ขอบคุณ"}, {"role": "assistant", "content": "ยินดีครับ"}]},
    {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hello! How can I help you today?"}]},
    {"messages": [{"role": "user", "content": "what are your hours"}, {"role": "assistant", "content": "We open from 9am to 8pm every day."}]},
    {"messages": [{"role": "user", "content": "สะดวกคุยวันไหน"}, {"role": "assistant", "content": "สะดวกวันจันทร์ถึงศุกร์ครับ"}]},
    {"messages": [{"role": "user", "content": "ติดต่อยังไง"}, {"role": "assistant", "content": "ติดต่อได้ที่อีเมล support@example.com ครับ"}]},
]


def main():
    parser = argparse.ArgumentParser(description="Smoke test: tiny train on local GPU")
    parser.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
    args = parser.parse_args()

    started = time.time()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("SMOKE FAIL: torch.cuda.is_available() is False")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    work = Path(tempfile.mkdtemp(prefix="smoke_"))
    data_path = work / "dummy.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in DUMMY_DATA) + "\n",
        encoding="utf-8",
    )
    config_path = work / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_name": args.model,
                "dataset_path": str(data_path),
                "output_dir": str(work / "output"),
                "max_seq_length": 512,
                "lora_rank": 8,
                "epochs": 1,
                "batch_size": 1,
                "grad_accum": 1,
                "logging_steps": 1,
                "save_steps": 5,
                "save_total_limit": 1,
                "report_to": "none",
                "resume_from_checkpoint": False,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "train_lora.py"), "--config", str(config_path), "--dry-run"],
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit("SMOKE FAIL: training step returned non-zero")

    adapter_dir = work / "output" / "lora_adapter"
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise SystemExit(f"SMOKE FAIL: no adapter saved at {adapter_dir}")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=512,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    messages = [{"role": "user", "content": "สวัสดี"}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=32)
    text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    if not text:
        raise SystemExit("SMOKE FAIL: reloaded model generated empty output")

    print(f"Sample output: {text}")
    print(f"SMOKE PASS in {time.time() - started:.0f}s | workdir: {work}")


if __name__ == "__main__":
    main()
