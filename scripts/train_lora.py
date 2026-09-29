#!/usr/bin/env python3
"""Fine-tune an LLM with QLoRA + LoRA via Unsloth, driven by a YAML config."""

import argparse
import json
from pathlib import Path

import yaml

DEFAULTS = {
    "client": "client",
    "model_name": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "dataset_path": None,
    "val_path": None,
    "output_dir": "output",
    "max_seq_length": 1024,
    "lora_rank": 16,
    "lora_alpha": None,
    "lora_dropout": 0,
    "epochs": 3,
    "batch_size": 1,
    "grad_accum": 8,
    "learning_rate": 0.0002,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 3,
    "seed": 42,
    "report_to": "tensorboard",
    "resume_from_checkpoint": True,
}


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = {**DEFAULTS, **raw}
    if not cfg.get("dataset_path"):
        raise ValueError("config must set dataset_path")
    if not cfg.get("lora_alpha"):
        cfg["lora_alpha"] = cfg["lora_rank"] * 2
    return cfg


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def format_chat(messages, tokenizer):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _load_text_dataset(path, tokenizer):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(path), split="train")
    return dataset.map(
        lambda example: {"text": format_chat(example["messages"], tokenizer)},
        remove_columns=dataset.column_names,
    )


def _last_checkpoint(output_dir):
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def train(cfg, dry_run=False):
    import time

    import unsloth
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        use_gradient_checkpointing=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing=True,
        random_state=cfg["seed"],
    )

    train_dataset = _load_text_dataset(cfg["dataset_path"], tokenizer)
    eval_dataset = None
    if cfg["val_path"] and Path(cfg["val_path"]).exists():
        eval_dataset = _load_text_dataset(cfg["val_path"], tokenizer)

    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        fp16 = not bf16
    else:
        bf16 = False
        fp16 = False
    args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        max_steps=10 if dry_run else -1,
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=1 if dry_run else cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        report_to=cfg["report_to"],
        bf16=bf16,
        fp16=fp16,
        max_length=cfg["max_seq_length"],
        dataset_text_field="text",
        gradient_checkpointing=True,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg["save_steps"],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    resume_path = _last_checkpoint(cfg["output_dir"]) if cfg["resume_from_checkpoint"] else None
    started = time.time()
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_path) if resume_path else None
    )
    duration = time.time() - started

    adapter_dir = Path(cfg["output_dir"]) / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    summary = {
        "client": cfg["client"],
        "model_name": cfg["model_name"],
        "dataset_path": str(cfg["dataset_path"]),
        "adapter_dir": str(adapter_dir),
        "dry_run": dry_run,
        "steps": train_result.global_step,
        "final_loss": train_result.training_loss,
        "duration_seconds": round(duration, 1),
        "seconds_per_step": round(duration / max(train_result.global_step, 1), 2),
        "gpu_hours": round(duration / 3600, 4),
        "bf16": bf16,
        "config": cfg,
    }
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output_dir"]) / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter from a YAML config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = train(cfg, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
