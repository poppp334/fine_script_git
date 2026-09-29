#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model for delivery."""

import argparse
from pathlib import Path


def merge(adapter_dir, output_dir, save_method="merged_16bit", max_seq_length=1024):
    import unsloth
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=False,
    )
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method=save_method)
    return Path(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a standalone model")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="merged_16bit", choices=["merged_16bit", "merged_4bit"])
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    merged_dir = merge(
        args.adapter,
        args.output,
        save_method=args.method,
        max_seq_length=args.max_seq_length,
    )
    print(f"merged model written to {merged_dir}")


if __name__ == "__main__":
    main()
