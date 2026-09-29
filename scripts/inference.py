#!/usr/bin/env python3
"""Run inference against a Hugging Face chat model via CLI."""

import argparse

import pandas as pd


def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto")
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt,
    system_prompt="",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):
    import torch

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**kwargs)
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference against a chat model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-csv")
    parser.add_argument("--out-csv")
    parser.add_argument("--system", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if not args.prompt and not args.prompts_csv:
        parser.error("provide --prompt or --prompts-csv")

    model, tokenizer = load_model(args.model)

    if args.prompt:
        answer = generate(
            model,
            tokenizer,
            args.prompt,
            args.system,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        print(answer)
        return

    prompts_df = pd.read_csv(args.prompts_csv)
    responses = [
        generate(
            model,
            tokenizer,
            str(prompt),
            args.system,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        for prompt in prompts_df["prompt"]
    ]
    result_df = pd.DataFrame({"prompt": prompts_df["prompt"], "response": responses})
    result_df.to_csv(args.out_csv, index=False)
    print(f"wrote {len(result_df)} responses to {args.out_csv}")


if __name__ == "__main__":
    main()
