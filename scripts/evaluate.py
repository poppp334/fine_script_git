#!/usr/bin/env python3
"""Evaluate a fine-tuned model against its base: perplexity + side-by-side report."""

import html
import json
import math
from pathlib import Path

import pandas as pd


def perplexity_from_nll(total_nll, n_tokens):
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    return math.exp(total_nll / n_tokens)


def summarize(pairs, base_ppl, ft_ppl):
    scored = [
        pair
        for pair in pairs
        if pair.get("score_base") is not None and pair.get("score_ft") is not None
    ]
    improved_pct = None
    if scored:
        wins = sum(1 for pair in scored if pair["score_ft"] > pair["score_base"])
        improved_pct = round(100 * wins / len(scored), 1)
    return {
        "n_prompts": len(pairs),
        "n_scored": len(scored),
        "improved_pct": improved_pct,
        "base_perplexity": base_ppl,
        "finetuned_perplexity": ft_ppl,
        "perplexity_improved": ft_ppl < base_ppl,
    }


def load_prompts(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        return [str(value) for value in frame["prompt"].tolist()]
    if suffix == ".jsonl":
        prompts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            messages = json.loads(line).get("messages", [])
            user_contents = [m["content"] for m in messages if m.get("role") == "user"]
            if user_contents:
                prompts.append(user_contents[-1])
        return prompts
    raise ValueError(f"unsupported prompts format: {suffix}")


def write_pairs_csv(pairs, path):
    rows = []
    for index, pair in enumerate(pairs, start=1):
        rows.append(
            {
                "id": pair.get("id", index),
                "prompt": pair.get("prompt", ""),
                "base_response": pair.get("base_response", ""),
                "ft_response": pair.get("ft_response", ""),
                "score_base": pair.get("score_base"),
                "score_ft": pair.get("score_ft"),
            }
        )
    columns = ["id", "prompt", "base_response", "ft_response", "score_base", "score_ft"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def merge_scores(pairs, scores_path):
    frame = pd.read_csv(scores_path)
    scores = {
        int(row["id"]): (row.get("score_base"), row.get("score_ft"))
        for _, row in frame.iterrows()
    }
    merged = []
    for index, pair in enumerate(pairs, start=1):
        entry = dict(pair)
        pair_id = entry.get("id", index)
        if pair_id in scores:
            base, ft = scores[pair_id]
            entry["score_base"] = None if pd.isna(base) else float(base)
            entry["score_ft"] = None if pd.isna(ft) else float(ft)
        merged.append(entry)
    return merged


def _card(pair):
    def esc(value):
        return html.escape(str(value))

    scores = ""
    if pair.get("score_base") is not None and pair.get("score_ft") is not None:
        scores = (
            f'<div class="scores">คะแนน: base {esc(pair["score_base"])}'
            f" → fine-tuned {esc(pair['score_ft'])}</div>"
        )
    return f"""<div class="card">
  <div class="prompt">{esc(pair.get("prompt", ""))}</div>
  <div class="answers">
    <div class="answer base"><div class="label">Base</div><div class="text">{esc(pair.get("base_response", ""))}</div></div>
    <div class="answer ft"><div class="label">Fine-tuned</div><div class="text">{esc(pair.get("ft_response", ""))}</div></div>
  </div>
  {scores}
</div>"""


def render_html(pairs, metrics, regression_pairs):
    def esc(value):
        return html.escape(str(value))

    improved = ""
    if metrics.get("improved_pct") is not None:
        improved = (
            f'<div class="metric"><div class="value">{esc(metrics["improved_pct"])}%</div>'
            f'<div class="name">prompt ที่ดีขึ้น ({esc(metrics.get("n_scored", 0))} ข้อที่ให้คะแนน)</div></div>'
        )

    cards = "".join(_card(pair) for pair in pairs)
    regression_section = ""
    if regression_pairs:
        regression_cards = "".join(_card(pair) for pair in regression_pairs)
        regression_section = (
            "<h2>Regression check — กันลืมความรู้เดิม</h2>" + regression_cards
        )

    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>Fine-tuning evaluation report</title>
<style>
  body {{ font-family: "Segoe UI", "Noto Sans Thai", sans-serif; background: #f6f7f9; color: #1c2330; margin: 0; padding: 32px; }}
  h1 {{ margin: 0 0 4px; }}
  .sub {{ color: #66707f; margin-bottom: 24px; }}
  .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
  .metric {{ background: #fff; border: 1px solid #e3e7ee; border-radius: 10px; padding: 14px 18px; min-width: 150px; }}
  .metric .value {{ font-size: 24px; font-weight: 700; }}
  .metric .name {{ color: #66707f; font-size: 13px; margin-top: 4px; }}
  .card {{ background: #fff; border: 1px solid #e3e7ee; border-radius: 12px; padding: 18px; margin-bottom: 16px; }}
  .prompt {{ font-weight: 600; margin-bottom: 12px; }}
  .answers {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .answer {{ border-radius: 8px; padding: 12px; background: #fafbfc; }}
  .answer.ft {{ background: #eef7ee; }}
  .label {{ font-size: 12px; color: #66707f; text-transform: uppercase; margin-bottom: 6px; }}
  .text {{ white-space: pre-wrap; }}
  .scores {{ margin-top: 10px; font-size: 13px; color: #33507a; }}
  h2 {{ margin: 32px 0 12px; }}
</style>
</head>
<body>
<h1>Fine-tuning evaluation report</h1>
<div class="sub">ก่อน vs หลัง fine-tune — side-by-side</div>
<div class="metrics">
  <div class="metric"><div class="value">{metrics["base_perplexity"]:.2f}</div><div class="name">Perplexity — base (ต่ำกว่าดี)</div></div>
  <div class="metric"><div class="value">{metrics["finetuned_perplexity"]:.2f}</div><div class="name">Perplexity — fine-tuned</div></div>
  {improved}
</div>
{cards}
{regression_section}
</body>
</html>
"""


def load_test_messages(path):
    message_lists = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("messages"):
            message_lists.append(entry["messages"])
    return message_lists


def compute_perplexity(model, tokenizer, message_lists, max_length=512):
    import torch

    total_nll = 0.0
    total_tokens = 0
    for messages in message_lists:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(model.device)
        n_tokens = input_ids.shape[1] - 1
        if n_tokens <= 0:
            continue
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=input_ids)
        total_nll += float(output.loss) * n_tokens
        total_tokens += n_tokens
    return perplexity_from_nll(total_nll, total_tokens)


def generate_responses(model, tokenizer, prompts, max_new_tokens=128):
    import torch

    responses = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses.append(
            tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        )
    return responses


def run_eval(
    base_model,
    adapter_dir,
    test_file,
    prompts_file,
    out_dir,
    regression_file=None,
    scores_file=None,
    max_length=512,
    max_new_tokens=128,
):
    import gc

    import torch
    import unsloth
    from unsloth import FastLanguageModel

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_messages = load_test_messages(test_file)
    prompts = load_prompts(prompts_file)
    regression_prompts = load_prompts(regression_file) if regression_file else []

    def run_model(model_name):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_name), max_seq_length=max_length, load_in_4bit=True
        )
        ppl = compute_perplexity(model, tokenizer, test_messages, max_length=max_length)
        FastLanguageModel.for_inference(model)
        responses = generate_responses(model, tokenizer, prompts, max_new_tokens)
        regressions = (
            generate_responses(model, tokenizer, regression_prompts, max_new_tokens)
            if regression_prompts
            else []
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return ppl, responses, regressions

    base_ppl, base_responses, base_regressions = run_model(base_model)
    ft_ppl, ft_responses, ft_regressions = run_model(adapter_dir)

    pairs = [
        {
            "prompt": prompt,
            "base_response": base,
            "ft_response": ft,
            "score_base": None,
            "score_ft": None,
        }
        for prompt, base, ft in zip(prompts, base_responses, ft_responses)
    ]
    regression_pairs = [
        {
            "prompt": prompt,
            "base_response": base,
            "ft_response": ft,
            "score_base": None,
            "score_ft": None,
        }
        for prompt, base, ft in zip(regression_prompts, base_regressions, ft_regressions)
    ]
    if scores_file:
        pairs = merge_scores(pairs, scores_file)

    metrics = summarize(pairs, base_ppl, ft_ppl)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_pairs_csv(pairs, out_dir / "eval_pairs.csv")
    (out_dir / "eval_report.html").write_text(
        render_html(pairs, metrics, regression_pairs), encoding="utf-8"
    )
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model vs its base")
    parser.add_argument("--config")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter")
    parser.add_argument("--test-file")
    parser.add_argument("--prompts-file")
    parser.add_argument("--regression-file")
    parser.add_argument("--scores")
    parser.add_argument("--out-dir")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.config:
        from train_lora import load_config

        cfg = load_config(args.config)
        base_model = args.base_model or cfg["model_name"]
        adapter = args.adapter or str(Path(cfg["output_dir"]) / "lora_adapter")
        test_file = args.test_file or cfg["val_path"] or cfg["dataset_path"]
        out_dir = args.out_dir or str(Path(cfg["output_dir"]) / "eval")
    else:
        base_model = args.base_model
        adapter = args.adapter
        test_file = args.test_file
        out_dir = args.out_dir

    missing = [
        name
        for name, value in (
            ("--base-model", base_model),
            ("--adapter", adapter),
            ("--test-file", test_file),
            ("--prompts-file", args.prompts_file),
            ("--out-dir", out_dir),
        )
        if not value
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    metrics = run_eval(
        base_model=base_model,
        adapter_dir=adapter,
        test_file=test_file,
        prompts_file=args.prompts_file,
        out_dir=out_dir,
        regression_file=args.regression_file,
        scores_file=args.scores,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
