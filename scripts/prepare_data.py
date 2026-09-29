#!/usr/bin/env python3
"""Convert client datasets (Excel/CSV/JSONL) into ChatML JSONL."""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _make_entry(question, answer, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    return {"messages": messages}


def _clean(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _read_records(path):
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records
    raise ValueError(f"unsupported input format: {suffix}")


def _write_jsonl(entries, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def convert(input_path, output_path, system_prompt="", val_split=0.0, val_output=None, seed=42):
    records = _read_records(Path(input_path))

    entries = []
    seen = set()
    n_skipped = 0
    n_duplicates_removed = 0
    for record in records:
        if "messages" in record:
            entry = {"messages": record["messages"]}
        else:
            question = _clean(record.get("question"))
            answer = _clean(record.get("answer"))
            if not question or not answer:
                n_skipped += 1
                continue
            entry = _make_entry(question, answer, system_prompt)
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        if key in seen:
            n_duplicates_removed += 1
            continue
        seen.add(key)
        entries.append(entry)

    train_entries = entries
    val_entries = []
    if val_split and val_output:
        shuffled = entries[:]
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_split))
        val_entries = shuffled[:n_val]
        train_entries = shuffled[n_val:]

    _write_jsonl(train_entries, Path(output_path))
    if val_entries:
        _write_jsonl(val_entries, Path(val_output))

    return {
        "n_input": len(records),
        "n_entries": len(entries),
        "n_skipped": n_skipped,
        "n_duplicates_removed": n_duplicates_removed,
        "n_train": len(train_entries),
        "n_val": len(val_entries),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert client data into ChatML JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--val-output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = convert(
        args.input,
        args.output,
        system_prompt=args.system_prompt,
        val_split=args.val_split,
        val_output=args.val_output,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
