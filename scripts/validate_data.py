#!/usr/bin/env python3
"""Validate ChatML JSONL datasets before training."""

import argparse
import json
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}


def validate(path, max_chars=8000):
    issues = []
    n = 0
    seen = set()
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        n += 1
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            issues.append(
                {"index": index, "type": "parse_error", "severity": "error", "detail": str(error)}
            )
            continue
        messages = entry.get("messages")
        if not isinstance(messages, list) or not messages:
            issues.append(
                {
                    "index": index,
                    "type": "missing_messages",
                    "severity": "error",
                    "detail": "entry has no non-empty messages list",
                }
            )
            continue
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if key in seen:
            issues.append(
                {
                    "index": index,
                    "type": "duplicate",
                    "severity": "warning",
                    "detail": "identical to an earlier entry",
                }
            )
        seen.add(key)
        for message in messages:
            if not isinstance(message, dict):
                issues.append(
                    {
                        "index": index,
                        "type": "invalid_message",
                        "severity": "error",
                        "detail": f"message is {type(message).__name__}, expected an object",
                    }
                )
                continue
            role = message.get("role")
            if role not in VALID_ROLES:
                issues.append(
                    {
                        "index": index,
                        "type": "invalid_role",
                        "severity": "error",
                        "detail": f"unknown role '{role}'",
                    }
                )
            content = str(message.get("content", ""))
            if not content.strip():
                issues.append(
                    {
                        "index": index,
                        "type": "empty_content",
                        "severity": "error",
                        "detail": f"empty content in role '{role}'",
                    }
                )
            if len(content) > max_chars:
                issues.append(
                    {
                        "index": index,
                        "type": "too_long",
                        "severity": "warning",
                        "detail": f"{len(content)} chars > max_chars {max_chars} in role '{role}'",
                    }
                )
    n_errors = sum(1 for issue in issues if issue["severity"] == "error")
    return {"ok": n_errors == 0, "n": n, "n_errors": n_errors, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description="Validate ChatML JSONL datasets")
    parser.add_argument("--data", required=True)
    parser.add_argument("--report")
    parser.add_argument("--max-chars", type=int, default=8000)
    args = parser.parse_args()

    report = validate(args.data, max_chars=args.max_chars)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
