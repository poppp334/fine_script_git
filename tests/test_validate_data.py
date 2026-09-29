import json
import subprocess
import sys
from pathlib import Path

import validate_data

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def write_jsonl(path, entries):
    text = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)
    path.write_text(text + "\n", encoding="utf-8")


def clean_entry():
    return {
        "messages": [
            {"role": "user", "content": "สวัสดี"},
            {"role": "assistant", "content": "สวัสดีครับ"},
        ]
    }


def test_clean_file_passes(tmp_path):
    src = tmp_path / "d.jsonl"
    write_jsonl(src, [clean_entry()])

    report = validate_data.validate(src)

    assert report["ok"] is True
    assert report["n"] == 1
    assert report["issues"] == []


def test_empty_content_is_error(tmp_path):
    src = tmp_path / "d.jsonl"
    entry = clean_entry()
    entry["messages"][0]["content"] = ""
    write_jsonl(src, [entry])

    report = validate_data.validate(src)

    assert report["ok"] is False
    assert any(
        issue["type"] == "empty_content" and issue["severity"] == "error"
        for issue in report["issues"]
    )


def test_duplicate_entry_is_warning_not_error(tmp_path):
    src = tmp_path / "d.jsonl"
    write_jsonl(src, [clean_entry(), clean_entry()])

    report = validate_data.validate(src)

    assert report["ok"] is True
    assert any(
        issue["type"] == "duplicate" and issue["severity"] == "warning"
        for issue in report["issues"]
    )


def test_parse_error_is_reported(tmp_path):
    src = tmp_path / "d.jsonl"
    src.write_text('{"messages": [{"role": "user", "content": "ok"}]}\nnot-json\n', encoding="utf-8")

    report = validate_data.validate(src)

    assert report["ok"] is False
    assert any(issue["type"] == "parse_error" for issue in report["issues"])


def test_missing_messages_is_error(tmp_path):
    src = tmp_path / "d.jsonl"
    write_jsonl(src, [{"foo": "bar"}])

    report = validate_data.validate(src)

    assert report["ok"] is False
    assert any(issue["type"] == "missing_messages" for issue in report["issues"])


def test_invalid_role_is_error(tmp_path):
    src = tmp_path / "d.jsonl"
    entry = {
        "messages": [
            {"role": "robot", "content": "beep"},
            {"role": "assistant", "content": "a"},
        ]
    }
    write_jsonl(src, [entry])

    report = validate_data.validate(src)

    assert report["ok"] is False
    assert any(issue["type"] == "invalid_role" for issue in report["issues"])


def test_non_dict_message_is_reported_not_crashed(tmp_path):
    src = tmp_path / "d.jsonl"
    entry = {"messages": ["not-a-dict", {"role": "assistant", "content": "a"}]}
    write_jsonl(src, [entry])

    report = validate_data.validate(src)

    assert report["ok"] is False
    assert any(issue["type"] == "invalid_message" for issue in report["issues"])


def test_too_long_content_is_warning(tmp_path):
    src = tmp_path / "d.jsonl"
    entry = clean_entry()
    entry["messages"][1]["content"] = "x" * 50
    write_jsonl(src, [entry])

    report = validate_data.validate(src, max_chars=10)

    assert report["ok"] is True
    assert any(
        issue["type"] == "too_long" and issue["severity"] == "warning"
        for issue in report["issues"]
    )


def test_cli_clean_file_exits_zero(tmp_path):
    src = tmp_path / "ok.jsonl"
    write_jsonl(src, [clean_entry()])

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_data.py"), "--data", str(src)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cli_error_file_exits_one_and_writes_report(tmp_path):
    src = tmp_path / "bad.jsonl"
    entry = clean_entry()
    entry["messages"][0]["content"] = ""
    write_jsonl(src, [entry])
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_data.py"),
            "--data",
            str(src),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["ok"] is False
