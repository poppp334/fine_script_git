import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import prepare_data

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def read_jsonl(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_csv_rows_become_chat_messages(tmp_path):
    src = tmp_path / "in.csv"
    write_csv(src, [{"question": "สวัสดี", "answer": "สวัสดีครับ"}])
    out = tmp_path / "out.jsonl"

    prepare_data.convert(src, out)

    assert read_jsonl(out) == [
        {
            "messages": [
                {"role": "user", "content": "สวัสดี"},
                {"role": "assistant", "content": "สวัสดีครับ"},
            ]
        }
    ]


def test_system_prompt_added_when_provided(tmp_path):
    src = tmp_path / "in.csv"
    write_csv(src, [{"question": "q", "answer": "a"}])
    out = tmp_path / "out.jsonl"

    prepare_data.convert(src, out, system_prompt="เป็นผู้ช่วยที่สุภาพ")

    entry = read_jsonl(out)[0]
    assert entry["messages"][0] == {"role": "system", "content": "เป็นผู้ช่วยที่สุภาพ"}
    assert entry["messages"][1]["role"] == "user"
    assert entry["messages"][2]["role"] == "assistant"


def test_no_system_message_when_prompt_empty(tmp_path):
    src = tmp_path / "in.csv"
    write_csv(src, [{"question": "q", "answer": "a"}])
    out = tmp_path / "out.jsonl"

    prepare_data.convert(src, out)

    roles = [m["role"] for m in read_jsonl(out)[0]["messages"]]
    assert roles == ["user", "assistant"]


def test_duplicate_rows_removed(tmp_path):
    src = tmp_path / "in.csv"
    write_csv(
        src,
        [
            {"question": "q1", "answer": "a1"},
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ],
    )
    out = tmp_path / "out.jsonl"

    stats = prepare_data.convert(src, out)

    assert len(read_jsonl(out)) == 2
    assert stats["n_duplicates_removed"] == 1


def test_rows_with_missing_answer_are_skipped(tmp_path):
    src = tmp_path / "in.csv"
    write_csv(
        src,
        [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": ""},
            {"question": "", "answer": "a3"},
        ],
    )
    out = tmp_path / "out.jsonl"

    stats = prepare_data.convert(src, out)

    assert len(read_jsonl(out)) == 1
    assert stats["n_skipped"] == 2


def test_jsonl_chat_input_passes_through(tmp_path):
    entry = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    }
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"

    prepare_data.convert(src, out)

    assert read_jsonl(out) == [entry]


def test_excel_input_supported(tmp_path):
    src = tmp_path / "in.xlsx"
    pd.DataFrame([{"question": "q", "answer": "a"}]).to_excel(src, index=False)
    out = tmp_path / "out.jsonl"

    prepare_data.convert(src, out)

    assert len(read_jsonl(out)) == 1


def test_val_split_writes_disjoint_files(tmp_path):
    src = tmp_path / "in.csv"
    write_csv(src, [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)])
    train_out = tmp_path / "train.jsonl"
    val_out = tmp_path / "val.jsonl"

    stats = prepare_data.convert(src, train_out, val_output=val_out, val_split=0.2)

    train = read_jsonl(train_out)
    val = read_jsonl(val_out)
    assert len(train) == 8
    assert len(val) == 2
    assert stats["n_train"] == 8
    assert stats["n_val"] == 2
    train_keys = {json.dumps(e, sort_keys=True) for e in train}
    val_keys = {json.dumps(e, sort_keys=True) for e in val}
    assert train_keys.isdisjoint(val_keys)


def test_unsupported_extension_raises(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        prepare_data.convert(src, tmp_path / "out.jsonl")


def test_cli_prints_stats_and_exits_zero(tmp_path):
    src = tmp_path / "in.csv"
    write_csv(src, [{"question": "q", "answer": "a"}])
    out = tmp_path / "out.jsonl"

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "prepare_data.py"), "--input", str(src), "--output", str(out)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["n_entries"] == 1
