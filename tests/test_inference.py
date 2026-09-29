import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "inference.py"), *args],
        capture_output=True,
        text=True,
    )


@pytest.mark.slow
def test_cli_single_prompt():
    result = run_cli("--model", MODEL, "--prompt", "สวัสดี", "--max-new-tokens", "16")

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip()


@pytest.mark.slow
def test_cli_batch_csv_roundtrip(tmp_path):
    prompts_path = tmp_path / "prompts.csv"
    pd.DataFrame({"prompt": ["สวัสดี", "ร้านเปิดกี่โมง"]}).to_csv(prompts_path, index=False)
    out_path = tmp_path / "out.csv"

    result = run_cli(
        "--model",
        MODEL,
        "--prompts-csv",
        str(prompts_path),
        "--out-csv",
        str(out_path),
        "--max-new-tokens",
        "16",
    )

    assert result.returncode == 0, result.stderr[-2000:]
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["prompt", "response"]
    assert len(df) == 2
    assert df["response"].str.strip().str.len().min() > 0
