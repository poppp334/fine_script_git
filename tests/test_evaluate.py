import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import evaluate

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_perplexity_is_exp_of_mean_nll():
    assert evaluate.perplexity_from_nll(0.0, 4) == 1.0
    assert evaluate.perplexity_from_nll(math.log(2) * 2, 2) == pytest.approx(2.0)


def test_perplexity_rejects_zero_tokens():
    with pytest.raises(ValueError):
        evaluate.perplexity_from_nll(1.0, 0)


def test_summarize_counts_prompts_where_finetuned_scores_higher():
    pairs = [
        {"prompt": "a", "score_base": 2, "score_ft": 4},
        {"prompt": "b", "score_base": 4, "score_ft": 3},
        {"prompt": "c", "score_base": 1, "score_ft": 2},
    ]

    metrics = evaluate.summarize(pairs, base_ppl=10.0, ft_ppl=8.0)

    assert metrics["n_prompts"] == 3
    assert metrics["n_scored"] == 3
    assert metrics["improved_pct"] == pytest.approx(66.7)
    assert metrics["base_perplexity"] == 10.0
    assert metrics["finetuned_perplexity"] == 8.0
    assert metrics["perplexity_improved"] is True


def test_summarize_without_scores_has_no_improved_pct():
    pairs = [{"prompt": "a", "score_base": None, "score_ft": None}]

    metrics = evaluate.summarize(pairs, base_ppl=10.0, ft_ppl=8.0)

    assert metrics["improved_pct"] is None
    assert metrics["n_scored"] == 0
    assert metrics["perplexity_improved"] is True


def test_cli_without_required_args_exits_nonzero():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "evaluate.py")],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_load_prompts_from_csv(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("prompt\nสวัสดี\nร้านเปิดกี่โมง\n", encoding="utf-8")

    assert evaluate.load_prompts(path) == ["สวัสดี", "ร้านเปิดกี่โมง"]


def test_load_prompts_from_jsonl_takes_last_user_message(tmp_path):
    entry = {
        "messages": [
            {"role": "user", "content": "คำถามแรก"},
            {"role": "assistant", "content": "ตอบ"},
            {"role": "user", "content": "คำถามสอง"},
        ]
    }
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    assert evaluate.load_prompts(path) == ["คำถามสอง"]


def test_write_pairs_csv_roundtrip(tmp_path):
    pairs = [
        {
            "prompt": "a",
            "base_response": "x",
            "ft_response": "y",
            "score_base": None,
            "score_ft": None,
        }
    ]
    path = tmp_path / "pairs.csv"

    evaluate.write_pairs_csv(pairs, path)

    import pandas as pd

    df = pd.read_csv(path)
    assert list(df.columns) == ["id", "prompt", "base_response", "ft_response", "score_base", "score_ft"]
    assert df.loc[0, "id"] == 1
    assert df.loc[0, "prompt"] == "a"


def test_merge_scores_matches_by_id(tmp_path):
    pairs = [
        {"id": 1, "prompt": "a", "score_base": None, "score_ft": None},
        {"id": 2, "prompt": "b", "score_base": None, "score_ft": None},
    ]
    scores_path = tmp_path / "scores.csv"
    scores_path.write_text(
        "id,prompt,base_response,ft_response,score_base,score_ft\n"
        "1,a,x,y,3,4\n"
        "2,b,x2,y2,5,2\n",
        encoding="utf-8",
    )

    merged = evaluate.merge_scores(pairs, scores_path)

    assert merged[0]["score_base"] == 3
    assert merged[0]["score_ft"] == 4
    assert merged[1]["score_base"] == 5
    assert merged[1]["score_ft"] == 2


def test_render_html_escapes_model_output():
    pairs = [
        {
            "prompt": "<b>hi</b>",
            "base_response": "<script>alert(1)</script>",
            "ft_response": "ok",
            "score_base": None,
            "score_ft": None,
        }
    ]
    metrics = {
        "base_perplexity": 10.0,
        "finetuned_perplexity": 8.0,
        "improved_pct": None,
        "n_scored": 0,
    }

    text = evaluate.render_html(pairs, metrics, regression_pairs=[])

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "<script>alert(1)</script>" not in text
    assert "&lt;b&gt;hi&lt;/b&gt;" in text
    assert "10.00" in text
    assert "8.00" in text


def test_render_html_shows_improvement_and_regression_section():
    pairs = [
        {
            "prompt": "q1",
            "base_response": "base ans",
            "ft_response": "ft ans",
            "score_base": 2,
            "score_ft": 4,
        }
    ]
    regression = [
        {
            "prompt": "gen prompt",
            "base_response": "b",
            "ft_response": "f",
            "score_base": None,
            "score_ft": None,
        }
    ]
    metrics = {
        "base_perplexity": 10.0,
        "finetuned_perplexity": 8.0,
        "improved_pct": 100.0,
        "n_scored": 1,
    }

    text = evaluate.render_html(pairs, metrics, regression_pairs=regression)

    assert "base ans" in text
    assert "ft ans" in text
    assert "100.0" in text
    assert "gen prompt" in text
