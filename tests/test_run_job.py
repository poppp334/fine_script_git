import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RUN_JOB = SCRIPTS / "run_job.sh"


def test_dry_run_lists_all_stages(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "client: demo\noutput_dir: output\ndataset_path: data/train.jsonl\n",
        encoding="utf-8",
    )
    prompts = tmp_path / "p.csv"
    prompts.write_text("prompt\nสวัสดี\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(RUN_JOB),
            "--config",
            str(config),
            "--prompts",
            str(prompts),
            "--gguf",
            "--package",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for name in [
        "validate_data.py",
        "train_lora.py",
        "merge_lora.py",
        "export_gguf.py",
        "evaluate.py",
        "package_deliverables.py",
    ]:
        assert name in result.stdout, f"{name} missing from dry-run plan"


def test_dry_run_without_optional_stages_skips_them(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "client: demo\noutput_dir: output\ndataset_path: data/train.jsonl\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(RUN_JOB), "--config", str(config), "--dry-run"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "evaluate.py" not in result.stdout
    assert "package_deliverables.py" not in result.stdout


def test_missing_config_exits_with_usage():
    result = subprocess.run(["bash", str(RUN_JOB)], capture_output=True, text=True)

    assert result.returncode != 0
    assert "usage" in (result.stdout + result.stderr).lower()
