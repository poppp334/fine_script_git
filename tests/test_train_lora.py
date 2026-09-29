import yaml
import pytest

import train_lora


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_missing_dataset_path_raises(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    write_yaml(cfg_path, {"model_name": "some/model"})

    with pytest.raises(ValueError):
        train_lora.load_config(cfg_path)


def test_defaults_and_overrides(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    write_yaml(cfg_path, {"dataset_path": "data/train.jsonl", "epochs": 5, "lora_rank": 8})

    cfg = train_lora.load_config(cfg_path)

    assert cfg["epochs"] == 5
    assert cfg["lora_rank"] == 8
    assert cfg["model_name"] == "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    assert cfg["batch_size"] == 1
    assert cfg["resume_from_checkpoint"] is True


def test_lora_alpha_defaults_to_double_rank(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    write_yaml(cfg_path, {"dataset_path": "data/train.jsonl", "lora_rank": 8})

    cfg = train_lora.load_config(cfg_path)

    assert cfg["lora_alpha"] == 16


def test_explicit_lora_alpha_is_kept(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    write_yaml(cfg_path, {"dataset_path": "data/train.jsonl", "lora_rank": 8, "lora_alpha": 64})

    cfg = train_lora.load_config(cfg_path)

    assert cfg["lora_alpha"] == 64
