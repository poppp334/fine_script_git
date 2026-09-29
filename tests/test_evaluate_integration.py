import gc
import json

import pytest
import torch

import evaluate
import train_lora

MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"

TRAIN_ENTRIES = [
    {"messages": [{"role": "user", "content": "สวัสดี"}, {"role": "assistant", "content": "สวัสดีครับ มีอะไรให้ช่วยไหมครับ"}]},
    {"messages": [{"role": "user", "content": "ร้านเปิดกี่โมง"}, {"role": "assistant", "content": "ร้านเปิด 9 โมงเช้าถึง 2 ทุ่มครับ"}]},
    {"messages": [{"role": "user", "content": "ค่าส่งเท่าไหร่"}, {"role": "assistant", "content": "ค่าส่งเริ่มต้น 50 บาทครับ"}]},
    {"messages": [{"role": "user", "content": "ขอบคุณ"}, {"role": "assistant", "content": "ยินดีครับ"}]},
    {"messages": [{"role": "user", "content": "มีสาขาที่ไหนบ้าง"}, {"role": "assistant", "content": "มีสาขาที่กรุงเทพและเชียงใหม่ครับ"}]},
    {"messages": [{"role": "user", "content": "ชำระเงินยังไง"}, {"role": "assistant", "content": "รับโอนและเก็บเงินปลายทางครับ"}]},
]

TEST_ENTRIES = [
    {"messages": [{"role": "user", "content": "สั่งซื้อขั้นต่ำกี่ชิ้น"}, {"role": "assistant", "content": "ขั้นต่ำ 1 ชิ้นครับ"}]},
    {"messages": [{"role": "user", "content": "ส่งของวันไหน"}, {"role": "assistant", "content": "ส่งทุกวันจันทร์ถึงศุกร์ครับ"}]},
    {"messages": [{"role": "user", "content": "รับประกันสินค้าไหม"}, {"role": "assistant", "content": "รับประกัน 7 วันครับ"}]},
]


def write_jsonl(path, entries):
    text = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)
    path.write_text(text + "\n", encoding="utf-8")


@pytest.mark.slow
def test_run_eval_produces_metrics_and_report(tmp_path):
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, TRAIN_ENTRIES)
    test_file = tmp_path / "val.jsonl"
    write_jsonl(test_file, TEST_ENTRIES)
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text(
        "prompt\nมีโปรโมชั่นไหม\nต้องการคุยกับพนักงาน\n", encoding="utf-8"
    )
    out_dir = tmp_path / "eval"

    cfg = {
        "client": "test",
        "model_name": MODEL,
        "dataset_path": str(train_file),
        "val_path": None,
        "output_dir": str(tmp_path / "output"),
        "max_seq_length": 512,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 1,
        "learning_rate": 0.0002,
        "warmup_ratio": 0.1,
        "logging_steps": 1,
        "save_steps": 5,
        "save_total_limit": 1,
        "seed": 42,
        "report_to": "none",
        "resume_from_checkpoint": False,
    }
    train_lora.train(cfg, dry_run=True)
    gc.collect()
    torch.cuda.empty_cache()

    metrics = evaluate.run_eval(
        base_model=MODEL,
        adapter_dir=str(tmp_path / "output" / "lora_adapter"),
        test_file=test_file,
        prompts_file=prompts_file,
        out_dir=out_dir,
        max_new_tokens=16,
    )

    assert metrics["base_perplexity"] > 0
    assert metrics["finetuned_perplexity"] > 0
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "eval_pairs.csv").exists()
    report = (out_dir / "eval_report.html").read_text(encoding="utf-8")
    assert "มีโปรโมชั่นไหม" in report
