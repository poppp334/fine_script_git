# fine_script_git — LLM Fine-tuning Template Scripts

สคลิปต์เทมเพลตสำหรับงาน Custom LLM Fine-tuning (Lab → Cloud ลูกค้า)
สแต็ก: Unsloth + TRL + Transformers + PEFT (QLoRA), รันจริงบน GPU

## Requirements

- Linux + NVIDIA GPU (CUDA)
- [uv](https://docs.astral.sh/uv/) (สำหรับสร้าง venv + ติดตั้ง)
- Python 3.11–3.12 (uv จัดการให้ได้ — เครื่อง host อาจเป็น 3.14 ก็ไม่ต้องสน)

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.lock
python scripts/smoke_test.py        # เทรน 5 steps บนโมเดลจิ๋ว — ต้อง SMOKE PASS ก่อนเริ่มงานทุกครั้ง
```

## Scripts

| สคริปต์ | หน้าที่ |
|---|---|
| `scripts/prepare_data.py` | แปลง Excel/CSV/JSONL → ChatML JSONL (+ dedup, skip แถวเสีย, val split) |
| `scripts/validate_data.py` | ตรวจคุณภาพ data + รายงาน JSON (exit 1 ถ้ามี error) |
| `scripts/train_lora.py` | เทรน QLoRA จาก config YAML (`--dry-run` สำหรับทดสอบ) |
| `scripts/smoke_test.py` | ตรวจ env + GPU: เทรน 5 steps → save → reload → generate |
| `scripts/merge_lora.py` | รวม adapter เข้า base เป็น merged model ส่งมอบ |
| `scripts/export_gguf.py` | แปลงเป็น GGUF (q4_k_m) + Modelfile สำหรับ Ollama |
| `scripts/inference.py` | ทดสอบโมเดล: ทีละ prompt หรือ batch CSV (before/after) |

## ตัวอย่างใช้งาน

```bash
python scripts/prepare_data.py --input data/raw.xlsx --output data/train.jsonl --val-split 0.05 --val-output data/val.jsonl
python scripts/validate_data.py --data data/train.jsonl
python scripts/train_lora.py --config configs/client_x.yaml
python scripts/merge_lora.py --adapter output/lora_adapter --output output/merged
python scripts/export_gguf.py --adapter output/lora_adapter --output output/gguf
python scripts/inference.py --model output/merged --prompts-csv prompts.csv --out-csv results.csv

# นำ GGUF เข้า Ollama (รันในโฟลเดอร์ที่มี Modelfile — ดู path จาก output ของ export_gguf)
cd output/gguf_gguf
ollama create my-model -f Modelfile
ollama run my-model "สวัสดี"
```

## Tests

```bash
.venv/bin/python -m pytest -q              # ทั้งหมด (มี GPU tests)
.venv/bin/python -m pytest -m "not slow"   # เฉพาะเทสต์เร็ว
```

## Data policy (สำคัญ)

- **ห้าม commit ข้อมูลลูกค้า / config ลูกค้า** ขึ้น repo นี้เด็ดขาด — ส่งด้วย `rsync` เท่านั้น
- repo นี้เก็บเฉพาะโค้ด/เทมเพลต; adapter/model/gguf ไม่ commit (ดู `.gitignore`)
