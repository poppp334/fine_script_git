# คู่มือใช้งานโมเดล — [ชื่อลูกค้า/โครงการ]

ส่งมอบเมื่อ: `[วันที่]` · แพ็ก: `[Micro/Starter/Standard/Pro]` · ผู้ให้บริการ: `[ชื่อ/ช่องทางติดต่อ]`

## 1. สิ่งที่ได้รับในแพ็กนี้

| ไฟล์/โฟลเดอร์ | คำอธิบาย |
|---|---|
| `lora_adapter/` | LoRA adapter — ใช้คู่กับ base model `[ชื่อโมเดลฐาน]` |
| `merged/` | โมเดลรวมแล้ว (16-bit) — ใช้ได้ทันทีกับ transformers / vLLM |
| `gguf_gguf/` | ไฟล์ GGUF + `Modelfile` — สำหรับ Ollama / LM Studio |
| `eval/` | `eval_report.html` + `metrics.json` — ผลประเมินก่อน/หลัง |
| `MANIFEST.txt`, `SHA256SUMS` | รายการไฟล์ + checksum สำหรับตรวจความถูกต้อง |
| `scripts/` | สคริปต์ inferance/ทดสอบ (ถ้าแพ็กมี) |

## 2. วิธีใช้แบบเร็วสุด — Ollama (GGUF)

```bash
cd gguf_gguf
ollama create [ชื่อโมเดล] -f Modelfile
ollama run [ชื่อโมเดล] "สวัสดี"
# โหมดแชทต่อเนื่อง:
ollama run [ชื่อโมเดล]
```

## 3. ใช้กับ Python (merged model)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./merged", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./merged")

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "สวัสดี"}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## 4. ใช้แบบ Production — vLLM

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model ./merged --port 8000
# เรียกผ่าน OpenAI-compatible API ที่ http://localhost:8000/v1
```

## 5. ใช้ adapter กับ base model (ประหยัดพื้นที่)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("[ชื่อโมเดลฐาน]", device_map="auto")
model = PeftModel.from_pretrained(base, "./lora_adapter")
```

## 6. ผลการทดสอบที่แนบมา

- เปิด `eval/eval_report.html` — เทียบคำตอบก่อน/หลังทีละคำถาม
- `metrics.json` — perplexity ก่อน/หลัง และสถิติอื่น
- ตรวจไฟล์: `sha256sum -c SHA256SUMS` (รันในโฟลเดอร์ที่แตกไฟล์)

## 7. การสนับสนุนหลังส่งมอบ

- แจ้งปัญหาการใช้งาน/บั๊กสคริปต์ได้ภายใน **14 วัน** ที่ `[ช่องทางติดต่อ]` — แก้ให้ฟรี
- อยากเทรนเพิ่ม/ปรับข้อมูล/เปลี่ยนโมเดล → quote งานใหม่
- รายละเอียดเงื่อนไข: ดู `docs/support_policy.md`

## 8. หมายเหตุลิขสิทธิ์โมเดลฐาน

โมเดลฐาน `[ชื่อ]` เผยแพร่ภายใต้ `[ชื่อ license]` — สรุปเงื่อนไขการใช้งานเชิงพาณิชย์: `[สรุปสั้นจาก license_cheatsheet]`
