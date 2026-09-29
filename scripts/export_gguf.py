#!/usr/bin/env python3
"""Export GGUF files (Ollama / llama.cpp) from a LoRA adapter."""

import argparse
from pathlib import Path

MODELFILE_TEMPLATE = '''FROM {gguf_name}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
'''


def export(adapter_dir, output_dir, quantization="q4_k_m", max_seq_length=2048):
    import unsloth
    from unsloth import FastLanguageModel

    output_dir = Path(output_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model.save_pretrained_gguf(str(output_dir), tokenizer, quantization_method=quantization)

    candidates = []
    for directory in (output_dir, Path(str(output_dir) + "_gguf")):
        if directory.exists():
            candidates.extend(sorted(directory.rglob("*.gguf")))
    if not candidates:
        raise RuntimeError(f"no .gguf file produced under {output_dir} or {output_dir}_gguf")
    preferred = [path for path in candidates if quantization in path.name]
    gguf_path = preferred[0] if preferred else candidates[0]

    modelfile_path = gguf_path.parent / "Modelfile"
    if not modelfile_path.exists():
        modelfile_path.write_text(
            MODELFILE_TEMPLATE.replace("{gguf_name}", gguf_path.name),
            encoding="utf-8",
        )
    return {"gguf": gguf_path, "modelfile": modelfile_path}


def main():
    parser = argparse.ArgumentParser(description="Export GGUF from a LoRA adapter")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quant", default="q4_k_m")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    artifacts = export(
        args.adapter,
        args.output,
        quantization=args.quant,
        max_seq_length=args.max_seq_length,
    )
    print(f"gguf: {artifacts['gguf']}")
    print(f"Modelfile: {artifacts['modelfile']}")


if __name__ == "__main__":
    main()
