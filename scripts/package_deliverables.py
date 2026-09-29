#!/usr/bin/env python3
"""Package fine-tuning deliverables into a zip with manifest and checksums."""

import argparse
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

OPTIONAL_DIRS = ["merged", "gguf_gguf", "eval"]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(run_dir):
    run_dir = Path(run_dir)
    files = []
    adapter_dir = run_dir / "lora_adapter"
    if adapter_dir.exists():
        files.extend(sorted(path for path in adapter_dir.rglob("*") if path.is_file()))
    for name in OPTIONAL_DIRS:
        directory = run_dir / name
        if directory.exists():
            files.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return files


def package(run_dir, zip_path, client="client"):
    run_dir = Path(run_dir)
    zip_path = Path(zip_path)
    files = collect_files(run_dir)
    if not files:
        raise ValueError(f"nothing to package under {run_dir}")
    entries = [(path, str(path.relative_to(run_dir))) for path in files]

    manifest_lines = [
        f"# deliverable - client: {client}",
        f"# created: {date.today().isoformat()}",
        f"# files: {len(entries)}",
        "",
    ]
    checksum_lines = []
    for path, relative in entries:
        digest = sha256_of(path)
        manifest_lines.append(f"{relative}  {path.stat().st_size} bytes  sha256:{digest}")
        checksum_lines.append(f"{digest}  {relative}")
    manifest = "\n".join(manifest_lines) + "\n"
    checksums = "\n".join(checksum_lines) + "\n"

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, relative in entries:
            archive.write(path, relative)
        archive.writestr("MANIFEST.txt", manifest)
        archive.writestr("SHA256SUMS", checksums)

    return {"zip": str(zip_path), "file_count": len(entries), "client": client}


def main():
    parser = argparse.ArgumentParser(description="Package deliverables into a zip")
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--out")
    parser.add_argument("--client", default="client")
    args = parser.parse_args()

    if args.config:
        from train_lora import load_config

        cfg = load_config(args.config)
        client = args.client if args.client != "client" else cfg["client"]
        run_dir = args.run_dir or cfg["output_dir"]
        out = args.out or f"deliverable_{client}.zip"
    else:
        client = args.client
        run_dir = args.run_dir
        out = args.out

    missing = [
        name
        for name, value in (("--run-dir", run_dir), ("--out", out))
        if not value
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    result = package(run_dir, out, client=client)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
