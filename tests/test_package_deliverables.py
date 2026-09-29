import hashlib
import zipfile
from pathlib import Path

import package_deliverables

HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def make_run_dir(tmp_path):
    run = tmp_path / "output"
    (run / "lora_adapter").mkdir(parents=True)
    (run / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
    (run / "lora_adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    return run


def test_sha256_matches_known_literal(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")

    assert package_deliverables.sha256_of(path) == HELLO_SHA256


def test_collect_files_includes_adapter_and_skips_missing_dirs(tmp_path):
    run = make_run_dir(tmp_path)

    files = package_deliverables.collect_files(run)

    relative = {str(Path(f).relative_to(run)) for f in files}
    assert "lora_adapter/adapter_model.safetensors" in relative
    assert "lora_adapter/adapter_config.json" in relative
    assert not any("merged" in item for item in relative)
    assert not any("eval" in item for item in relative)


def test_collect_files_includes_optional_dirs_when_present(tmp_path):
    run = make_run_dir(tmp_path)
    (run / "merged").mkdir()
    (run / "merged" / "model.safetensors").write_bytes(b"merged-bytes")
    (run / "eval").mkdir()
    (run / "eval" / "eval_report.html").write_text("<html></html>", encoding="utf-8")

    files = package_deliverables.collect_files(run)

    relative = {str(Path(f).relative_to(run)) for f in files}
    assert "merged/model.safetensors" in relative
    assert "eval/eval_report.html" in relative


def test_package_writes_zip_with_manifest_and_checksums(tmp_path):
    run = make_run_dir(tmp_path)
    zip_path = tmp_path / "deliverable_test.zip"

    result = package_deliverables.package(run, zip_path, client="demo")

    assert zip_path.exists()
    assert result["file_count"] == 2
    adapter_digest = hashlib.sha256(b"adapter-bytes").hexdigest()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert "MANIFEST.txt" in names
        assert "SHA256SUMS" in names
        assert "lora_adapter/adapter_model.safetensors" in names
        manifest = archive.read("MANIFEST.txt").decode("utf-8")
        checksums = archive.read("SHA256SUMS").decode("utf-8")
    assert "demo" in manifest
    assert "adapter_model.safetensors" in manifest
    assert f"{adapter_digest}  lora_adapter/adapter_model.safetensors" in checksums


def test_package_rejects_empty_run_dir(tmp_path):
    empty = tmp_path / "output"
    empty.mkdir()

    import pytest

    with pytest.raises(ValueError):
        package_deliverables.package(empty, tmp_path / "x.zip", client="demo")
