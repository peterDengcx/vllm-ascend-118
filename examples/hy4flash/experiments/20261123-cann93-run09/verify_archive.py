"""Verify the measured archive without importing or starting an NPU runtime."""

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def check_resources(source, directory):
    spec = importlib.util.spec_from_file_location("archived_mooncake", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    descriptor = {
        "version": "1.3",
        "endpoint_list": [{"protocol": "ub_ctp", "placement": "device"}],
    }
    platform = SimpleNamespace(
        visible_device_id_to_physical_device_id=lambda logical: [4, 5, 6, 7][logical]
    )
    modules = {
        "torch": SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 2)),
        "vllm.platforms": SimpleNamespace(current_platform=platform),
    }
    with patch.dict(os.environ, {"ASCEND_LOCAL_COMM_RES": "unchanged"}, clear=True):
        module._configure_local_comm_res()
        assert os.environ["ASCEND_LOCAL_COMM_RES"] == "unchanged"
    path = directory / "6.json"
    path.write_text(json.dumps(descriptor))
    env = {"HY4_MOONCAKE_LOCAL_COMM_RES_DIR": str(directory)}
    with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True):
        module._configure_local_comm_res()
        assert json.loads(os.environ["ASCEND_LOCAL_COMM_RES"]) == descriptor
        invalid = [
            {"version": "1.2", "endpoint_list": descriptor["endpoint_list"]},
            {"version": "1.3", "endpoint_list": []},
            {"version": "1.3", "endpoint_list": [{"protocol": "tcp", "placement": "device"}]},
            {"version": "1.3", "endpoint_list": [{"protocol": "ub_ctp", "placement": "host"}]},
        ]
        for value in invalid:
            path.write_text(json.dumps(value))
            try:
                module._configure_local_comm_res()
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid descriptor: {value}")
        path.unlink()
        try:
            module._configure_local_comm_res()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing physical-card descriptor accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source-manifest.json").read_text())
    evidence = json.loads((root / "archive-sha256.json").read_text())
    for name, expected in evidence.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    archive = subprocess.check_output(
        ["git", "-C", str(args.base_repo), "archive", manifest["upstream_commit"]]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        base = {m.name: source.extractfile(m).read() for m in source if m.isfile()}
    with tempfile.TemporaryDirectory(prefix="hy4-run09-verify-") as tmp:
        directory = Path(tmp)
        for item in manifest["changed_files"]:
            name = item["path"]
            if name not in base:
                assert item["base_sha256"] is None
                continue
            data = base[name].replace(b"\r\n", b"\n")
            assert hashlib.sha256(data).hexdigest() == item["base_sha256"], name
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        patch_file = str(root / "patches/cann93-run09.patch")
        git = ["git", "-c", "core.autocrlf=false", "apply"]
        subprocess.run([*git, "--check", patch_file], cwd=directory, check=True)
        subprocess.run([*git, patch_file], cwd=directory, check=True)
        for item in manifest["changed_files"]:
            data = (directory / item["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["run09_sha256"], item["path"]
            ast.parse(data, filename=item["path"])
        check_resources(
            directory / "vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py",
            directory,
        )
    for source in (root / "reference").rglob("*.py"):
        ast.parse(source.read_bytes(), filename=str(source))
    summary = json.loads((root / "results/decode-summary.json").read_text())
    assert summary["valid"] and summary["requests"] == 128 and summary["failed"] == 0
    assert summary["prefix_cache"]["hit_ratio"] == 0.9
    assert summary["server"]["P"]["request_prefill_time_seconds"]["count"] == 128
    transport = json.loads((root / "results/communication-validation.json").read_text())
    assert transport["success"] and transport["directional_pairs_passed"] == 32
    assert len(set(map(tuple, transport["pairs"]))) == 32
    print("PASS: archive hashes, 8-file reconstruction, syntax, 7 resource checks, run09 evidence")


if __name__ == "__main__":
    main()
