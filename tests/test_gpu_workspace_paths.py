"""CPU-only path/delivery tests. No model weights or GPU dependencies are loaded."""
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
import pytest
from recovery_agent.paths import WorkspacePaths, project_path, resolve_cli_paths
from recovery_agent.training.data import create_delivery, verify_delivery, sha256
from recovery_agent.gpu_workspace import model_inventory

ROOT = Path(__file__).resolve().parents[1]

def test_relative_paths_ignore_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert project_path("data/v0.3b_seed42_final") == ROOT/"data/v0.3b_seed42_final"
    from argparse import Namespace
    args = Namespace(output="outputs/example")
    resolve_cli_paths(args, ["output"])
    assert args.output == ROOT/"outputs/example"
    assert WorkspacePaths.from_env({}).sft == ROOT/"outputs/v0.4a_training_readiness"

def test_configured_workspace_and_idempotent_setup(tmp_path):
    storage = tmp_path/"data"
    project = storage/"RecoveryAgent"
    project.mkdir(parents=True)
    paths = WorkspacePaths.from_env({"RECOVERY_DATA_ROOT": str(storage),
                                    "RECOVERY_MODEL_PATH": str(storage/"models/custom")}, project)
    first = paths.initialize()
    protected = paths.datasets/"user.txt"
    protected.write_text("KEEP", encoding="utf-8")
    assert paths.initialize() == first and protected.read_text() == "KEEP"
    assert paths.outputs == storage/"outputs/recovery_agent"
    assert paths.model == storage/"models/custom"
    with pytest.raises(ValueError, match="under"):
        WorkspacePaths.from_env({"RECOVERY_DATA_ROOT": str(storage),
                                 "RECOVERY_OUTPUT_PATH": str(tmp_path/"outside")}, project)

def test_delivery_sha256_and_no_overwrite(tmp_path):
    workspace = WorkspacePaths.from_env({})
    destination = tmp_path/"delivery.json"
    manifest = create_delivery(workspace, destination)
    assert len(manifest["files"]) == 18
    assert verify_delivery(workspace, destination)["content_digest"] == manifest["content_digest"]
    with pytest.raises(FileExistsError):
        create_delivery(workspace, destination)
    changed = json.loads(destination.read_text())
    changed["files"][0]["sha256"] = "0"*64
    destination.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest"):
        verify_delivery(workspace, destination)
    sample = tmp_path/"sample.bin"
    sample.write_bytes(b"cpu-hash-fixture")
    assert sha256(sample) == __import__("hashlib").sha256(sample.read_bytes()).hexdigest()
    # These are inventory fixtures, not usable or claimed real model weights.
    model = tmp_path/"fake_model"
    model.mkdir()
    (model/"config.json").write_text('{"model_type":"qwen3"}')
    (model/"model.safetensors.index.json").write_text('{"weight_map":{"a":"part.safetensors"}}')
    with pytest.raises(FileNotFoundError):
        model_inventory(model)
    (model/"part.safetensors").write_bytes(b"inventory-only")
    assert model_inventory(model)["weight_bytes"] == 14

def test_existing_clis_from_another_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/"scripts/setup_gpu_workspace.py"), "--dry-run"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and str(ROOT) in json.loads(result.stdout)["project"]
    result = subprocess.run([sys.executable, str(ROOT/"scripts/harden_data.py"), "collect", "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
