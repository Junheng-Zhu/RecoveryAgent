"""Saved-label/metadata guards on CPU; no forward/backward or real PEFT claim."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from recovery_agent.paths import WorkspacePaths
from recovery_agent.training.data import (
    load_training_data, DynamicCollator, SavedTokenDataset, supervised_count,
    validate_rows, tokenizer_identity, json_file, sha256)
from recovery_agent.training.trainer import (
    load_config, batches_for_epoch, resolve_lora_targets, resume_contract, validate_resume)
from recovery_agent.hardening.tasks import validate_frozen
from recovery_agent.hardening.protocol import schema_hash
from recovery_agent.data_pipeline.common import fingerprint

ROOT = Path(__file__).resolve().parents[1]

def test_real_saved_cohorts_and_dynamic_padding():
    dataset, report = load_training_data(WorkspacePaths.from_env({}))
    assert len(dataset) == 204 and report["quality"]["teacher"]["samples"] == 84
    assert report["quality"]["oracle"]["effective_supervised_tokens"] == 4294
    original = copy.deepcopy(dataset.rows[:2])
    batch = DynamicCollator(report["tokenizer_identity"]["pad_token_id"])([dataset[0], dataset[1]])
    assert len(batch["input_ids"][0]) == max(len(row["input_ids"]) for row in original) < 4096
    assert dataset.rows[:2] == original
    assert all(label == -100 for labels, masks in zip(batch["labels"], batch["attention_mask"])
               for label, mask in zip(labels, masks) if not mask)
    assert sum(supervised_count(labels) for labels in batch["labels"]) == sum(
        row["assistant_supervised_tokens"] for row in original)
    teacher, _ = load_training_data(WorkspacePaths.from_env({}), source="teacher")
    assert len(teacher) == 84

def test_label_split_source_corruption_and_lengths():
    workspace = WorkspacePaths.from_env({})
    dataset, report = load_training_data(workspace)
    tasks, _ = validate_frozen(workspace.frozen, schema_hash())
    stats = json_file(workspace.sft/"token_stats.json")
    audit = json_file(workspace.sft/"source_audit.json")
    rows = copy.deepcopy(dataset.rows[:2])
    audit["trustworthy_decisions"]["oracle_reference"] = 2
    identity = report["tokenizer_identity"]
    result = validate_rows(rows, "oracle", tasks, stats, audit, 4096, identity)
    assert result["repeated_task_decisions"] == 1  # several decisions != independent tasks
    corrupted = copy.deepcopy(rows)
    corrupted[0]["labels"][0] = corrupted[0]["input_ids"][0]
    with pytest.raises(ValueError, match="labels/EOS"):
        validate_rows(corrupted, "oracle", tasks, stats, audit, 4096, identity)
    corrupted = copy.deepcopy(rows)
    corrupted[0]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        validate_rows(corrupted, "oracle", tasks, stats, audit, 4096, identity)
    with pytest.raises(ValueError, match="split"):
        validate_rows(rows, "teacher", tasks, stats, audit, 4096, identity)
    with pytest.raises(ValueError, match="truncation"):
        validate_rows(rows, "oracle", tasks, stats, audit, 8, identity)

def test_training_config_sampler_and_defaults(tmp_path):
    workspace = WorkspacePaths.from_env({})
    config, _ = load_config("configs/qwen3_1.7b_oracle_lora.yaml", workspace)
    assert config["data"]["source"] == "oracle" and config["training"]["gradient_accumulation_steps"] == 8
    assert config["model"]["max_seq_length"] == 4096
    a = batches_for_epoch(17, 3, 42, 0)
    assert a == batches_for_epoch(17, 3, 42, 0) and a != batches_for_epoch(17, 3, 42, 1)
    assert sorted(i for batch in a for i in batch) == list(range(17))
    import yaml
    bad = copy.deepcopy(config)
    bad.pop("_paths")
    bad["training"]["dtype"] = "fp16"
    p = tmp_path/"bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="configuration"):
        load_config(p, workspace)

def test_lora_targets_inspect_each_layer_and_type():
    class FakeLinear:
        weight = SimpleNamespace(ndim=2)
    scopes = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
              "mlp": ("gate_proj", "up_proj", "down_proj")}
    modules = {f"model.layers.{i}.{scope}.{name}": FakeLinear()
               for i in range(2) for scope, names in scopes.items() for name in names}
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3", num_hidden_layers=2),
                            named_modules=lambda: modules.items())
    assert len(resolve_lora_targets(model, FakeLinear)) == 14
    modules["model.layers.1.self_attn.q_proj"] = SimpleNamespace(weight=SimpleNamespace(ndim=2))
    with pytest.raises(ValueError, match="Linear"):
        resolve_lora_targets(model, FakeLinear)

def test_checkpoint_hash_and_resume_compatibility(tmp_path):
    config, _ = load_config("configs/qwen3_1.7b_oracle_lora.yaml", WorkspacePaths.from_env({}))
    report = {"selected_file_sha256": "data", "delivery_digest": "delivery",
              "frozen_manifest_sha256": "frozen", "tokenizer_identity": {"template": "official"}}
    contract = resume_contract(config, report, {"digest": "model"}, ["verified.linear"], "train")
    # Metadata fixtures do not test deserialization, optimizer continuity or real GPU resume.
    for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.pt"):
        (tmp_path/name).write_bytes(b"cpu-metadata-fixture")
    meta = {"contract": contract, "contract_digest": fingerprint(contract),
            "progress": {"epoch": 1, "batch_offset": 0, "global_step": 26},
            "artifact_sha256": {p.name: sha256(p) for p in tmp_path.iterdir()}}
    (tmp_path/"checkpoint_metadata.json").write_text(json.dumps(meta))
    assert validate_resume(tmp_path, contract)["progress"]["global_step"] == 26
    incompatible = copy.deepcopy(contract)
    incompatible["training"]["learning_rate"] /= 2
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume(tmp_path, incompatible)
    (tmp_path/"trainer_state.pt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash"):
        validate_resume(tmp_path, contract)
