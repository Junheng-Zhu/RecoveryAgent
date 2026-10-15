"""Immutable v0.4-A delivery, label validation and dynamic padding (no model imports)."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from ..paths import project_path, writable
from ..data_pipeline.common import fingerprint, read_jsonl, write_json
from ..hardening.tasks import validate_frozen
from ..hardening.protocol import schema_hash
from ..sft_preparation import sample_key
from ..qwen3_tokenizer import validate_messages

TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
SFT_FILES = ("source_audit.json", "oracle_sft.jsonl", "teacher_sft.jsonl",
             "token_stats.json", "mask_examples.json", "readiness_report.md")

def sha256(path):
    """Streaming hashes work for large local model shards without loading weights."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def create_delivery(workspace, output):
    """Inventory existing files only; never rebuild labels, copy datasets or fetch a model."""
    output = project_path(output, workspace.project)
    if output.exists():
        raise FileExistsError("Delivery manifest already exists; verify it or choose a new path")
    audit = json_file(workspace.sft/"source_audit.json")
    stats = json_file(workspace.sft/"token_stats.json")
    if stats["status"] != "verified" or not audit["original_sources_unchanged"]:
        raise ValueError("v0.4-A is not verified")
    files = []
    for cohort, directory, names in (
        ("frozen", workspace.frozen, [p.name for p in workspace.frozen.iterdir() if p.is_file()]),
        ("sft", workspace.sft, SFT_FILES),
        ("tokenizer", workspace.tokenizer, TOKENIZER_FILES)):
        for name in sorted(names):
            p = directory/name
            if not p.is_file():
                raise FileNotFoundError(p)
            files.append({"cohort": cohort, "name": name, "size": p.stat().st_size,
                          "sha256": sha256(p), "source": str(p.relative_to(workspace.project))
                          if p.is_relative_to(workspace.project) else str(p),
                          "server_destination": str(directory/name)})
    manifest = {"version": "v0.4-B", "first_training_source": "oracle",
                "dataset_version": "0.3-B", "preparation_version": "v0.4-A",
                "files": files, "tokenizer": stats["tokenizer"], "template": stats["template"],
                "trusted_decisions": audit["trustworthy_decisions"],
                "weights": {"downloaded": False, "shard_hashes": None,
                            "required_action": "User provides local complete base weights; GPU preflight hashes them."}}
    manifest["content_digest"] = fingerprint(manifest)
    write_json(output, manifest)
    return manifest

def verify_delivery(workspace, manifest_path=None):
    """The shipped manifest is immutable; altered/missing artifacts block every training mode."""
    path = Path(manifest_path or workspace.delivery)
    manifest = json_file(path)
    core = {k: v for k, v in manifest.items() if k != "content_digest"}
    if manifest.get("version") != "v0.4-B" or fingerprint(core) != manifest.get("content_digest"):
        raise ValueError("Invalid delivery manifest")
    roots = {"frozen": workspace.frozen, "sft": workspace.sft, "tokenizer": workspace.tokenizer}
    required = {(c, n) for c, names in (("sft", SFT_FILES), ("tokenizer", TOKENIZER_FILES)) for n in names}
    seen = set()
    for record in manifest["files"]:
        key = (record["cohort"], record["name"])
        if key in seen or record["cohort"] not in roots or Path(record["name"]).name != record["name"]:
            raise ValueError("Invalid/duplicate delivery artifact")
        seen.add(key)
        p = roots[record["cohort"]]/record["name"]
        if not p.is_file() or p.stat().st_size != record["size"] or sha256(p) != record["sha256"]:
            raise ValueError(f"Delivery SHA-256 mismatch: {key}")
    if not required.issubset(seen):
        raise ValueError("Incomplete delivery manifest")
    return manifest

def tokenizer_identity(directory, expected):
    """Compare original official files and template hashes without re-encoding any sample."""
    directory = Path(directory)
    for name, digest in expected["tokenizer"]["file_sha256"].items():
        if not (directory/name).is_file() or sha256(directory/name) != digest:
            raise ValueError(f"Tokenizer/model config hash mismatch: {name}")
    config = json_file(directory/"tokenizer_config.json")
    template = config.get("chat_template")
    if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != expected["template"]["template_sha256"]:
        raise ValueError("Chat template differs from v0.4-A")
    model_config = json_file(directory/"config.json")
    if model_config.get("model_type") != "qwen3":
        raise ValueError("Expected official Qwen3 model config")
    pad = config.get("pad_token")
    pad = pad.get("content") if isinstance(pad, dict) else pad
    pad_ids = [int(i) for i, value in config.get("added_tokens_decoder", {}).items()
               if value.get("content") == pad]
    if len(pad_ids) != 1:
        raise ValueError("Official tokenizer pad token is missing/ambiguous")
    return {"files": expected["tokenizer"]["file_sha256"],
            "template_sha256": expected["template"]["template_sha256"],
            "vocab_size": model_config["vocab_size"],
            "native_context": model_config["max_position_embeddings"],
            "eos_token_id": expected["template"]["eos_token_id"],
            "pad_token_id": pad_ids[0]}

def supervised_count(labels):
    """Causal LM shifts targets one position; index zero can never receive loss."""
    return sum(label != -100 for label in labels[1:])

def validate_rows(rows, source, tasks, stats, audit, max_seq_length, identity):
    """Trust requires Train membership, causal prompt hashes, audit provenance and exact saved masks."""
    expected_source = "oracle_reference" if source == "oracle" else "teacher"
    task_map = {t["id"]: t for t in tasks}
    episode_map = {e["trajectory_id"]: e for e in audit["episodes"]}
    seen_ids, seen_content = set(), set()
    for row in rows:
        task = task_map.get(row["task_id"])
        episode = episode_map.get(row["trajectory_id"])
        if (not task or task["split"] != "train" or task["eval_group"] != "train" or
            row["split"] != "train" or row["eval_group"] != "train" or row["source"] != expected_source):
            raise ValueError("Source/split leakage in training data")
        if (not episode or episode["task_id"] != task["id"] or not episode["replay_verified"] or
            not episode["final_verifier"]["success"] or not row["evidence"]["replay_verified"] or
            not row["evidence"]["final_verifier_success"]):
            raise ValueError("Missing trusted execution provenance")
        if row["sample_id"] in seen_ids or sample_key(row) in seen_content:
            raise ValueError("Duplicate SFT decision")
        seen_ids.add(row["sample_id"])
        seen_content.add(sample_key(row))
        if (row["schema_hash"] != schema_hash() or row["dataset_version"] != task["dataset_version"] or
            row["prompt_hash"] != fingerprint({"messages": row["messages"][:-1], "tools": row["tools"]})):
            raise ValueError("Unverified schema, dataset or causal prompt")
        bindings = validate_messages(row["messages"], row["tools"])
        if bindings != row["tool_bindings"]:
            raise ValueError("Tool ID bindings altered")
        ids, labels, attention = row["input_ids"], row["labels"], row["attention_mask"]
        span = row["target_token_span"]
        if (not ids or len(ids) != len(labels) or attention != [1]*len(ids) or
            len(span) != 2 or not all(type(v) is int for v in span) or not 0 < span[0] < span[1] <= len(ids)):
            raise ValueError("Invalid saved token arrays/span")
        start, end = span
        expected_labels = [-100]*len(ids)
        expected_labels[start:end] = ids[start:end]
        if (labels != expected_labels or ids[end-1] != identity["eos_token_id"] or
            row["loss_scope"] != "final_assistant_only" or row.get("truncated") or
            any(type(t) is not int or not 0 <= t < identity["vocab_size"] for t in ids)):
            raise ValueError("Assistant-only labels/EOS corrupted")
        count = supervised_count(labels)
        recorded = stats["per_sample"].get(row["sample_id"])
        if (count < 2 or row["token_length"] != len(ids) or row["assistant_supervised_tokens"] != count or
            recorded != {"length": len(ids), "supervised": count}):
            raise ValueError("Token statistics disagree with saved labels")
        if len(ids) > max_seq_length:
            raise ValueError("Sequence exceeds configured context; truncation/filtering is forbidden")
    if not rows or len(rows) != audit["trustworthy_decisions"][expected_source]:
        raise ValueError("Incomplete trusted source cohort")
    return {"samples": len(rows), "independent_tasks": len({r["task_id"] for r in rows}),
            "repeated_task_decisions": len(rows)-len({r["task_id"] for r in rows}),
            "mechanisms": dict(Counter(r["mechanism_id"] for r in rows)),
            "effective_supervised_tokens": sum(supervised_count(r["labels"]) for r in rows),
            "max_length": max(len(r["input_ids"]) for r in rows),
            "duplicate_samples": 0, "split_leakage": 0}

class SavedTokenDataset:
    """Pass immutable saved token IDs/labels to the trainer; no tokenize/map/truncate step."""
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return {key: list(self.rows[index][key]) for key in ("input_ids", "attention_mask", "labels")}

class DynamicCollator:
    """Right-pad only to this batch's maximum; padding positions never participate in loss."""
    def __init__(self, pad_token_id, tensors=False):
        self.pad_token_id, self.tensors = pad_token_id, tensors
    def __call__(self, batch):
        if not batch:
            raise ValueError("Empty batch")
        size = max(len(row["input_ids"]) for row in batch)
        result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in batch:
            pad = size-len(row["input_ids"])
            for key, value in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                result[key].append(list(row[key])+[value]*pad)
        if self.tensors:
            import torch
            return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        return result

def load_training_data(workspace, source="oracle", max_seq_length=4096, manifest_path=None):
    """Validate both separate cohorts; select exactly one only after all quality gates pass."""
    if source not in ("oracle", "teacher"):
        raise ValueError("Choose Oracle or Teacher explicitly; implicit mixed training is not supported")
    delivery = verify_delivery(workspace, manifest_path)
    tasks, manifest = validate_frozen(workspace.frozen, schema_hash())
    stats = json_file(workspace.sft/"token_stats.json")
    audit = json_file(workspace.sft/"source_audit.json")
    if stats["status"] != "verified" or audit["cross_split_leakage"] or audit["invalid_exported_conversations"]:
        raise ValueError("v0.4-A readiness is not trustworthy")
    identity = tokenizer_identity(workspace.tokenizer, stats)
    if max_seq_length > identity["native_context"]:
        raise ValueError("Configured sequence length exceeds native model context")
    cohorts, quality = {}, {}
    for name in ("oracle", "teacher"):
        rows = read_jsonl(workspace.sft/f"{name}_sft.jsonl")
        quality[name] = validate_rows(rows, name, tasks, stats, audit, max_seq_length, identity)
        cohorts[name] = rows
    return SavedTokenDataset(cohorts[source]), {
        "source": source, "quality": quality, "tokenizer_identity": identity,
        "delivery_digest": delivery["content_digest"],
        "selected_file_sha256": sha256(workspace.sft/f"{source}_sft.jsonl"),
        "frozen_manifest_sha256": sha256(workspace.frozen/"manifest.json"),
        "dataset_manifest": manifest, "model_loaded": False, "labels_reused_unchanged": True}
