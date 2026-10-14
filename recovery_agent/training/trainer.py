"""Single-GPU LoRA training with saved labels, atomic checkpoints and explicit execution modes."""
import copy
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from dataclasses import replace
from ..paths import project_path, writable
from ..data_pipeline.common import fingerprint, write_json, canonical
from ..gpu_workspace import gpu_environment, versions
from .data import (DynamicCollator, load_training_data, json_file, sha256,
                   supervised_count, tokenizer_identity)

def load_config(path, workspace, source=None, model_path=None, output=None, checkpoint_path=None):
    """Validate small YAML configuration; relative paths always resolve against the project."""
    import yaml
    config = yaml.safe_load(project_path(path, workspace.project).read_text(encoding="utf-8"))
    allowed = {"model": {"path", "max_seq_length"}, "lora": {"r", "alpha", "dropout"},
               "training": {"learning_rate", "batch_size", "gradient_accumulation_steps", "epochs",
                            "seed", "gradient_checkpointing", "dtype", "weight_decay", "max_grad_norm", "save_steps"},
               "data": {"source", "path"}, "output": {"run_name", "path", "checkpoint_path"}}
    if not isinstance(config, dict) or set(config) != set(allowed):
        raise ValueError("Training config needs exactly model/lora/training/data/output")
    for name, keys in allowed.items():
        if not isinstance(config[name], dict) or set(config[name])-keys:
            raise ValueError(f"Unknown configuration keys: {name}")
    defaults = {"dtype": "bf16", "weight_decay": 0.0, "max_grad_norm": 1.0, "save_steps": 10}
    for key, value in defaults.items():
        config["training"].setdefault(key, value)
    if source:
        config["data"]["source"] = source
    if config["data"].get("source") not in ("oracle", "teacher"):
        raise ValueError("Data source must be Oracle or Teacher; first run defaults to Oracle")
    for section, keys in (("model", ("max_seq_length",)), ("lora", ("r", "alpha")),
                          ("training", ("batch_size", "gradient_accumulation_steps", "epochs", "save_steps"))):
        if any(type(config[section].get(key)) is not int or config[section][key] <= 0 for key in keys):
            raise ValueError("Lengths, ranks, batches, epochs and save intervals must be positive integers")
    t = config["training"]
    if (type(t.get("seed")) is not int or t["seed"] < 0 or
        type(t.get("gradient_checkpointing")) is not bool or t["dtype"] not in ("bf16", "float32") or
        not 0 < t["learning_rate"] <= 1 or not 0 <= config["lora"]["dropout"] < 1 or
        t["weight_decay"] < 0 or t["max_grad_norm"] <= 0):
        raise ValueError("Invalid training/decode numeric or boolean configuration")
    name = config["output"]["run_name"]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a single directory name")
    resolved_model = project_path(model_path or os.environ.get("RECOVERY_MODEL_PATH") or
                                  config["model"].get("path") or workspace.model, workspace.project)
    sft = project_path(config["data"].get("path") or workspace.sft, workspace.project)
    workspace = replace(workspace, model=resolved_model, sft=sft)
    out = project_path(output or config["output"].get("path") or workspace.outputs/name, workspace.project)
    checkpoints = project_path(checkpoint_path or config["output"].get("checkpoint_path") or
                               workspace.checkpoints/name, workspace.project)
    if workspace.root and any(not p.is_relative_to(workspace.root) for p in (resolved_model, sft, out, checkpoints)):
        raise ValueError("Configured persistent paths must stay under RECOVERY_DATA_ROOT")
    config["_paths"] = {"model": str(resolved_model), "sft": str(sft),
                        "output": str(out), "checkpoints": str(checkpoints)}
    return config, workspace

def batches_for_epoch(size, batch_size, seed, epoch):
    """An independent deterministic sampler is reconstructible from the checkpoint cursor."""
    indices = list(range(size))
    random.Random(seed+epoch).shuffle(indices)
    return [indices[i:i+batch_size] for i in range(0, size, batch_size)]

def resolve_lora_targets(model, linear_type):
    """Inspect every real Qwen3 decoder layer and require each projection to be a Linear module."""
    if model.config.model_type != "qwen3":
        raise ValueError("Only the pinned Qwen3 structure is supported")
    names = dict(model.named_modules())
    targets = []
    scopes = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
              "mlp": ("gate_proj", "up_proj", "down_proj")}
    for index in range(model.config.num_hidden_layers):
        for scope, projections in scopes.items():
            for projection in projections:
                path = f"model.layers.{index}.{scope}.{projection}"
                module = names.get(path)
                if (module is None or not isinstance(module, linear_type) or
                    getattr(module.weight, "ndim", None) != 2):
                    raise ValueError(f"Required real Linear projection is missing/incompatible: {path}")
                targets.append(path)
    if not targets:
        raise ValueError("No verified LoRA target modules")
    return targets

def assistant_cross_entropy(logits, labels, torch):
    """Shift causal targets; compute summed CE only on saved Assistant target positions."""
    shifted = labels[:, 1:].contiguous()
    valid = shifted != -100
    count = int(valid.sum().item())
    if not count:
        raise ValueError("Batch has no effective Assistant supervision")
    selected = logits[:, :-1, :][valid].float()
    targets = shifted[valid]
    loss_sum = torch.nn.functional.cross_entropy(selected, targets, reduction="sum", ignore_index=-100)
    return loss_sum, count

def resume_contract(config, data_report, model_manifest, targets, mode):
    """Machine paths/output names are excluded; actual data, weights, masks and hyperparameters bind resume."""
    return {"version": "v0.4-B", "mode": mode,
            "model_max_seq_length": config["model"]["max_seq_length"],
            "lora": config["lora"], "training": config["training"], "targets": targets,
            "source": config["data"]["source"], "data_sha256": data_report["selected_file_sha256"],
            "delivery_digest": data_report["delivery_digest"],
            "frozen_manifest_sha256": data_report["frozen_manifest_sha256"],
            "tokenizer_identity": data_report["tokenizer_identity"],
            "model_manifest": model_manifest,
            "dependency_versions": {k: v for k, v in versions().items()
                                    if k in ("torch", "transformers", "peft", "tokenizers", "jinja2")}}

def validate_resume(checkpoint, contract):
    """Accept complete, hash-verified checkpoints only; reject source/config/version changes."""
    checkpoint = Path(checkpoint)
    metadata = json_file(checkpoint/"checkpoint_metadata.json")
    if metadata.get("contract") != contract or metadata.get("contract_digest") != fingerprint(contract):
        raise ValueError("Resume configuration/data/model/tokenizer is incompatible")
    for name, digest in metadata.get("artifact_sha256", {}).items():
        if Path(name).name != name or sha256(checkpoint/name) != digest:
            raise ValueError("Checkpoint artifact hash mismatch")
    required = {"adapter_config.json", "adapter_model.safetensors", "trainer_state.pt"}
    if not required.issubset(metadata.get("artifact_sha256", {})):
        raise ValueError("Checkpoint is incomplete")
    p = metadata["progress"]
    if any(type(p.get(k)) is not int or p[k] < 0 for k in ("epoch", "batch_offset", "global_step")):
        raise ValueError("Invalid resume cursor")
    return metadata

def save_checkpoint(directory, model, optimizer, scheduler, progress, contract, torch):
    """Atomic directory rename publishes adapter, optimizer/scheduler/RNG state at an update boundary."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Checkpoint already exists; no overwrite")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name+".tmp-"+uuid.uuid4().hex)
    temporary.mkdir()
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                 "python_rng": random.getstate()}
        torch.save(state, temporary/"trainer_state.pt")
        metadata = {"contract": contract, "contract_digest": fingerprint(contract),
                    "progress": progress,
                    "artifact_sha256": {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}}
        write_json(temporary/"checkpoint_metadata.json", metadata)
        temporary.rename(directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory

def cpu_dry_run(config, workspace, output):
    """No model/torch import: delivery, saved labels, collator and actual permissions only."""
    dataset, report = load_training_data(workspace, config["data"]["source"], config["model"]["max_seq_length"])
    output = project_path(output, workspace.project)
    if output.exists():
        raise FileExistsError("Dry-run output exists; choose a new directory")
    output.mkdir(parents=True)
    workspace.checkpoints.mkdir(parents=True, exist_ok=True)
    writable(output)
    writable(workspace.checkpoints)
    collator = DynamicCollator(report["tokenizer_identity"]["pad_token_id"])
    # Two unequal existing lengths exercise real dynamic padding without modifying source arrays.
    first = dataset[0]
    second_index = next((i for i in range(1, len(dataset)) if len(dataset[i]["input_ids"]) != len(first["input_ids"])), 1)
    batch = collator([first, dataset[second_index]])
    if any(label != -100 for labels, attention in zip(batch["labels"], batch["attention_mask"])
           for label, mask in zip(labels, attention) if mask == 0):
        raise ValueError("Padding participates in loss")
    report.update(status="CPU_DATA_DRY_RUN_PASSED", config=config, paths=workspace.public(),
                  versions=versions(), model_weights_checked=False, gpu_training_verified=False,
                  collator={"batch_shape": [2, len(batch["input_ids"][0])],
                            "padding_labels": -100, "dynamic_padding": True,
                            "supervised_tokens": sum(supervised_count(row) for row in batch["labels"])})
    write_json(output/"dry_run_report.json", report)
    return report

def gpu_train(config, workspace, smoke=False, resume=None, device=0):
    """Explicit GPU-only entry; smoke executes exactly one batch/update and is separate from formal training."""
    os.environ["USE_TORCH"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    dataset, data_report = load_training_data(workspace, config["data"]["source"], config["model"]["max_seq_length"])
    preflight = gpu_environment(workspace, config["training"]["dtype"], device=device)
    if not preflight["gpu_verified"]:
        raise RuntimeError("GPU preflight blocked: "+"; ".join(preflight["blockers"]))
    # On GPU the model's own tokenizer/config files must match the prepared token IDs exactly.
    stats = json_file(workspace.sft/"token_stats.json")
    tokenizer_identity(workspace.model, stats)
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    torch.cuda.set_device(device)
    t = config["training"]
    torch.manual_seed(t["seed"])
    torch.cuda.manual_seed_all(t["seed"])
    random.seed(t["seed"])
    dtype = torch.bfloat16 if t["dtype"] == "bf16" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(str(workspace.model), local_files_only=True,
             trust_remote_code=False, torch_dtype=dtype, attn_implementation="sdpa")
    targets = resolve_lora_targets(base, torch.nn.Linear)
    contract = resume_contract(config, data_report, preflight["model_inventory"], targets, "smoke" if smoke else "train")
    metadata = validate_resume(resume, contract) if resume else None
    model = (PeftModel.from_pretrained(base, str(resume), local_files_only=True, is_trainable=True) if resume else
             get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=config["lora"]["r"],
                lora_alpha=config["lora"]["alpha"], lora_dropout=config["lora"]["dropout"],
                target_modules=targets, bias="none")))
    model.to(torch.device("cuda", device))
    model.config.use_cache = False
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Only verified LoRA parameters may be trainable")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=t["learning_rate"], weight_decay=t["weight_decay"])
    batch_count = math.ceil(len(dataset)/t["batch_size"])
    steps_per_epoch = math.ceil(batch_count/t["gradient_accumulation_steps"])
    total_steps = 1 if smoke else steps_per_epoch*t["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1-step/max(1, total_steps)))
    progress = metadata["progress"] if metadata else {"epoch": 0, "batch_offset": 0, "global_step": 0}
    if progress["epoch"] > t["epochs"] or progress["batch_offset"] > batch_count or progress["global_step"] > total_steps:
        raise ValueError("Resume cursor exceeds configured schedule")
    if metadata:
        # Load only a local hash-verified tensor/primitive state, not a remote pickle.
        state = torch.load(Path(resume)/"trainer_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
    output = Path(config["_paths"]["output"])
    checkpoints = Path(config["_paths"]["checkpoints"])
    if output.exists() and any(output.iterdir()):
        if not resume or json_file(output/"run_contract.json") != contract:
            raise FileExistsError("Run output exists or resume contract differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/"run_contract.json", contract)
        write_json(output/"training_config.json", config)
        write_json(output/"data_manifest.json", json_file(workspace.delivery))
        write_json(output/"gpu_preflight.json", preflight)
    checkpoints.mkdir(parents=True, exist_ok=True)
    writable(output)
    writable(checkpoints)
    attempt = uuid.uuid4().hex
    model.train()
    collator = DynamicCollator(data_report["tokenizer_identity"]["pad_token_id"], tensors=True)
    torch.cuda.reset_peak_memory_stats(device)
    completed_tokens = 0
    for epoch in range(progress["epoch"], 1 if smoke else t["epochs"]):
        batches = batches_for_epoch(len(dataset), t["batch_size"], t["seed"], epoch)
        offset = progress["batch_offset"] if epoch == progress["epoch"] else 0
        while offset < len(batches) and progress["global_step"] < total_steps:
            window_size = 1 if smoke else t["gradient_accumulation_steps"]
            window = batches[offset:offset+window_size]
            denominator = sum(supervised_count(dataset[i]["labels"]) for batch in window for i in batch)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            observed, tokens = 0.0, 0
            before = [p.detach().clone() for _, p in trainable] if smoke else None
            for indices in window:
                batch = {k: v.to(torch.device("cuda", device)) for k, v in collator([dataset[i] for i in indices]).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t["dtype"] == "bf16"):
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   use_cache=False).logits
                    loss_sum, count = assistant_cross_entropy(logits, batch["labels"], torch)
                if not torch.isfinite(loss_sum):
                    raise ValueError("Non-finite training loss")
                (loss_sum/denominator).backward()
                observed += float(loss_sum.detach())
                tokens += count
                del logits, batch, loss_sum
            if tokens != denominator:
                raise ValueError("Gradient accumulation token denominator mismatch")
            grads = [p.grad for _, p in trainable if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads) or not any(bool((g != 0).any()) for g in grads):
                raise ValueError("Missing/non-finite/zero LoRA gradients")
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], t["max_grad_norm"])
            if not torch.isfinite(norm):
                raise ValueError("Non-finite gradient norm")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            if smoke and not any(not torch.equal(old, p.detach()) for old, (_, p) in zip(before, trainable)):
                raise ValueError("Optimizer did not update LoRA parameters")
            offset += len(window)
            progress = {"epoch": epoch+1 if offset == len(batches) else epoch,
                        "batch_offset": 0 if offset == len(batches) else offset,
                        "global_step": progress["global_step"]+1}
            completed_tokens += tokens
            metric = {"attempt_id": attempt, "step": progress["global_step"], "loss": observed/tokens,
                      "learning_rate": lr, "effective_supervised_tokens": tokens,
                      "gradient_norm": float(norm), "step_time_seconds": time.perf_counter()-started,
                      "gpu_allocated_bytes": torch.cuda.memory_allocated(device),
                      "gpu_reserved_bytes": torch.cuda.memory_reserved(device),
                      "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)}
            with (output/"train_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical(metric)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(canonical(metric), flush=True)
            if smoke or progress["global_step"] % t["save_steps"] == 0 or progress["global_step"] == total_steps:
                save_checkpoint(checkpoints/f"step-{progress['global_step']:06d}", model, optimizer,
                                scheduler, progress, contract, torch)
    adapter = output/"adapter"
    if adapter.exists():
        raise FileExistsError("Final adapter already exists; use a separate run")
    model.save_pretrained(adapter, safe_serialization=True)
    write_json(adapter/"run_contract.json", contract)
    write_json(adapter/"tokenizer_hash.json", data_report["tokenizer_identity"])
    write_json(adapter/"data_manifest.json", json_file(workspace.delivery))
    write_json(adapter/"training_config.json", config)
    report = {"mode": "gpu_one_step_smoke" if smoke else "oracle_sft" if config["data"]["source"] == "oracle" else "teacher_sft",
              "gpu_verified": True, "optimizer_steps": progress["global_step"],
              "supervised_tokens_this_invocation": completed_tokens,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
              "adapter": str(adapter), "contract_digest": fingerprint(contract)}
    write_json(output/"training_report.json", report)
    return report
