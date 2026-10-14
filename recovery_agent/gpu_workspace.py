"""GPU preflight and local model inventory; never installs drivers or downloads files."""
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path
from .paths import WorkspacePaths, writable
from .training.data import sha256, json_file
from .data_pipeline.common import fingerprint

def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "peft", "accelerate", "tokenizers", "huggingface-hub", "jinja2", "pyyaml"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result

def model_inventory(directory):
    """Require complete safe local shards and hash each; no hub/network fallback."""
    directory = Path(directory)
    config = directory/"config.json"
    if not config.is_file():
        raise FileNotFoundError("Local model config is missing")
    if json_file(config).get("model_type") != "qwen3":
        raise ValueError("Local base model must be Qwen3")
    index = directory/"model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json_file(index)["weight_map"].values()))
        if not names:
            raise ValueError("Empty weight index")
    elif (directory/"model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        raise FileNotFoundError("Complete local safetensors model weights are missing")
    files = {"config.json": sha256(config)}
    if index.exists():
        files[index.name] = sha256(index)
    for name in names:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Invalid model shard path")
        path = directory/name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty model shard: {name}")
        files[name] = sha256(path)
    return {"files": files, "digest": fingerprint(files), "model_type": "qwen3",
            "weight_bytes": sum((directory/name).stat().st_size for name in names)}

def gpu_environment(workspace, dtype="bf16", min_memory_gib=12, device=0, hash_weights=True):
    """Read-only CUDA/dependency/path checks plus isolated output permission probes."""
    if dtype not in ("bf16", "float32"):
        raise ValueError("Supported dtypes are bf16 or explicitly selected float32")
    report = {"versions": versions(), "paths": workspace.public(), "gpu_verified": False,
              "training_started": False, "blockers": [], "dtype": dtype}
    for name in ("datasets", "outputs", "checkpoints", "model", "cache", "sft", "frozen", "tokenizer"):
        p = getattr(workspace, name)
        if not p.is_dir():
            report["blockers"].append(f"Missing {name}: {p}")
    for name in ("outputs", "checkpoints", "cache"):
        try:
            writable(getattr(workspace, name))
        except OSError as error:
            report["blockers"].append(f"{name} permission check: {type(error).__name__}")
    if workspace.root and os.name != "nt" and not workspace.project.is_relative_to(workspace.root):
        report["blockers"].append("The project must also be located under RECOVERY_DATA_ROOT")
    if not shutil.which("git"):
        report["blockers"].append("git executable missing (isolated Git tasks require it)")
    command = shutil.which("nvidia-smi")
    if command:
        try:
            result = subprocess.run([command, "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            report["nvidia_smi"] = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            report["nvidia_smi"] = "unavailable"
    for dependency in ("torch", "transformers", "peft", "accelerate"):
        if report["versions"][dependency] is None:
            report["blockers"].append(f"Missing dependency: {dependency}")
    if importlib.util.find_spec("torch"):
        try:
            import torch
            report["cuda_runtime"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            if not torch.cuda.is_available():
                report["blockers"].append("CUDA is unavailable")
            elif not 0 <= device < torch.cuda.device_count():
                report["blockers"].append("Selected CUDA device does not exist")
            else:
                torch.cuda.set_device(device)
                properties = torch.cuda.get_device_properties(device)
                report["gpu"] = {"device": device, "name": properties.name,
                                 "total_memory_gib": properties.total_memory/1024**3,
                                 "free_memory_gib": torch.cuda.mem_get_info(device)[0]/1024**3,
                                 "bf16_supported": torch.cuda.is_bf16_supported()}
                if report["gpu"]["free_memory_gib"] < min_memory_gib:
                    report["blockers"].append("Insufficient free VRAM for configured conservative minimum")
                if dtype == "bf16" and not report["gpu"]["bf16_supported"]:
                    report["blockers"].append("BF16 unsupported: explicitly choose float32 and recheck VRAM")
        except Exception as error:
            report["blockers"].append(f"CUDA probe failed: {type(error).__name__}: {error}")
    else:
        report["cuda_available"] = False
        report["blockers"].append("No PyTorch CUDA probe was performed")
    try:
        report["model_inventory"] = model_inventory(workspace.model) if hash_weights else None
    except (OSError, ValueError, KeyError) as error:
        report["blockers"].append(str(error))
    report["gpu_verified"] = not report["blockers"]
    return report
