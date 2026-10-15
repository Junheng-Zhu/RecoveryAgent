"""Project-relative paths and optional /data workspace; no process-wide chdir."""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

KAGGLE_MODEL_PATH = Path("/kaggle/input/models/zhuzhu135/qwen3-1-7b/transformers/default/3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(value, project_root=PROJECT_ROOT):
    """Resolve relative CLI/config paths against the project, independent of terminal cwd."""
    path = Path(value).expanduser()
    if os.name != "nt" and PureWindowsPath(str(value)).drive:
        raise ValueError("A Windows drive path cannot be used on Linux; configure a native path")
    return path.resolve() if path.is_absolute() else (Path(project_root)/path).resolve()

@dataclass(frozen=True)
class WorkspacePaths:
    project: Path
    root: Path | None
    datasets: Path
    outputs: Path
    checkpoints: Path
    model: Path
    cache: Path
    tokenizer: Path
    sft: Path
    frozen: Path
    delivery: Path

    @classmethod
    def from_env(cls, environ=None, project_root=PROJECT_ROOT):
        """Repository data defaults work after cloning; legacy /data layout requires explicit opt-in."""
        env = os.environ if environ is None else environ
        project = Path(project_root).resolve()
        storage = project_path(env["RECOVERY_DATA_ROOT"], project) if env.get("RECOVERY_DATA_ROOT") else None
        def choose(name, local, server):
            default = storage/server if storage else project/local
            return project_path(env.get(name) or default, project)
        datasets = choose("RECOVERY_DATASET_PATH", "data", "datasets/recovery_agent")
        outputs = choose("RECOVERY_OUTPUT_PATH", "outputs", "outputs/recovery_agent")
        checkpoints = choose("RECOVERY_CHECKPOINT_PATH", "checkpoints", "checkpoints/recovery_agent")
        model = project_path(env.get("RECOVERY_MODEL_PATH") or (
            storage/"models/Qwen3-1.7B" if storage else KAGGLE_MODEL_PATH), project)
        cache = choose("RECOVERY_CACHE_PATH", "cache/huggingface", "cache/huggingface")
        # CPU data validation can use the existing tokenizer-only directory, never weights.
        tokenizer = project_path(env.get("RECOVERY_TOKENIZER_PATH") or (
            model if storage else project/"assets/tokenizers/Qwen3-1.7B"), project)
        sft = project_path(env.get("RECOVERY_SFT_PATH") or (
            datasets/"v0.4a_training_readiness" if storage else datasets/"v0.4a_training_readiness"), project)
        frozen = project_path(env.get("RECOVERY_FROZEN_PATH") or datasets/"v0.3b_seed42_final", project)
        delivery = project_path(env.get("RECOVERY_DELIVERY_MANIFEST") or (
            datasets/"delivery_manifest.json" if storage else datasets/"delivery_manifest.json"), project)
        value = cls(project, storage, datasets, outputs, checkpoints, model, cache, tokenizer, sft, frozen, delivery)
        if storage:
            for name in ("datasets", "outputs", "checkpoints", "model", "cache", "tokenizer", "sft", "frozen", "delivery"):
                if not getattr(value, name).is_relative_to(storage):
                    raise ValueError(f"{name} must remain under RECOVERY_DATA_ROOT")
        return value

    def public(self):
        return {name: str(value) if value is not None else None for name, value in vars(self).items()}

    def initialize(self):
        """Idempotent directory creation only; never copy, migrate or replace data."""
        for directory in (self.datasets, self.outputs, self.checkpoints, self.model, self.cache):
            # Kaggle mounted model weights are read-only inputs.
            if directory == self.model and directory.is_relative_to(Path("/kaggle/input")):
                continue
            directory.mkdir(parents=True, exist_ok=True)
            writable(directory)
        return self.public()

    def legacy_cli_path(self, value):
        """Old data/outputs prefixes map to configured stores; all other relative paths use project."""
        parts = Path(value).parts
        if self.root and not Path(value).is_absolute() and parts:
            if parts[0] == "data":
                return self.datasets.joinpath(*parts[1:]).resolve()
            if parts[0] == "outputs":
                return self.outputs.joinpath(*parts[1:]).resolve()
        return project_path(value, self.project)

def resolve_cli_paths(args, fields):
    """Small CLI-only adapter; underlying Runtime and data interfaces are untouched."""
    workspace = WorkspacePaths.from_env()
    for field in fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, project_path(value, workspace.project))
    return args

def writable(directory):
    """Verify real write permission with an isolated temporary file."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    with tempfile.TemporaryFile(dir=directory):
        pass
    return True


def default_cli_path(value):
    """Configured store defaults; explicitly supplied relative paths remain project-relative."""
    return WorkspacePaths.from_env().legacy_cli_path(value)
