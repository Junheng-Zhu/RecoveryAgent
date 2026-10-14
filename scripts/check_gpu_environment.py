"""Report GPU readiness without loading a model, downloading or training."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_agent.paths import WorkspacePaths, project_path
from recovery_agent.gpu_workspace import gpu_environment
from recovery_agent.data_pipeline.common import write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--min-memory-gib", type=float, default=12)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.min_memory_gib <= 0:
        parser.error("--min-memory-gib must be positive")
    report = gpu_environment(WorkspacePaths.from_env(), args.dtype, args.min_memory_gib, args.device)
    if args.output:
        output = project_path(args.output)
        if output.exists():
            raise FileExistsError("Preflight report already exists")
        write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["gpu_verified"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
