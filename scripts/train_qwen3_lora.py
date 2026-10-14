"""Separate CPU data dry-run, GPU one-batch smoke, and explicit full LoRA training."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_agent.paths import WorkspacePaths
from recovery_agent.training.trainer import load_config, cpu_dry_run, gpu_train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_1.7b_oracle_lora.yaml")
    parser.add_argument("--mode", choices=("dry-run", "smoke", "train"), default="dry-run")
    parser.add_argument("--source", choices=("oracle", "teacher"))
    parser.add_argument("--model-path")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--resume", help="A complete local checkpoint; full training only")
    parser.add_argument("--dtype", choices=("bf16", "float32"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.resume and args.mode != "train":
        parser.error("--resume is only valid for explicit full training")
    workspace = WorkspacePaths.from_env()
    config, workspace = load_config(args.config, workspace, args.source, args.model_path,
                                     args.output, args.checkpoint_path)
    if args.dtype:
        config["training"]["dtype"] = args.dtype
    if args.mode == "dry-run":
        output = args.output or workspace.outputs/(config["output"]["run_name"]+"_cpu_dryrun")
        report = cpu_dry_run(config, workspace, output)
    else:
        if args.mode == "smoke":
            if not args.output:
                config["_paths"]["output"] = str(workspace.outputs/(config["output"]["run_name"]+"_gpu_smoke"))
            if not args.checkpoint_path:
                config["_paths"]["checkpoints"] = str(workspace.checkpoints/(config["output"]["run_name"]+"_gpu_smoke"))
        from recovery_agent.paths import project_path
        resume = project_path(args.resume) if args.resume else None
        report = gpu_train(config, workspace, args.mode == "smoke", resume, args.device)
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
