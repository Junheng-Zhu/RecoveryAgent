"""Prepare or explicitly execute Base/Adapter evaluation; Mock is a Runtime test, not model performance."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_agent.paths import WorkspacePaths, project_path
from recovery_agent.training.local_model import LocalConfig
from recovery_agent.training.evaluation import evaluate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("dry-run", "mock", "gpu"), default="dry-run")
    parser.add_argument("--model-mode", choices=("base", "adapter"), default="base")
    parser.add_argument("--model-path")
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=("train_smoke", "iid_validation", "iid_test", "ood"), default="train_smoke")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-protocol", default="termination-v0.3c-2")
    parser.add_argument("--stop-on-verified-success", action="store_true")
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    workspace = WorkspacePaths.from_env()
    if args.model_path:
        workspace = replace(workspace, model=project_path(args.model_path))
    adapter = str(project_path(args.adapter)) if args.adapter else None
    if workspace.root and any(not p.is_relative_to(workspace.root) for p in
                              [workspace.model]+([Path(adapter)] if adapter else [])):
        parser.error("Model/Adapter must remain under RECOVERY_DATA_ROOT")
    config = LocalConfig(model_mode=args.model_mode, adapter=adapter, temperature=args.temperature,
                         max_tokens=args.max_tokens, max_requests=args.max_requests,
                         max_total_tokens=args.max_total_tokens, seed=args.seed, dtype=args.dtype)
    output = project_path(args.output) if args.output else workspace.outputs/f"{args.model_mode}_{args.split}_{args.execution}"
    if workspace.root and not output.is_relative_to(workspace.root):
        parser.error("Output must remain under RECOVERY_DATA_ROOT")
    report = evaluate(workspace, config, args.split, output, args.execution, args.seed, args.max_tool_calls,
                      args.stop_protocol, args.stop_on_verified_success, args.resume, args.device)
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
