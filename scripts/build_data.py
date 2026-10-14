import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery_agent.paths import default_cli_path
from recovery_agent.data_pipeline.dataset import DEFAULT_CONFIG
from recovery_agent.data_pipeline.pipeline import run_pipeline, compare_directories
from recovery_agent.data_pipeline.quality import validate_directory

# 提供 build/validate/compare 三个入口；加载保存的完整配置可以精确复现同一数据版本。
def main():
    parser = argparse.ArgumentParser(description="RecoveryAgent v0.2 deterministic data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", default=default_cli_path("data/v0.2_seed42"))
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--config", help="Optional complete JSON config; overrides --seed")
    build.add_argument("--preference-decisions", type=int, default=1)
    validate = sub.add_parser("validate")
    validate.add_argument("--input", default=default_cli_path("data/v0.2_seed42"))
    compare = sub.add_parser("compare")
    compare.add_argument("--first", required=True)
    compare.add_argument("--second", required=True)
    args = parser.parse_args()
    from recovery_agent.paths import resolve_cli_paths
    resolve_cli_paths(args, ['output', 'input', 'first', 'second', 'config'])
    if args.command == "build":
        config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else copy.deepcopy(DEFAULT_CONFIG)
        if not args.config:
            config["seed"] = args.seed
        report, _, quality = run_pipeline(args.output, config, args.preference_decisions,
                                          lambda message: print(message, flush=True))
        print(json.dumps({"tasks": report["task_statistics"],
                          "sft": report["sft_statistics"],
                          "preferences": report["preference_statistics"],
                          "quality": quality}, indent=2))
    elif args.command == "validate":
        print(json.dumps(validate_directory(args.input), indent=2))
    else:
        print(json.dumps(compare_directories(args.first, args.second), indent=2))

if __name__ == "__main__":
    main()
