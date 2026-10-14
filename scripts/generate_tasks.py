import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery_agent.paths import default_cli_path
from recovery_agent.dataset import generate_tasks, save_tasks

# 解析数量、seed 和输出路径，生成 v0.1 任务及固定模板族的 train/val/test 文件。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=default_cli_path("data/tasks/smoke.jsonl"))
    args = parser.parse_args()
    from recovery_agent.paths import resolve_cli_paths
    resolve_cli_paths(args, ['output'])
    tasks = generate_tasks(args.count, args.seed)
    save_tasks(args.output, tasks)
    base = Path(args.output).parent
    for split in ("train", "val", "test"):
        save_tasks(base / f"{split}.jsonl", [t for t in tasks if t["split"] == split])
    print(f"Generated {len(tasks)} tasks at {args.output}")

if __name__ == "__main__":
    main()
