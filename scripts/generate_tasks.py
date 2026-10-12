import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery_agent.dataset import generate_tasks, save_tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/tasks/smoke.jsonl")
    args = parser.parse_args()
    tasks = generate_tasks(args.count, args.seed)
    save_tasks(args.output, tasks)
    base = Path(args.output).parent
    for split in ("train", "val", "test"):
        save_tasks(base / f"{split}.jsonl", [t for t in tasks if t["split"] == split])
    print(f"Generated {len(tasks)} tasks at {args.output}")

if __name__ == "__main__":
    main()
