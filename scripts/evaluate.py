import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery_agent.paths import default_cli_path
from recovery_agent.dataset import load_tasks
from recovery_agent.policy import RulePolicy, RetryPolicy, StopPolicy
from recovery_agent.oracle import Oracle
from recovery_agent.runner import evaluate, replay

# 解析评测策略与 split，执行恢复评测；可选 Replay 并保存验证数量。
# Oracle 使用独立认证模式，不能把其成功率当作普通模型评测结果。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=default_cli_path("data/tasks/smoke.jsonl"))
    parser.add_argument("--policy", choices=("rule", "retry", "stop", "oracle"), default="rule")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--output", default=default_cli_path("outputs/rule"))
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    from recovery_agent.paths import resolve_cli_paths
    resolve_cli_paths(args, ['tasks', 'output'])
    tasks = [t for t in load_tasks(args.tasks) if args.split == "all" or t["split"] == args.split]
    factories = {"rule": lambda _: RulePolicy(), "retry": lambda _: RetryPolicy(),
                 "stop": lambda _: StopPolicy(), "oracle": Oracle}
    mode = "oracle_certification" if args.policy == "oracle" else "evaluation"
    report = evaluate(tasks, factories[args.policy], args.output, mode)
    if args.replay:
        traces = load_tasks(Path(args.output) / "trajectories.jsonl")
        by_id = {t["id"]: t for t in tasks}
        for trace in traces:
            replay(by_id[trace["task_id"]], trace)
        report["replay_verified"] = len(traces)
        (Path(args.output) / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
