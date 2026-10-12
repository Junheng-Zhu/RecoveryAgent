import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery_agent.data_pipeline.common import write_jsonl, write_json, read_jsonl
from recovery_agent.stateful.tasks import generate_tasks, VERSION
from recovery_agent.stateful.predictor import load_dotenv, LLMConfig, MockPredictor, OnlinePredictor
from recovery_agent.stateful.collector import collect, sample_tasks, replay, summarize
from recovery_agent.stateful.training import prepare

def main():
    # Network use requires --online; default invocation generates and collects 16 local mock tasks.
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--output", default="outputs/v0.3_mock")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-split", choices=("train", "val", "test", "all"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--episode-requests", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-total-tokens", type=int, default=160000)
    parser.add_argument("--max-cost-cny", type=float, default=10)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    if min(args.count,args.variants,args.max_tool_calls,args.episode_requests) < 1:
        parser.error("counts and limits must be positive")
    tasks = generate_tasks(args.variants, args.seed)
    output = Path(args.output)
    if args.online:
        if args.evaluate_split:
            chosen = tasks if args.evaluate_split == "all" else [t for t in tasks if t["split"] == args.evaluate_split]
            chosen = chosen[:args.count]
        else:
            chosen = sample_tasks(tasks, args.count, args.seed, "train")
        load_dotenv(Path(".env"))
        config = LLMConfig(model=os.getenv("MODEL_NAME", ""), base_url=os.getenv("OPENAI_BASE_URL", ""),
                           temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout,
                           max_requests=args.max_requests, max_total_tokens=args.max_total_tokens,
                           max_cost_cny=args.max_cost_cny)
        predictor = OnlinePredictor(config, os.getenv("OPENAI_API_KEY", ""), online=True,
                                    budget_path=output/"budget.json")
        factory = lambda _: predictor
        session = {"source": "llm", "config": config.public()}
    else:
        chosen = tasks if not args.evaluate_split or args.evaluate_split == "all" else [
            t for t in tasks if t["split"] == args.evaluate_split]
        chosen = chosen[:args.count]
        factory = lambda t: MockPredictor(t)
        session = {"source": "oracle_mock"}
    session.update(seed=args.seed, max_tool_calls=args.max_tool_calls, episode_requests=args.episode_requests)
    traces, report = collect(chosen, factory, output, session, args.resume, args.clean)
    write_jsonl(output/"tasks.jsonl", chosen)
    preparation = prepare(chosen, traces, output/"training_candidates")
    if args.online:
        write_json(output/"budget.json", {"config": predictor.config.public(), "state": predictor.budget.state})
    print(json.dumps({"metrics": report, "training_candidates": preparation}, indent=2))

if __name__ == "__main__":
    main()
