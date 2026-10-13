import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recovery_agent.hardening.tasks import freeze,validate_frozen
from recovery_agent.hardening.protocol import schema_hash,STOP_PROTOCOLS
from recovery_agent.hardening.batch import collect,compare_protocols,review_collection
from recovery_agent.hardening.training import export
from recovery_agent.hardening.quality import validate_collection,import_legacy
from recovery_agent.stateful.collector import sample_tasks
from recovery_agent.data_pipeline.common import read_jsonl

def main():
    # Network is disabled by default; --online explicitly enables one or user-triggered many episodes.
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("freeze");build.add_argument("--output",default="data/v0.3b_seed42_final")
    build.add_argument("--variants",type=int,default=6);build.add_argument("--seed",type=int,default=42)
    run=sub.add_parser("collect")
    run.add_argument("--dataset",default="data/v0.3b_seed42_final")
    run.add_argument("--output",default="outputs/v0.3b_mock")
    run.add_argument("--stop-protocol",choices=STOP_PROTOCOLS,default="legacy")
    run.add_argument("--stop-on-verified-success",action="store_true")
    run.add_argument("--online",action="store_true");run.add_argument("--resume",action="store_true")
    run.add_argument("--allow-floating-model",action="store_true")
    run.add_argument("--count",type=int);run.add_argument("--task-id")
    run.add_argument("--group",choices=("train","iid_validation","iid_test","ood","all"),default="train")
    run.add_argument("--clean",action="store_true");run.add_argument("--clean-kind",choices=("workflow","control"),default="workflow");run.add_argument("--temperature",type=float);run.add_argument("--seed",type=int,default=42)
    run.add_argument("--max-tool-calls",type=int,default=12);run.add_argument("--episode-requests",type=int,default=16)
    run.add_argument("--max-requests",type=int,default=8);run.add_argument("--max-tokens",type=int,default=4096)
    run.add_argument("--max-total-tokens",type=int,default=80000);run.add_argument("--max-cost-cny",type=float,default=3)
    run.add_argument("--timeout",type=float,default=90)
    e=sub.add_parser("export");e.add_argument("--input",required=True);e.add_argument("--output",required=True)
    e.add_argument("--legacy-input")
    v=sub.add_parser("validate");v.add_argument("--dataset",default="data/v0.3b_seed42_final")
    v.add_argument("--input",required=True);v.add_argument("--export")
    review=sub.add_parser("review");review.add_argument("--input",required=True);review.add_argument("--output",required=True)
    compare=sub.add_parser("compare");compare.add_argument("--first",required=True);compare.add_argument("--second",required=True)
    args=parser.parse_args()
    if args.command=="freeze":
        tasks,manifest=freeze(args.output,args.variants,args.seed,schema_hash())
        print(json.dumps({"tasks":len(tasks),"schema_hash":manifest["schema_hash"]},indent=2))
    elif args.command=="collect":
        tasks,_=validate_frozen(args.dataset,schema_hash())
        if args.task_id:
            selected=[t for t in tasks if t["id"]==args.task_id]
            if not selected: parser.error("Unknown task ID")
        else:
            selected=tasks if args.group=="all" else [t for t in tasks if t["eval_group"]==args.group]
            if args.online and args.group=="train" and args.count:
                if args.count>len(selected): parser.error("Not enough independent train instances")
                selected=sample_tasks(selected,args.count,args.seed,"train")
            elif args.count is not None:
                if args.count<1 or args.count>len(selected): parser.error("Invalid task count")
                selected=selected[:args.count]
        if args.online and len(selected)>1 and args.count is None:
            parser.error("Online collection needs explicit --count or --task-id")
        _,_,report=collect(selected,args.output,online=args.online,resume=args.resume,seed=args.seed,clean=args.clean,
                          max_tool_calls=args.max_tool_calls,episode_requests=args.episode_requests,
                          max_requests=args.max_requests,max_tokens=args.max_tokens,max_total_tokens=args.max_total_tokens,
                          max_cost_cny=args.max_cost_cny,timeout=args.timeout,allow_floating=args.allow_floating_model,clean_kind=args.clean_kind,temperature=args.temperature,
                          stop_protocol=args.stop_protocol,stop_on_verified_success=args.stop_on_verified_success)
        report.pop("by_task",None)
        print(json.dumps(report,indent=2))
    elif args.command=="export":
        tasks=read_jsonl(Path(args.input)/"tasks.jsonl");traces=read_jsonl(Path(args.input)/"trajectories.jsonl")
        print(json.dumps(export(tasks,traces,args.output),indent=2))
        if args.legacy_input:
            print(json.dumps(import_legacy(args.legacy_input,Path(args.output)/"legacy_v03a"),indent=2))
    elif args.command=="review":
        report=review_collection(args.input,args.output);report.pop("by_task",None);print(json.dumps(report,indent=2))
    elif args.command=="validate":
        print(json.dumps(validate_collection(args.dataset,args.input,args.export),indent=2))
    else:
        print(json.dumps({"comparable":compare_protocols(args.first,args.second)}))
if __name__=="__main__":
    main()
