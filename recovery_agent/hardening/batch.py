import copy
from functools import partial
import json
import os
import re
from pathlib import Path
from ..stateful.collector import collect as shared_collect,sample_tasks
from ..stateful.predictor import load_dotenv,LLMConfig,OnlinePredictor
from ..data_pipeline.common import fingerprint,write_json,write_jsonl,read_jsonl
from .tasks import VERSION
from .runner import run_episode,OracleMock
from .protocol import schema_hash,STOP_PROTOCOLS
from .analysis import summary

def clean_control(task):
    # Counterfactual goal-already-satisfied control, explicitly distinct from a normal clean task.
    t=copy.deepcopy(task)
    t["paired_task_id"]=t["id"];t["id"]+=".clean"
    t["initial_state"]={"files":copy.deepcopy(t["expected"]["files"]),"tables":copy.deepcopy(t["expected"]["tables"])}
    t["sqlite_initial"]=copy.deepcopy(t["expected"]["sqlite"])
    for k in ("dirty_files","post_stage_files","staged_paths","bootstrap_commit","initial_branch"):
        t.pop(k,None)
    t["clean_control"]=True
    t["allowed_resources"]={}
    t["goal"]["instruction"]+=" The requested goal state is already present; diagnose and finish without new mutations."
    t["reference_plan"]=[{"name":"git_status" if t["expected"]["git"] else "execute_select",
                           "arguments":{} if t["expected"]["git"] else {"table":"accounts"}}]
    return t

def clean_workflow(task):
    # Paired normal workflow: omit the failed attempt and injected transient fault, preserve real constraints.
    t=copy.deepcopy(task)
    t["paired_task_id"]=t["id"];t["id"]+=".workflow"
    t["clean_workflow"]=True
    t["fault"]["type"]="native"
    t["goal"]["instruction"]+=" Execute the requested work directly; no previous tool call has failed."
    return t

def collect(tasks,output,online=False,resume=False,seed=42,clean=False,max_tool_calls=12,
            episode_requests=16,max_requests=8,max_tokens=4096,max_total_tokens=80000,max_cost_cny=3,timeout=90,
            allow_floating=False,progress=None,clean_kind="workflow",temperature=None,stop_protocol="legacy",stop_on_verified_success=False):
    # Reuse durable append, ownership lock, resume checks and the existing HTTP/budget implementation.
    if stop_protocol not in STOP_PROTOCOLS: raise ValueError("Unknown stop protocol")
    output=Path(output)
    if (output/"review_manifest.json").exists(): raise ValueError("Derived review directory is not a collector destination")
    if clean: tasks=[(clean_control if clean_kind=="control" else clean_workflow)(t) for t in tasks]
    session={"dataset_version":VERSION,"source":"llm" if online else "oracle_mock",
             "seed":seed,"max_tool_calls":max_tool_calls,"episode_requests":episode_requests,
             "schema_hash":schema_hash(),"clean_protocol":("goal_already_satisfied_control" if clean_kind=="control" else "fault_omitted_workflow") if clean else None}
    if online:
        load_dotenv(Path(".env"))
        model=os.getenv("MODEL_NAME","")
        pinned=bool(re.search(r"(?:-\d{4,8}|-\d{4}-\d{2}-\d{2})$",model))
        if not pinned and not allow_floating:
            raise ValueError("Use an environment MODEL_NAME snapshot, or explicitly --allow-floating-model")
        config=LLMConfig(model=model,base_url=os.getenv("OPENAI_BASE_URL",""),temperature=temperature,max_tokens=max_tokens,
                         max_requests=max_requests,max_total_tokens=max_total_tokens,max_cost_cny=max_cost_cny,timeout=timeout)
        predictor=OnlinePredictor(config,os.getenv("OPENAI_API_KEY",""),online=True,budget_path=output/"budget.json")
        factory=lambda _:predictor
        session.update(config=config.public(),model_version_policy="snapshot" if pinned else "explicit_floating")
    else:
        factory=lambda task:OracleMock(task)
    episode_runner=run_episode
    if stop_protocol!="legacy" or stop_on_verified_success:
        session["termination_protocol"]={"prompt_version":stop_protocol,"stop_on_verified_success":bool(stop_on_verified_success)}
        episode_runner=partial(run_episode,stop_protocol=stop_protocol,stop_on_verified_success=stop_on_verified_success)
    traces,_=shared_collect(tasks,factory,output,session,resume,clean,runner=episode_runner,dataset_version=VERSION)
    write_jsonl(output/"tasks.jsonl",tasks)
    report=summary(traces)
    write_json(output/"report.json",report)
    # Human-review file is an evidence projection, not model input or a training label.
    reviews=[{"task_id":t["task_id"],"call":e["call"],"result":e["result"],
              "behavior":e["behavior"],"states":e["review_states"]}
             for t in traces for e in t.get("events",[]) if e.get("behavior")]
    write_jsonl(output/"call_review.jsonl",reviews)
    write_json(output/"protocol.json",session)
    return tasks,traces,report

def compare_protocols(first,second):
    # Different models may be compared only when data, schema, budgets and decode parameters agree.
    a,b=json.loads((Path(first)/"session.json").read_text()),json.loads((Path(second)/"session.json").read_text())
    aa,bb=copy.deepcopy(a),copy.deepcopy(b)
    for value in (aa,bb):
        cfg=value["session"].get("config",{})
        cfg.pop("model",None);cfg.pop("base_url",None)
    if aa!=bb: raise ValueError("Task/initial-state/schema/budget/decode protocol differs")
    return True

def review_collection(directory,output):
    # Recompute derived behavior evidence into a new directory, preserving raw collection bytes.
    from .analysis import annotate
    import hashlib
    directory,output=Path(directory),Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError("Review output already populated")
    tasks=read_jsonl(directory/"tasks.jsonl");traces=read_jsonl(directory/"trajectories.jsonl")
    by_id={t["id"]:t for t in tasks}
    for trace in traces:
        if trace.get("events"): annotate(by_id[trace["task_id"]],trace)
    write_jsonl(output/"tasks.jsonl",tasks)
    write_jsonl(output/"trajectories.jsonl",traces)
    report=summary(traces);write_json(output/"report.json",report)
    for name in ("protocol.json","session.json"):
        if (directory/name).exists(): (output/name).write_bytes((directory/name).read_bytes())
    reviews=[{"task_id":t["task_id"],"call":e["call"],"result":e["result"],"behavior":e["behavior"],"states":e["review_states"]}
             for t in traces for e in t.get("events",[]) if e.get("behavior")]
    write_jsonl(output/"call_review.jsonl",reviews)
    write_json(output/"review_manifest.json",{"source":str(directory),"source_trace_sha256":hashlib.sha256((directory/"trajectories.jsonl").read_bytes()).hexdigest(),
                "behavior_policy":"evidence_v1","derived_only":True})
    return report
