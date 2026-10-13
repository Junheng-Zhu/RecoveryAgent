"""Offline termination diagnostics using the unchanged full verifier and recorded-action Replay."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from ..data_pipeline.common import read_jsonl,write_json,write_jsonl
from .environment import verify
from .runner import replay

BUDGET_REASONS={"tool_budget","request_budget","global_api_budget"}

def diagnose_episode(task,trace,event_observer=None):
    # The Replay observer sees private state only here, never in a predictor prompt.
    steps=0
    first=None
    prefixes=[]
    def observe(env,event):
        nonlocal steps,first
        if event is not None and event["phase"]=="recovery":
            steps+=1
        verdict=verify(task,env)
        if first is None and verdict["success"]:
            first=steps
        prefixes.append({"event_count":len(prefixes),"recovery_step":steps,"verification":verdict})
        if event_observer:
            event_observer(env,event,verdict)
    replay(task,trace,state_observer=observe)
    reason=trace["termination_reason"]
    configured=trace.get("termination_protocol",{}).get("stop_on_verified_success",False)
    assisted=bool(configured or trace.get("stop_on_verified_success",False) or reason=="runtime_verified_success")
    decisions=trace.get("decisions",[])
    model_stopped=bool(reason=="assistant_text" and decisions and decisions[-1]["parsed"]["kind"]=="stop")
    final=trace["verification"]["success"]
    # Correct stop measures a successful model-ended episode, not zero diagnostic calls.
    autonomous=bool(not assisted and model_stopped and final)
    return {"task_id":task["id"],"model_id":trace.get("model_id"),
            "prompt_version":trace.get("termination_protocol",{}).get("prompt_version","legacy"),
            "stop_regime":"runtime_assisted" if assisted else "model_autonomous",
            "first_goal_achieved_step":first,"post_success_tool_calls":steps-first if first is not None else 0,
            "ever_full_success":first is not None,"final_full_success":bool(final),
            "model_stopped":model_stopped,"model_stop_correct":autonomous,
            "success_before_budget_exhaustion":first is not None,
            "autonomous_completion":autonomous,"termination_reason":reason,
            "stop_type":"runtime_verified" if reason=="runtime_verified_success" else
                        "model_text" if model_stopped else "budget_forced" if reason in BUDGET_REASONS else "other",
            "recovery_steps":steps,"budget_forced":reason in BUDGET_REASONS,
            "ledger_violation_observed":any(p["verification"]["unauthorized_modifications"] or
                                          p["verification"]["extra_side_effects"] for p in prefixes),
            "replay_verified":True,"verifier_prefixes":prefixes}

def _ratio(n,d):
    return {"numerator":n,"denominator":d,"value":n/d if d else None}

def summarize_termination(rows):
    # Runtime-assisted and model-autonomous protocols always have separate denominators.
    groups={}
    for regime in ("model_autonomous","runtime_assisted"):
        subset=[r for r in rows if r["stop_regime"]==regime]
        reached=[r for r in subset if r["ever_full_success"]]
        first_sum=sum(r["first_goal_achieved_step"] for r in reached)
        post=sum(r["post_success_tool_calls"] for r in subset)
        active=regime=="model_autonomous"
        groups[regime]={"episode_count":len(subset),"final_successes":sum(r["final_full_success"] for r in subset),
                       "ever_successes":len(reached),"termination_reasons":dict(Counter(r["termination_reason"] for r in subset)),
                       "first_goal_achieved_step":{"sum":first_sum,"denominator":len(reached),"mean":first_sum/len(reached) if reached else None},
                       "post_success_tool_calls":{"total":post,"denominator":len(reached),"mean":post/len(reached) if reached else None},
                       "model_stop_accuracy":_ratio(sum(r["model_stop_correct"] for r in reached),len(reached)) if active else _ratio(0,0),
                       "success_before_budget_exhaustion":_ratio(len(reached),len(subset)),
                       "autonomous_completion_rate":_ratio(sum(r["autonomous_completion"] for r in subset),len(subset)) if active else _ratio(0,0)}
    return {"episode_count":len(rows),"replays_verified":sum(r["replay_verified"] for r in rows),"groups":groups,
            "definitions":{
              "first_goal_achieved_step":"First full state+audit success; 0 means initial/failure prefix already satisfies the goal.",
              "post_success_tool_calls":"Executed recovery calls after first full success, including calls in the same model batch; zero if never successful. Not all are proven unnecessary.",
              "model_stop_accuracy":"Among ever-successful model-autonomous episodes, fraction ending with parsed assistant text and final full success; late stopping is allowed and measured separately by post-success calls.",
              "success_before_budget_exhaustion":"Any full success in a legally executed prefix before terminal handling; includes success on the last permitted call. Denominator: all episodes in the same stop regime.",
              "autonomous_completion_rate":"Parsed assistant-text termination with final full success / all model-autonomous episodes. Runtime-assisted episodes are excluded.",
              "runtime_assisted":"Enabled experiments and runtime_verified_success records form a separate group; model-stop/autonomous rates are not applicable."}}

def diagnose_directory(directory,output):
    # No models, credentials or network. Existing artifacts remain byte-for-byte untouched.
    directory,output=Path(directory),Path(output)
    if output.resolve()==directory.resolve() or output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Diagnostic output must be separate from the original collection")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Diagnostic output already exists")
    tasks=read_jsonl(directory/"tasks.jsonl");traces=read_jsonl(directory/"trajectories.jsonl")
    by_id={t["id"]:t for t in tasks}
    if len(by_id)!=len(tasks) or len({t["task_id"] for t in traces})!=len(traces):
        raise ValueError("Duplicate task/episode ID")
    rows=[diagnose_episode(by_id[t["task_id"]],t) for t in traces]
    report=summarize_termination(rows)
    report.update(diagnostic_version="v0.3-C.1",source=str(directory),offline=True,
                  source_hashes={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                 for name in ("tasks.jsonl","trajectories.jsonl","session.json") if (directory/name).exists()})
    write_jsonl(output/"episodes.jsonl",rows);write_json(output/"summary.json",report)
    table=["# Offline termination diagnosis v0.3-C.1","",
           "| Task | First goal step | Post-success calls | Ever success | Final success | Stop reason | Autonomous |",
           "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for r in rows:
        table.append(f"| {r['task_id']} | {r['first_goal_achieved_step'] if r['first_goal_achieved_step'] is not None else 'never'} | {r['post_success_tool_calls']} | {r['ever_full_success']} | {r['final_full_success']} | {r['termination_reason']} | {r['autonomous_completion']} |")
    table+=["","Metrics and explicit denominators:","", chr(96)*3+"json",json.dumps(report["groups"],indent=2),chr(96)*3,"",
            "This is offline recorded-action Replay. No model requests were issued. Full success includes the complete side-effect ledger."]
    (output/"report.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    return report

def compare_stop_protocols(first,second,*additional):
    # Only prompt versions may differ; assisted deployments remain a separate, fixed regime.
    from .protocol import STOP_PROTOCOLS
    directories=(first,second)+additional
    if len(directories)>3:
        raise ValueError("At most three prompt protocols can be compared")
    normalized=[];versions=[];assisted=[];returned=[];tasks=[]
    for directory in directories:
        directory=Path(directory)
        value=json.loads((directory/"session.json").read_text(encoding="utf-8"))
        policy=value["session"].pop("termination_protocol",{})
        version=policy.get("prompt_version","legacy")
        if version not in STOP_PROTOCOLS:
            raise ValueError("Unknown stop protocol")
        versions.append(version);assisted.append(bool(policy.get("stop_on_verified_success",False)))
        normalized.append(value)
        records=read_jsonl(directory/"trajectories.jsonl")
        if any(t.get("termination_protocol",{}).get("prompt_version","legacy")!=version or
               bool(t.get("stop_on_verified_success",False) or t.get("termination_protocol",{}).get("stop_on_verified_success",False))!=assisted[-1]
               for t in records if t.get("decisions")):
            raise ValueError("Recorded trajectory stop protocol differs from session")
        returned.append({t["model_id"] for t in records if t.get("decisions")})
        tasks.append(read_jsonl(directory/"tasks.jsonl"))
    if len(set(versions))!=len(versions):
        raise ValueError("Compare distinct prompt versions")
    if any(x!=normalized[0] for x in normalized[1:]) or len(set(assisted))!=1 or any(x!=tasks[0] for x in tasks[1:]):
        raise ValueError("Task/model/budget/decoding/schema/seed/runtime configuration differs")
    if any(x!=returned[0] for x in returned[1:]):
        raise ValueError("Returned model versions differ")
    return {"compatible":True,"prompt_versions":versions,
            "stop_regime":"runtime_assisted" if assisted[0] else "model_autonomous"}
