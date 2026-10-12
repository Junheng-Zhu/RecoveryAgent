import json
from collections import Counter
from pathlib import Path
from ..data_pipeline.common import read_jsonl,write_json,fingerprint
from .tasks import validate_frozen
from .protocol import schema_hash
from .training import verify_inputs,validate_pairs
from .runner import replay
from .batch import clean_control,clean_workflow
from ..data_pipeline.serialization import action_message

def validate_collection(dataset,directory,export_dir=None):
    # Validate persisted artifacts and causal inputs; no model requests are issued.
    canonical,manifest=validate_frozen(dataset,schema_hash())
    directory=Path(directory)
    tasks=read_jsonl(directory/"tasks.jsonl")
    traces=read_jsonl(directory/"trajectories.jsonl")
    by_id={t["id"]:t for t in tasks}
    if len(by_id)!=len(tasks) or len({t["task_id"] for t in traces})!=len(traces):
        raise ValueError("Duplicate task/trajectory IDs")
    canonical_ids={t["id"]:t for t in canonical}
    for t in tasks:
        expected=(clean_control if t.get("clean_control") else clean_workflow)(canonical_ids[t["paired_task_id"]]) if t.get("paired_task_id") else canonical_ids.get(t["id"])
        if expected!=t: raise ValueError("Task not in frozen dataset")
    count=0
    for trace in traces:
        task=by_id[trace["task_id"]]
        if not trace.get("events") and trace["verification"]["success"]:
            raise ValueError("Successful trajectory missing evidence")
        if trace.get("events"):
            verify_inputs(task,trace)
            if trace["split"]!=task["split"] or trace["schema_hash"]!=schema_hash():
                raise ValueError("Trajectory protocol metadata mismatch")
            count+=1
    report={"passed":True,"tasks":len(tasks),"records":len(traces),"replays_verified":count,
            "successful":sum(t["verification"]["success"] for t in traces),
            "groups":dict(Counter(t["eval_group"] for t in tasks)),
            "task_template_distribution":dict(Counter(t["template_family"] for t in tasks)),
            "domain_distribution":dict(Counter(t["domain"] for t in tasks)),
            "reference_length_distribution":dict(Counter(len(t["reference_plan"]) for t in tasks)),
            "actual_trajectory_lengths":dict(Counter(t.get("recovery_steps",0) for t in traces)),
            "observed_error_types":dict(Counter(e["result"]["error"]["code"] for t in traces for e in t.get("events",[]) if e["result"]["error"])),
            "per_split":{g:{"tasks":sum(t["eval_group"]==g for t in tasks),
                           "mechanisms":dict(Counter(t["mechanism_id"] for t in tasks if t["eval_group"]==g)),
                           "reference_lengths":dict(Counter(len(t["reference_plan"]) for t in tasks if t["eval_group"]==g))}
                         for g in sorted({t["eval_group"] for t in tasks})},
            "dataset_reproducible":True,"causal_prompt_validation":True,"test_training_samples":0}
    if export_dir:
        export_dir=Path(export_dir)
        samples=[]
        for source in ("llm","oracle_reference"):
            samples+=read_jsonl(export_dir/f"{source}_sft_train.jsonl")
        seen=set()
        for sample in samples:
            task=by_id[sample["task_id"]]
            if by_id[sample["task_id"]]["split"]!="train" or sample["split"]!="train":
                raise ValueError("Evaluation data in training export")
            if sample["loss_scope"]!="final_assistant_only":
                raise ValueError("Unsafe supervision scope")
            key=fingerprint({"messages":sample["messages"],"tools":sample["tools"]})
            if key in seen: raise ValueError("Duplicate SFT sample")
            seen.add(key)
            trace=next(t for t in traces if t["task_id"]==sample["task_id"])
            index=int(sample["sample_id"].rsplit(":",1)[-1])
            decision=trace["decisions"][index]
            if sample["messages"][:-1]!=decision["prompt"]["messages"] or sample["tools"]!=decision["prompt"]["tools"]:
                raise ValueError("SFT prompt differs from verified decision state")
            if (sample["schema_hash"]!=schema_hash() or sample["dataset_version"]!=task["dataset_version"] or
                sample["model_id"]!=trace["model_id"] or sample["prompt_hash"]!=fingerprint(decision["prompt"])):
                raise ValueError("SFT protocol metadata altered")
            parsed=decision["parsed"]
            if parsed["kind"]=="tools":
                target={"role":"assistant","content":None,"tool_calls":[action_message(c["call"])["tool_calls"][0] for c in parsed["calls"]]}
                actual=trace["events"][decision["before_event_count"]:decision["before_event_count"]+len(parsed["calls"])]
                if len(actual)!=len(parsed["calls"]) or any(not e["result"]["ok"] or e["call"]!=c["call"] or e["behavior"]["label"] in ("unsafe","invalid","redundant","non_progress","uncertain") for e,c in zip(actual,parsed["calls"])):
                    raise ValueError("Unexecuted/failed supervised action")
            elif parsed["kind"]=="stop": target={"role":"assistant","content":parsed["text"]}
            else: raise ValueError("Invalid supervised decision")
            source="llm" if trace["source"]=="llm" else "oracle_reference"
            if sample["messages"][-1]!=target or sample["source"]!=source or not trace["verification"]["success"]:
                raise ValueError("Supervision target/provenance differs from verified response")
        pairs=read_jsonl(export_dir/"dpo_candidates.jsonl")
        validate_pairs(tasks,traces,pairs)
        report.update(sft_samples=len(samples),duplicate_samples=0,verified_pairs=len(pairs))
    write_json(directory/"quality_report.json",report)
    return report

def import_legacy(input_dir,output_dir):
    # Preserve v0.3-A files; replay with its original protocol and export into a separate new directory.
    from ..stateful.training import prepare
    input_dir,output_dir=Path(input_dir),Path(output_dir)
    tasks,traces=read_jsonl(input_dir/"tasks.jsonl"),read_jsonl(input_dir/"trajectories.jsonl")
    report=prepare(tasks,traces,output_dir)
    from ..data_pipeline.common import write_jsonl
    by_task={t["task_id"]:t for t in traces}
    for source in ("llm","oracle_mock"):
        path=output_dir/f"{source}_sft_train.jsonl"
        samples=read_jsonl(path)
        for sample in samples:
            trace=by_task[sample["task_id"]]
            sample.update(dataset_version="0.3-A",model_id=trace.get("model_id"),schema_hash=fingerprint(sample["tools"]))
        write_jsonl(path,samples)
    return report
