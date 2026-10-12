import copy
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from ..models import ToolCall
from ..data_pipeline.common import canonical,fingerprint,write_json,write_jsonl,read_jsonl
from ..data_pipeline.serialization import action_message,stop_message
from .environment import HardenedEnvironment,verify
from .protocol import prompt_for,schema_hash
from .tasks import VERSION,Oracle
from .runner import replay

def verify_inputs(task,trace):
    # Reconstruct each causal prompt exactly, including parser/API feedback but excluding future results.
    replay(task,trace)
    from .analysis import annotate
    checked=copy.deepcopy(trace)
    annotate(task,checked)
    if any(old.get("behavior")!=new.get("behavior") or old.get("review_states")!=new.get("review_states")
           for old,new in zip(trace["events"],checked["events"])):
        raise ValueError("Behavior evidence differs from actual execution")
    if not trace.get("events") and trace.get("verification",{}).get("success"):
        raise ValueError("Successful episode missing execution evidence")
    feedback=None
    for decision in trace.get("decisions",[]):
        prefix=trace["events"][:decision["before_event_count"]]
        used=sum(e["phase"]=="recovery" for e in prefix)
        remaining=min(trace["collection_limits"]["max_tool_calls"],task["max_recovery_steps"])-used
        expected=prompt_for(task,prefix,remaining,feedback)
        if decision["prompt"]!=expected:
            raise ValueError("Prompt contains unverified fields/future information")
        parsed=decision["parsed"];feedback=None
        if decision.get("error"):
            feedback=decision["error"]
        elif parsed["kind"]=="invalid":
            feedback=parsed["error"]
        elif parsed["kind"]=="tools":
            if len(parsed["calls"])>remaining:
                feedback="tool_batch_exceeds_remaining_budget"
            elif any(c["call"] and c["call"]["call_id"] in {e["call"]["call_id"] for e in prefix} for c in parsed["calls"]):
                feedback="tool_call_id_reused"
            else:
                for call in parsed["calls"]:
                    if call["error"]: feedback=call["error"]
    return True

def physical(env):
    return fingerprint({"state":env.snapshot(),"audit":env.audit,"fault_fired":env.injector.fired})

def reference_states(task):
    # Trusted alternatives are matched by physical state+ledger, not by model step index.
    with HardenedEnvironment(task) as env:
        events=[]
        first=ToolCall(**task["failed_action"],call_id="initial")
        result=env.execute(first).to_dict()
        events.append({"call":asdict(first),"result":result})
        oracle=Oracle(task);states=[]
        while True:
            key=physical(env);index=oracle.index
            action=oracle.next_action(events)
            states.append({"state":key,"action":action,"index":index,"events":copy.deepcopy(events)})
            if action is None: break
            call=ToolCall(**action,call_id=f"reference-{index}")
            result=env.execute(call).to_dict()
            if not result["ok"]: raise ValueError("Reference action failed")
            events.append({"call":asdict(call),"result":result})
        if not verify(task,env)["success"]: raise ValueError("Reference goal failed")
        return states

def branch(task,prefix,candidate,reference_index,remaining):
    # Both branches replay the model's exact prefix; a fixed continuation failure alone is uncertain.
    with HardenedEnvironment(task) as env:
        events=copy.deepcopy(prefix)
        for event in prefix:
            if env.execute(ToolCall(**event["call"])).to_dict()!=event["result"]:
                raise ValueError("Candidate prefix diverged")
        start=physical(env);executed=[]
        if candidate:
            call=ToolCall(**candidate)
            result=env.execute(call).to_dict()
            event={"call":asdict(call),"result":result}
            events.append(event);executed.append(event)
        immediate=verify(task,env)
        if candidate:
            oracle=Oracle(task);oracle.index=reference_index+1
            for _ in range(max(0,remaining-1)):
                try: action=oracle.next_action(events)
                except (KeyError,TypeError,ValueError): break
                if action is None: break
                call=ToolCall(**action,call_id=f"continuation-{len(executed)}")
                result=env.execute(call).to_dict()
                event={"call":asdict(call),"result":result}
                events.append(event);executed.append(event)
        return {"start_state_id":start,"candidate":candidate,"remaining_steps":remaining,
                "candidate_verification":immediate,"final_verification":verify(task,env),
                "executed":executed,"final_state":env.snapshot(),"audit":copy.deepcopy(env.audit)}

def preferences(task,trace):
    # Real model errors only. Every decision is eligible; mock/reference responses are not negatives.
    if task["split"]!="train" or trace["source"]!="llm":
        return [],{"not_real_train":1},[]
    states=reference_states(task);pairs=[];dropped=Counter();uncertain=[]
    for decision in trace.get("decisions",[]):
        parsed=decision["parsed"]
        if parsed["kind"] not in ("tools","stop") or len(parsed["calls"])>1:
            dropped["malformed_or_multi_action"]+=1;continue
        rejected=parsed["calls"][0]["call"] if parsed["calls"] else None
        if parsed["calls"] and rejected is None:
            dropped["malformed_candidate"]+=1;continue
        prefix=trace["events"][:decision["before_event_count"]]
        remaining=min(task["max_recovery_steps"],trace["collection_limits"]["max_tool_calls"])-sum(e["phase"]=="recovery" for e in prefix)
        if remaining<1: dropped["no_remaining_budget"]+=1;continue
        with HardenedEnvironment(task) as env:
            for e in prefix: env.execute(ToolCall(**e["call"]))
            current=physical(env)
        matches=[s for s in reversed(states) if s["state"]==current]
        evidence=None
        for state in matches:
            chosen=copy.deepcopy(state["action"])
            if chosen: chosen["call_id"]="trusted-candidate"
            if chosen and rejected and all(chosen[k]==rejected[k] for k in ("name","arguments")):
                continue
            good=branch(task,prefix,chosen,state["index"],remaining)
            if not good["final_verification"]["success"]: continue
            bad=branch(task,prefix,rejected,state["index"],remaining)
            if good["start_state_id"]!=bad["start_state_id"]: raise ValueError("Different candidate states")
            if bad["final_verification"]["success"]: continue
            v=bad["candidate_verification"]
            proven=(rejected is None or v["unauthorized_modifications"] or v["extra_side_effects"] or
                    remaining==1 and not v["state_matches"])
            if not proven:
                uncertain.append({"task_id":task["id"],"decision_index":decision["index"],
                                  "reason":"fixed_continuation_failure_not_a_proof","chosen":good,"rejected":bad})
                continue
            evidence={"task_id":task["id"],"split":"train","dataset_version":VERSION,
                      "decision_index":decision["index"],"prompt":decision["prompt"],
                      "schema_hash":schema_hash(),"chosen_source":"oracle_reference","rejected_source":"llm",
                      "chosen":[action_message(chosen) if chosen else stop_message()],
                      "rejected":[action_message(rejected) if rejected else {"role":"assistant","content":parsed["text"]}],
                      "reference_index":state["index"],"evidence":{"chosen":good,"rejected":bad}}
            break
        if evidence: pairs.append(evidence)
        else: dropped["no_reliable_same_state_preference"]+=1
    return pairs,dict(dropped),uncertain

def export(tasks,traces,output):
    # All prompts are replay/causality validated; only final successful train actions receive loss.
    by_id={t["id"]:t for t in tasks};samples=[];pairs=[];uncertain=[];drops=Counter();seen=set()
    verified=0
    for trace in traces:
        task=by_id[trace["task_id"]]
        if trace["mode"]=="clean": drops["clean_control_excluded"]+=1;continue
        if not trace.get("events"): drops["missing_trace"]+=1;continue
        verify_inputs(task,trace);verified+=1
        if task["split"]!="train": drops["evaluation_excluded"]+=1;continue
        p,reasons,u=preferences(task,trace);pairs+=p;drops.update(reasons);uncertain+=u
        if not trace["verification"]["success"]:
            drops["failed_episode_sft"]+=1;continue
        for decision in trace.get("decisions",[]):
            parsed=decision["parsed"]
            if decision.get("error") or parsed["kind"] not in ("tools","stop"):
                drops["invalid_decision"]+=1;continue
            if parsed["kind"]=="tools":
                calls=parsed["calls"]
                actual=trace["events"][decision["before_event_count"]:decision["before_event_count"]+len(calls)]
                if any(c["error"] for c in calls) or len(actual)!=len(calls) or any(e["call"]!=c["call"] for e,c in zip(actual,calls)):
                    drops["unexecuted_or_malformed"]+=1;continue
                if any(not e["result"]["ok"] or e["behavior"]["label"] in ("unsafe","invalid","redundant","non_progress","uncertain") for e in actual):
                    drops["failed_or_untrusted_action"]+=1;continue
                target={"role":"assistant","content":None,"tool_calls":[action_message(c["call"])["tool_calls"][0] for c in calls]}
            else: target={"role":"assistant","content":parsed["text"]}
            messages=copy.deepcopy(decision["prompt"]["messages"])+[target]
            content_hash=fingerprint({"messages":messages,"tools":decision["prompt"]["tools"]})
            if content_hash in seen: drops["duplicate_sample"]+=1;continue
            seen.add(content_hash)
            source="llm" if trace["source"]=="llm" else "oracle_reference"
            samples.append({"sample_id":f"{task['id']}:{trace['source']}:{decision['index']}",
                            "source":source,"source_transport":trace["source"],"task_id":task["id"],"split":"train",
                            "dataset_version":VERSION,"mechanism_id":task["mechanism_id"],"model_id":trace["model_id"],
                            "schema_hash":schema_hash(),"prompt_hash":fingerprint(decision["prompt"]),
                            "messages":messages,"tools":decision["prompt"]["tools"],"loss_scope":"final_assistant_only"})
    output=Path(output)
    for source in ("llm","oracle_reference"):
        write_jsonl(output/f"{source}_sft_train.jsonl",[s for s in samples if s["source"]==source])
    write_jsonl(output/"dpo_candidates.jsonl",pairs)
    write_jsonl(output/"uncertain_candidates.jsonl",uncertain)
    report={"samples":len(samples),"sources":dict(Counter(s["source"] for s in samples)),
            "verified_trajectory_count":verified,"verified_dpo_candidates":len(pairs),
            "uncertain_candidates":len(uncertain),"discard_reasons":dict(drops),
            "trajectory_lengths":dict(Counter(t.get("recovery_steps",0) for t in traces)),
            "duplicate_rate":drops["duplicate_sample"]/(len(samples)+drops["duplicate_sample"]) if samples or drops["duplicate_sample"] else 0,
            "test_training_samples":0,"causal_prompt_validation":True,"hidden_answer_leak_found":False,
            "errors":dict(Counter(e["result"]["error"]["code"] for t in traces for e in t.get("events",[]) if e["result"]["error"])),
            "sample_mechanisms":dict(Counter(s["mechanism_id"] for s in samples)),
            "sample_target_lengths":dict(Counter(len(s["messages"][-1].get("tool_calls",[])) for s in samples)),
            "source_provenance_note":"Oracle-scripted Mock is reference data, not measured model behavior"}
    write_json(output/"report.json",report)
    return report

def validate_pairs(tasks,traces,pairs):
    # Independently re-execute retained evidence rather than trusting serialized verdict labels.
    by_task={t["id"]:t for t in tasks};by_trace={t["task_id"]:t for t in traces}
    for pair in pairs:
        task=by_task[pair["task_id"]];trace=by_trace[task["id"]]
        if task["split"]!="train" or trace["source"]!="llm": raise ValueError("Invalid preference source/split")
        decision=trace["decisions"][pair["decision_index"]]
        if pair["prompt"]!=decision["prompt"]: raise ValueError("Different preference prompt")
        prefix=trace["events"][:decision["before_event_count"]]
        parsed=decision["parsed"]
        rejected=parsed["calls"][0]["call"] if parsed["calls"] else None
        if pair["evidence"]["rejected"]["candidate"]!=rejected: raise ValueError("Rejected action not from actual model")
        trusted=reference_states(task)
        reference=next((s for s in trusted if s["index"]==pair["reference_index"]),None)
        chosen=copy.deepcopy(reference["action"]) if reference else None
        if chosen: chosen["call_id"]="trusted-candidate"
        if not reference or pair["evidence"]["chosen"]["candidate"]!=chosen: raise ValueError("Chosen action not trusted")
        for name in ("chosen","rejected"):
            old=pair["evidence"][name]
            actual=branch(task,prefix,old["candidate"],pair["reference_index"],old["remaining_steps"])
            if actual!=old: raise ValueError("Preference evidence altered")
        good,bad=pair["evidence"]["chosen"],pair["evidence"]["rejected"]
        v=bad["candidate_verification"]
        if not good["final_verification"]["success"] or bad["final_verification"]["success"] or not (
            rejected is None or v["unauthorized_modifications"] or v["extra_side_effects"] or
            bad["remaining_steps"]==1 and not v["state_matches"]):
            raise ValueError("Preference not proven")
        chosen_message=action_message(chosen) if chosen else stop_message()
        rejected_message=action_message(rejected) if rejected else {"role":"assistant","content":parsed["text"]}
        if pair["chosen"]!=[chosen_message] or pair["rejected"]!=[rejected_message]: raise ValueError("Preference serialization altered")
    return True
