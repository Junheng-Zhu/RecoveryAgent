"""Independent v0.3-C.2 attribution; never changes recorded labels, contracts or success rules."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from ..data_pipeline.common import read_jsonl,write_json,write_jsonl
from .termination import diagnose_episode,summarize_termination,compare_stop_protocols

def resource_state(state,resource):
    # Known snapshot scopes only: an unavailable scope cannot prove a no-op.
    if resource.startswith("file:"):
        return True,state["files"].get(resource[5:])
    if resource.startswith(("sqlite:","table:")):
        parts=resource.split(":",2)
        if len(parts)!=3: return False,None
        section="sqlite" if parts[0]=="sqlite" else "tables"
        return True,state[section].get(parts[1],{}).get(parts[2])
    if resource.startswith("git:") and state["git"]:
        key=resource[4:]
        if key=="head": return True,state["git"]
        if key in ("index","branch"): return True,state["git"].get(key)
    return False,None

def diagnose_mutations(task,trace):
    # A Replay observer correlates actual scoped state, new Audit records and full verifier deltas.
    evidence=[];counts=Counter();previous=None;old_verdict=None;audited=0;step=0;event_count=0
    def observe(env,event,verdict):
        nonlocal previous,old_verdict,audited,step,event_count
        state=env.snapshot()
        if event is None:
            previous=state;old_verdict=verdict;audited=len(env.audit)
            counts.update(a["resource"] for a in env.audit)
            return
        event_count+=1
        if event["phase"]=="recovery": step+=1
        mutations=env.audit[audited:];labels=set();records=[]
        for mutation in mutations:
            resource=mutation["resource"];counts[resource]+=1
            before_known,before=resource_state(previous,resource)
            after_known,after=resource_state(state,resource)
            known=before_known and after_known
            no_op=known and before==after and mutation["before"]==mutation["after"]
            permitted=resource in task["allowed_resources"]
            limit=task["allowed_resources"].get(resource)
            if no_op: labels.add("redundant_state_mutation")
            if not permitted: labels.add("unauthorized_mutation")
            if permitted and counts[resource]>limit: labels.add("mutation_budget_exceeded")
            if not known: labels.add("uncertain")
            records.append({"resource":resource,"audit":copy.deepcopy(mutation),"scope_known":known,
                            "resource_state_before":copy.deepcopy(before),"resource_state_after":copy.deepcopy(after),
                            "no_state_change":bool(no_op),"permitted":permitted,
                            "audit_occurrence":counts[resource],"contract_limit":limit,
                            "budget_exceeded":bool(permitted and counts[resource]>limit)})
        extra_delta=verdict["extra_side_effects"]-old_verdict["extra_side_effects"]
        unauthorized_delta=verdict["unauthorized_modifications"]-old_verdict["unauthorized_modifications"]
        if (extra_delta>0 and "mutation_budget_exceeded" not in labels or
            unauthorized_delta>0 and "unauthorized_mutation" not in labels):
            labels.add("uncertain")
        if mutations or labels:
            evidence.append({"event_index":event_count-1,"recovery_step":step,"phase":event["phase"],
                             "call":copy.deepcopy(event["call"]),"tool_ok":event["result"]["ok"],
                             "labels":sorted(labels),"new_audit_records":records,
                             "verifier_before":copy.deepcopy(old_verdict),"verifier_after":copy.deepcopy(verdict),
                             "extra_side_effect_delta":extra_delta,"unauthorized_modification_delta":unauthorized_delta})
        previous=state;old_verdict=verdict;audited=len(env.audit)
    row=diagnose_episode(task,trace,event_observer=observe)
    labels={label for call in evidence for label in call["labels"]}
    final=copy.deepcopy(trace["verification"])
    ledger_bad=bool(final["unauthorized_modifications"] or final["extra_side_effects"])
    # Do not relabel a correct final object state polluted by audit violations as premature stopping.
    if row["model_stopped"] and not final["state_matches"] and not ledger_bad:
        labels.add("premature_stop")
    if row["budget_forced"] and not row["ever_full_success"]:
        labels.add("incomplete_recovery")
    if not final["success"] and not labels:
        labels.add("uncertain")
    categories=[]
    if "premature_stop" in labels: categories.append("Premature Stop")
    if "redundant_state_mutation" in labels: categories.append("Redundant Mutation")
    if "unauthorized_mutation" in labels: categories.append("Unauthorized Mutation")
    if not final["success"]: categories.append("Recovery Failure")
    reasons=[]
    if not final["state_matches"]: reasons.append("final_state_mismatch")
    if final["unauthorized_modifications"]: reasons.append("unauthorized_modification_in_audit")
    if final["extra_side_effects"]: reasons.append("extra_side_effect_in_audit")
    first_state=next((p["recovery_step"] for p in row["verifier_prefixes"] if p["verification"]["state_matches"]),None)
    row.update(diagnostic_labels=sorted(labels),failure_categories=categories,final_verifier=final,
               failure_reasons=reasons,first_state_match_step=first_state,mutation_evidence=evidence)
    return row

def diagnose_pair(first,second,output):
    # Matched tasks and configurations are checked before native isolated Replay. No predictor is invoked.
    first,second,output=Path(first),Path(second),Path(output)
    if any(output.resolve()==p.resolve() or output.resolve().is_relative_to(p.resolve()) for p in (first,second)):
        raise ValueError("Output must be separate from source collections")
    if output.exists() and any(output.iterdir()): raise FileExistsError("Diagnostic output exists")
    compatibility=compare_stop_protocols(first,second)
    collections={}
    hashes={}
    for directory in (first,second):
        metadata=json.loads((directory/"session.json").read_text(encoding="utf-8"))
        version=metadata["session"].get("termination_protocol",{}).get("prompt_version","legacy")
        tasks=read_jsonl(directory/"tasks.jsonl");traces=read_jsonl(directory/"trajectories.jsonl")
        by_id={t["id"]:t for t in tasks}
        if len(by_id)!=len(tasks) or len({t["task_id"] for t in traces})!=len(traces):
            raise ValueError("Duplicate task/episode ID")
        collections[version]=[diagnose_mutations(by_id[t["task_id"]],t) for t in traces]
        hashes[str(directory)]={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                               for name in ("tasks.jsonl","trajectories.jsonl","session.json")}
    versions=list(collections)
    before,after=[{r["task_id"]:r for r in collections[v]} for v in versions]
    if set(before)!=set(after): raise ValueError("Unmatched episode IDs")
    pairs=[{"task_id":key,"before_protocol":versions[0],"after_protocol":versions[1],
            "before":before[key],"after":after[key],
            "delta_recovery_calls":after[key]["recovery_steps"]-before[key]["recovery_steps"],
            "delta_post_success_calls":after[key]["post_success_tool_calls"]-before[key]["post_success_tool_calls"],
            "new_labels":sorted(set(after[key]["diagnostic_labels"])-set(before[key]["diagnostic_labels"])),
            "removed_labels":sorted(set(before[key]["diagnostic_labels"])-set(after[key]["diagnostic_labels"]))}
           for key in before]
    report={"diagnostic_version":"v0.3-C.2","offline":True,"compatibility":compatibility,
            "episode_count":sum(len(r) for r in collections.values()),"paired_tasks":len(pairs),
            "original_termination_metrics":{v:summarize_termination(rows) for v,rows in collections.items()},
            "episode_label_counts":{v:dict(Counter(label for row in rows for label in row["diagnostic_labels"])) for v,rows in collections.items()},
            "call_label_counts":{v:dict(Counter(label for row in rows for e in row["mutation_evidence"] for label in e["labels"])) for v,rows in collections.items()},
            "source_hashes":hashes,
            "definitions":{"redundant_state_mutation":"Successful audited mutation with identical observed resource state and identical audit before/after; still consumes audit quota.",
                           "unauthorized_mutation":"Actual new Audit resource is absent from allowed_resources.",
                           "mutation_budget_exceeded":"Occurrence of an allowed Audit resource exceeds its unchanged task contract.",
                           "premature_stop":"Assistant text with unmet final state, without audit pollution; ledger-only failure is not attributed to early stop.",
                           "incomplete_recovery":"Budget termination and no prefix ever passed the complete verifier.",
                           "uncertain":"Unknown scope, unexplained verifier violation or otherwise unattributed failure. Labels may coexist."}}
    write_jsonl(output/"episodes.jsonl",[row for rows in collections.values() for row in rows])
    write_jsonl(output/"pairs.jsonl",pairs);write_json(output/"summary.json",report)
    table=["# Teacher10 paired mutation/termination diagnosis v0.3-C.2","",
           "| Task | Protocol | First full goal | Post-success calls | Final verifier (success/state/unauthorized/extra) | Labels |",
           "| --- | --- | ---: | ---: | --- | --- |"]
    for rows in collections.values():
        for row in rows:
            v=row["final_verifier"];first_goal=row["first_goal_achieved_step"]
            table.append(f"| {row['task_id']} | {row['prompt_version']} | {first_goal if first_goal is not None else 'never'} | {row['post_success_tool_calls']} | {v['success']}/{v['state_matches']}/{v['unauthorized_modifications']}/{v['extra_side_effects']} | {', '.join(row['diagnostic_labels']) or 'none'} |")
    table+=["","Paired changes:","",
            "| Task | Final success before/after | Recovery-call delta | Post-success-call delta | New labels | Removed labels |",
            "| --- | --- | ---: | ---: | --- | --- |"]
    for pair in pairs:
        table.append(f"| {pair['task_id']} | {pair['before']['final_full_success']}/{pair['after']['final_full_success']} | {pair['delta_recovery_calls']} | {pair['delta_post_success_calls']} | {', '.join(pair['new_labels']) or 'none'} | {', '.join(pair['removed_labels']) or 'none'} |")
    focus=next((p for p in pairs if p["task_id"]=="0.3-B-git_multifile_revision-002"),None)
    if focus:
        table+=["","## git_multifile_revision-002 evidence","",
                "Initial staging is trusted setup and does not consume recovery Audit quota. The redundant report staging consumes the sole permitted index mutation; necessary metadata staging then exceeds it. This is not an early-stop diagnosis.",
                "",chr(96)*3+"json",json.dumps(focus["after"]["mutation_evidence"],ensure_ascii=False,indent=2),chr(96)*3]
    (output/"report.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    return report
