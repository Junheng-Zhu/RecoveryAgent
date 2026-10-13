"""Four focused tests for v0.3-C.2; only isolated tools and scripted Mock outputs."""
import copy
import json
from recovery_agent.models import ToolCall
from recovery_agent.data_pipeline.serialization import build_prompt
from recovery_agent.hardening.tasks import generate_dataset
from recovery_agent.hardening.protocol import prompt_for,schemas,STOP_PROTOCOLS,STOP_INSTRUCTION,STOP_INSTRUCTION_C2
from recovery_agent.hardening.runner import OracleMock,run_episode,replay
from recovery_agent.hardening.mutation_diagnostics import diagnose_mutations

def task(family):
    return next(t for t in generate_dataset()[0] if t["template_family"]==family and t["variant_id"]==2)

def response(*calls):
    message={"role":"assistant","content":None if calls else "Done."}
    if calls:
        message["tool_calls"]=[{"id":c.call_id,"type":"function","function":{"name":c.name,"arguments":json.dumps(c.arguments)}} for c in calls]
    return {"model":"local-scripted-mock","choices":[{"message":message,"finish_reason":"tool_calls" if calls else "stop"}],
            "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

def test_repeated_git_add_without_state_change():
    t=task("git_multifile_revision")
    path=t["git_paths"][0]
    trace=run_episode(t,OracleMock(t,[response(ToolCall("git_add",{"path":path},"add1")),
                                    response(ToolCall("git_add",{"path":path},"add2")),response()]))
    row=diagnose_mutations(t,trace)
    calls=[e for e in row["mutation_evidence"] if e["call"]["name"]=="git_add"]
    assert len(calls)==2 and all("redundant_state_mutation" in e["labels"] for e in calls)
    assert all(e["new_audit_records"][0]["resource_state_before"]==e["new_audit_records"][0]["resource_state_after"] for e in calls)
    assert "mutation_budget_exceeded" in calls[1]["labels"]
    assert calls[1]["new_audit_records"][0]["audit_occurrence"]==2
    assert trace["verification"]["extra_side_effects"]==1

def test_redundancy_and_unauthorized_mutation_are_distinct_without_verifier_changes():
    t=task("git_multifile_revision")
    stage=run_episode(t,OracleMock(t,[response(ToolCall("git_add",{"path":t["git_paths"][0]},"stage")),response()]))
    protected=next(p for p in t["initial_state"]["files"] if p.endswith("protected.txt"))
    # Even rewriting identical protected contents is audited and unauthorized.
    illegal=run_episode(t,OracleMock(t,[response(ToolCall("write_file",{"path":protected,"content":t["initial_state"]["files"][protected]},"write")),response()]))
    snapshots=[copy.deepcopy(tr["verification"]) for tr in (stage,illegal)]
    a,b=diagnose_mutations(t,stage),diagnose_mutations(t,illegal)
    assert "redundant_state_mutation" in a["diagnostic_labels"] and "unauthorized_mutation" not in a["diagnostic_labels"]
    assert {"redundant_state_mutation","unauthorized_mutation"}<=set(b["diagnostic_labels"])
    assert illegal["verification"]["unauthorized_modifications"]==1
    assert [tr["verification"] for tr in (stage,illegal)]==snapshots
    assert a["final_verifier"]==snapshots[0] and b["final_verifier"]==snapshots[1]
    assert replay(t,stage) and replay(t,illegal)

def test_premature_stop_and_budget_incomplete_recovery():
    t=task("sqlite_primary_key")
    premature=run_episode(t,OracleMock(t,[response()]))
    exhausted=run_episode(t,OracleMock(t,[response(ToolCall("execute_select",{"table":"accounts"},"query"))]),max_requests=1)
    a,b=diagnose_mutations(t,premature),diagnose_mutations(t,exhausted)
    assert a["termination_reason"]=="assistant_text" and "premature_stop" in a["diagnostic_labels"]
    assert "incomplete_recovery" not in a["diagnostic_labels"]
    assert b["termination_reason"]=="request_budget" and "incomplete_recovery" in b["diagnostic_labels"]
    assert "premature_stop" not in b["diagnostic_labels"]
    assert not a["ever_full_success"] and not b["ever_full_success"]

def test_c2_prompt_safety_and_legacy_c1_compatibility():
    t=task("sqlite_primary_key")
    old=build_prompt(t["goal"],[],4);old["tools"]=schemas()
    assert STOP_PROTOCOLS==("legacy","termination-v0.3c-1","termination-v0.3c-2")
    assert prompt_for(t,[],4)==old
    expected=copy.deepcopy(old);expected["messages"][0]["content"]+="\n"+STOP_INSTRUCTION
    assert prompt_for(t,[],4,stop_protocol="termination-v0.3c-1")==expected
    hidden=copy.deepcopy(t)
    hidden.update(expected="HIDDEN_EXPECTED",reference_plan="HIDDEN_PLAN",allowed_resources={"HIDDEN_AUDIT":999})
    hidden["fault"]["seed"]="HIDDEN_SEED"
    new=prompt_for(hidden,[],4,stop_protocol="termination-v0.3c-2")
    assert new["messages"][0]["content"]==old["messages"][0]["content"]+"\n"+STOP_INSTRUCTION_C2
    assert new["messages"][1:]==old["messages"][1:] and new["tools"]==old["tools"]
    assert all(x not in json.dumps(new) for x in ("HIDDEN_EXPECTED","HIDDEN_PLAN","HIDDEN_AUDIT","HIDDEN_SEED"))
