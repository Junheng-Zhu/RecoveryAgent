"""Five focused v0.3-C checks; all predictors are local scripted Mock instances."""
import copy
import json
from recovery_agent.models import ToolCall
from recovery_agent.data_pipeline.serialization import build_prompt
from recovery_agent.hardening.tasks import generate_dataset
from recovery_agent.hardening.runner import OracleMock,run_episode,replay
from recovery_agent.hardening.protocol import prompt_for,schemas,STOP_INSTRUCTION
from recovery_agent.hardening.termination import diagnose_episode,summarize_termination

def task():
    return next(t for t in generate_dataset()[0] if t["template_family"]=="sqlite_primary_key" and t["variant_id"]==0)

def fix(t):
    key,row=next(iter(t["expected"]["sqlite"]["accounts"].items()))
    return ToolCall("update_record",{"table":"accounts","record_id":key,"values":{"balance":row["balance"]}},"fix")

def query(t,identifier="query"):
    key=next(iter(t["sqlite_initial"]["accounts"]))
    return ToolCall("execute_select",{"table":"accounts","record_id":key},identifier)

def response(*calls):
    message={"role":"assistant","content":None if calls else "Done."}
    if calls:
        message["tool_calls"]=[{"id":c.call_id,"type":"function","function":{"name":c.name,"arguments":json.dumps(c.arguments)}} for c in calls]
    return {"model":"scripted-local-mock","choices":[{"message":message,"finish_reason":"tool_calls" if calls else "stop"}],
            "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

def test_first_full_success_detection():
    t=task()
    trace=run_episode(t,OracleMock(t,[response(query(t)),response(fix(t)),response()]))
    row=diagnose_episode(t,trace)
    assert not row["verifier_prefixes"][1]["verification"]["success"]
    assert row["first_goal_achieved_step"]==2 and row["final_full_success"]
    assert row["replay_verified"] and row["autonomous_completion"]

def test_post_success_read_only_calls():
    t=task()
    trace=run_episode(t,OracleMock(t,[response(fix(t)),response(query(t)),response()]))
    row=diagnose_episode(t,trace)
    assert row["first_goal_achieved_step"]==1 and row["post_success_tool_calls"]==1
    assert row["model_stop_correct"] and row["autonomous_completion"]
    assert not trace["audit"][0]["resource"].startswith("file:")

def test_model_stop_and_budget_termination_are_distinct():
    t=task()
    model=run_episode(t,OracleMock(t,[response(fix(t)),response()]),max_tool_calls=3,max_requests=3)
    tool=run_episode(t,OracleMock(t,[response(fix(t))]),max_tool_calls=1,max_requests=3)
    request=run_episode(t,OracleMock(t,[response(fix(t))]),max_tool_calls=3,max_requests=1)
    assert [x["termination_reason"] for x in (model,tool,request)]==["assistant_text","tool_budget","request_budget"]
    rows=[diagnose_episode(t,x) for x in (model,tool,request)]
    assert [r["autonomous_completion"] for r in rows]==[True,False,False]
    assert all(r["success_before_budget_exhaustion"] for r in rows)
    metrics=summarize_termination(rows)["groups"]["model_autonomous"]
    assert metrics["model_stop_accuracy"]=={"numerator":1,"denominator":3,"value":1/3}
    assert metrics["autonomous_completion_rate"]["denominator"]==3

def test_runtime_assisted_stop_is_separate_and_stops_inside_batch():
    t=task()
    trace=run_episode(t,OracleMock(t,[response(fix(t),query(t)),response()]),stop_on_verified_success=True)
    row=diagnose_episode(t,trace)
    assert trace["termination_reason"]=="runtime_verified_success"
    assert trace["recovery_steps"]==1 and len(trace["decisions"])==1
    assert len(trace["decisions"][0]["parsed"]["calls"])==2 and len(trace["events"])==2
    assert trace["verification"]["success"] and len(trace["audit"])==1 and replay(t,trace)
    assert not row["autonomous_completion"] and row["post_success_tool_calls"]==0
    report=summarize_termination([row])
    assert report["groups"]["model_autonomous"]["episode_count"]==0
    assert report["groups"]["runtime_assisted"]["autonomous_completion_rate"]["value"] is None

def test_prompt_safety_and_default_compatibility():
    t=task()
    legacy=run_episode(t,OracleMock(t),max_tool_calls=4,max_requests=4)
    expected=build_prompt(t["goal"],legacy["events"][:1],4)
    expected["tools"]=schemas()
    assert legacy["decisions"][0]["prompt"]==expected
    assert legacy["termination_reason"]=="assistant_text" and "termination_protocol" not in legacy
    hidden=copy.deepcopy(t)
    hidden["expected"]="HIDDEN_EXPECTED_SENTINEL"
    hidden["reference_plan"]="HIDDEN_ORACLE_SENTINEL"
    hidden["fault"]["seed"]="HIDDEN_SEED_SENTINEL"
    new=prompt_for(hidden,legacy["events"][:1],4,stop_protocol="termination-v0.3c-1")
    assert new["tools"]==expected["tools"]
    assert new["messages"][0]["content"]==expected["messages"][0]["content"]+"\n"+STOP_INSTRUCTION
    assert new["messages"][1:]==expected["messages"][1:]
    assert all(secret not in json.dumps(new) for secret in ("HIDDEN_EXPECTED_SENTINEL","HIDDEN_ORACLE_SENTINEL","HIDDEN_SEED_SENTINEL"))
    changed=run_episode(t,OracleMock(t),max_tool_calls=4,max_requests=4,stop_protocol="termination-v0.3c-1")
    assert changed["termination_protocol"]=={"prompt_version":"termination-v0.3c-1","stop_on_verified_success":False}
    assert changed["verification"]["success"]
