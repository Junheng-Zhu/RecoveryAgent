"""Regression tests for v0.3-B causal data and isolated recovery mechanisms."""
import copy
import json
import pytest
from recovery_agent.models import ToolCall
from recovery_agent.hardening.tasks import generate_dataset, freeze, validate_frozen, NEW_FAMILIES
from recovery_agent.hardening.protocol import schema_hash
from recovery_agent.hardening.environment import HardenedEnvironment, verify
from recovery_agent.hardening.runner import run_episode, replay, OracleMock
from recovery_agent.hardening.batch import clean_control,clean_workflow, collect
from recovery_agent.hardening.analysis import summary
from recovery_agent.hardening.training import export, verify_inputs, preferences, validate_pairs
from recovery_agent.hardening.quality import validate_collection
from recovery_agent.stateful.predictor import OnlinePredictor, LLMConfig
from recovery_agent.data_pipeline.common import read_jsonl

TASKS,CONFIG=generate_dataset()
FIRST=[t for t in TASKS if t["variant_id"]==0]
def task_named(family):
    return next(t for t in FIRST if t["template_family"]==family)
def response(name=None,arguments=None,identifier="test",text="Done."):
    m={"role":"assistant","content":text if name is None else None}
    if name: m["tool_calls"]=[{"id":identifier,"type":"function","function":{"name":name,"arguments":json.dumps(arguments)}}]
    return {"model":"fixture-20260902","choices":[{"message":m,"finish_reason":"stop" if name is None else "tool_calls"}],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

@pytest.mark.parametrize("task",FIRST,ids=lambda t:t["template_family"])
def test_new_mechanisms_recover_replay_and_reset(task):
    trace=run_episode(task,OracleMock(task))
    assert trace["verification"]["success"] and trace["replay_verified"]
    assert replay(task,trace) and verify_inputs(task,trace)
    with HardenedEnvironment(task) as a,HardenedEnvironment(task) as b:
        before=b.snapshot()
        a.execute(ToolCall(**task["failed_action"]))
        a.reset()
        assert a.snapshot()==before and b.snapshot()==before and not a.audit
        folder=a.root
    assert not folder.exists()
    damaged=copy.deepcopy(trace);damaged["audit"].append({"resource":"fake"})
    with pytest.raises(ValueError): replay(task,damaged)

@pytest.mark.parametrize("task",FIRST,ids=lambda t:t["template_family"])
def test_clean_counterfactual_controls_are_separate(task):
    clean=clean_control(task)
    trace=run_episode(clean,OracleMock(clean),clean=True)
    assert trace["verification"]["success"] and not trace["audit"]
    assert trace["mode"]=="clean" and trace["replay_verified"]
    assert verify_inputs(clean,trace)

def test_manifest_reproducibility_and_namespace(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"
    tasks,manifest=freeze(a,schema_hash=schema_hash())
    freeze(b,schema_hash=schema_hash())
    assert {p.name:p.read_bytes() for p in a.iterdir()}=={p.name:p.read_bytes() for p in b.iterdir()}
    assert validate_frozen(a,schema_hash())[0]==tasks
    assert len(tasks)==96 and len(NEW_FAMILIES)==8
    for t in tasks:
        if "accounts/a1" in t["goal"]["instruction"]:
            assert "accounts/a1_"+str(t["variant_id"]).zfill(3) in t["goal"]["instruction"]
    train={t["mechanism_id"] for t in tasks if t["split"]=="train"}
    assert not train & {t["mechanism_id"] for t in tasks if t["eval_group"]=="ood"}
    with pytest.raises(FileExistsError): freeze(a,schema_hash=schema_hash())
    (a/"train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError,match="hash"): validate_frozen(a,schema_hash())

def test_queries_classified_by_evidence_not_oracle_steps():
    task=task_named("sqlite_primary_key")
    args={"table":"accounts","record_id":"a1_000"}
    scripted=[response("execute_select",args,"q1"),response("execute_select",args,"q2"),
              response("update_record",{"table":"accounts","record_id":"a1_000","values":{"balance":20}},"fix"),
              response()]
    trace=run_episode(task,OracleMock(task,scripted))
    assert trace["verification"]["success"]
    assert [e["behavior"]["label"] for e in trace["events"][1:]]==["diagnostic","redundant","recovery_action"]
    report=summary([trace])
    assert report["unnecessary_tool_call_rate"]==pytest.approx(1/3)
    assert report["behavior_counts"]["diagnostic"]==1 and report["by_initial_error"]

def test_retry_invalid_unsafe_and_uncertain_labels():
    task=task_named("sqlite_primary_key")
    bad=task["failed_action"]
    trace=run_episode(task,OracleMock(task,[response(bad["name"],bad["arguments"],"r"),
            response("imaginary",{},"invalid"),response("write_file",{"path":"forbidden.txt","content":"bad"},"unsafe"),response()]))
    assert [e["behavior"]["label"] for e in trace["events"][1:]]==["non_progress","invalid","unsafe"]
    assert not trace["verification"]["success"] and replay(task,trace)

def test_same_state_preferences_actual_model_and_later_error(tmp_path):
    task=task_named("sqlite_primary_key")
    responses=[response("execute_select",{"table":"accounts","record_id":"a1_000"},"q"),
               response("write_file",{"path":"forbidden.txt","content":"bad"},"bad"),response()]
    mock=OracleMock(task,responses)
    mock.source="llm"
    trace=run_episode(task,mock)
    # Explicit fixture provenance only exercises export; no claim of real-model performance.
    trace["source"]="llm"
    pairs,drops,uncertain=preferences(task,trace)
    assert any(p["decision_index"]==1 for p in pairs)
    assert validate_pairs([task],[trace],pairs)
    for p in pairs:
        assert p["evidence"]["chosen"]["start_state_id"]==p["evidence"]["rejected"]["start_state_id"]
        assert p["evidence"]["chosen"]["final_verification"]["success"]
    good=run_episode(task,OracleMock(task))
    tampered=copy.deepcopy(good);tampered["decisions"][0]["prompt"]["messages"][0]["content"]+=" expected_state=secret"
    with pytest.raises(ValueError,match="Prompt"): verify_inputs(task,tampered)
    test_task=next(t for t in TASKS if t["eval_group"]=="iid_test" and t["domain"]=="sqlite")
    test_trace=run_episode(test_task,OracleMock(test_task))
    report=export([task,test_task],[good,test_trace],tmp_path)
    assert report["test_training_samples"]==0 and report["discard_reasons"]["evaluation_excluded"]==1
    assert not read_jsonl(tmp_path/"llm_sft_train.jsonl")
    assert all(s["task_id"]==task["id"] and s["loss_scope"]=="final_assistant_only" for s in read_jsonl(tmp_path/"oracle_reference_sft_train.jsonl"))

def test_fixed_continuation_failure_is_uncertain():
    task=task_named("sqlite_primary_key")
    trace=run_episode(task,OracleMock(task,[response("execute_select",{"table":"entries"},"wrong_query"),response()]))
    trace["source"]="llm"
    pairs,drops,uncertain=preferences(task,trace)
    assert any(u["reason"]=="fixed_continuation_failure_not_a_proof" for u in uncertain)
    assert not any(p["decision_index"]==0 for p in pairs)

def test_collection_resume_quality_and_target_tampering(tmp_path):
    dataset=tmp_path/"dataset"
    freeze(dataset,schema_hash=schema_hash())
    task=task_named("sqlite_primary_key")
    output=tmp_path/"run";training=tmp_path/"training"
    tasks,traces,report=collect([task],output)
    before=(output/"trajectories.jsonl").read_bytes()
    collect([task],output,resume=True)
    assert before==(output/"trajectories.jsonl").read_bytes()
    export(tasks,traces,training)
    assert validate_collection(dataset,output,training)["passed"]
    p=training/"oracle_reference_sft_train.jsonl"
    samples=read_jsonl(p);samples[0]["messages"][-1]={"role":"assistant","content":"tampered"}
    p.write_text("\n".join(json.dumps(s) for s in samples)+"\n")
    with pytest.raises(ValueError,match="target"): validate_collection(dataset,output,training)

def test_online_mock_http_uses_extended_schema_and_no_default_temperature(tmp_path):
    task=task_named("sqlite_delete_dependency")
    scripted=OracleMock(task);requests=[]
    def transport(payload):
        requests.append(payload)
        return scripted.predict({"messages":payload["messages"],"tools":payload["tools"]})["raw_response"]
    # Use explicit HTTP-interface transport injection, never real network.
    config=LLMConfig(model="fixture-20260902",base_url="https://example.invalid/v1",max_requests=8)
    predictor=OnlinePredictor(config,"fixture-secret",online=True,transport=transport,budget_path=tmp_path/"budget.json")
    trace=run_episode(task,predictor,max_requests=8)
    assert trace["verification"]["success"]
    assert all("temperature" not in p for p in requests)
    assert len(requests[0]["tools"])==15
    assert "fixture-secret" not in json.dumps(trace)

@pytest.mark.parametrize("task",FIRST,ids=lambda t:t["template_family"])
def test_clean_workflows(task):
    clean=clean_workflow(task)
    trace=run_episode(clean,OracleMock(clean),clean=True)
    assert trace["verification"]["success"] and trace["replay_verified"]
    assert verify_inputs(clean,trace)

def test_malformed_dependency_is_observation_not_episode_exception():
    task=task_named("git_multifile_revision")
    with HardenedEnvironment(task) as env:
        env.execute(ToolCall("write_file",{"path":"case_000/metadata.json","content":"not json"}))
        result=env.execute(ToolCall("git_commit",{"message":"publish"}))
        assert not result.ok

def test_http_request_wire_format_is_mocked(monkeypatch):
    from recovery_agent.stateful.predictor import NoRedirect
    seen=[]
    class HTTPResponse:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,limit): return json.dumps(response()).encode()
    class Opener:
        def open(self,request,timeout):
            seen.append((request,timeout))
            return HTTPResponse()
    monkeypatch.setattr("urllib.request.build_opener",lambda *handlers:Opener())
    cfg=LLMConfig(model="fixture-20260902",base_url="https://example.invalid/v1",max_requests=1)
    predictor=OnlinePredictor(cfg,"private-fixture",online=True)
    result=predictor.predict({"messages":[{"role":"user","content":"fixture"}],"tools":[]})
    request,timeout=seen[0]
    assert request.full_url=="https://example.invalid/v1/chat/completions"
    assert request.get_header("Authorization")=="Bearer private-fixture"
    assert json.loads(request.data)["model"]==cfg.model and timeout==cfg.timeout
    assert result["error"] is None

def test_schema_unique_and_sampling_independent():
    from recovery_agent.hardening.protocol import schemas
    from recovery_agent.stateful.collector import sample_tasks
    names=[s["function"]["name"] for s in schemas()]
    assert len(names)==len(set(names))==15
    sample=sample_tasks(TASKS,50,42)
    assert len(sample)==50 and len({t["id"] for t in sample})==50
    assert all(t["split"]=="train" for t in sample)
    assert sample==sample_tasks(TASKS,50,42)

def test_preference_evidence_and_serialization_tampering():
    task=task_named("sqlite_primary_key")
    trace=run_episode(task,OracleMock(task,[response("write_file",{"path":"bad.txt","content":"bad"}),response()]))
    trace["source"]="llm"
    pairs,_,_=preferences(task,trace)
    pair=copy.deepcopy(pairs[0]);pair["rejected"][0]["content"]="fabricated"
    with pytest.raises(ValueError,match="serialization"): validate_pairs([task],[trace],[pair])

def test_offline_review_preserves_raw_and_uncertain_not_supervised(tmp_path):
    from recovery_agent.hardening.batch import review_collection
    task=task_named("sqlite_primary_key")
    original=tmp_path/"original";reviewed=tmp_path/"reviewed"
    collect([task],original)
    before=(original/"trajectories.jsonl").read_bytes()
    review_collection(original,reviewed)
    assert (original/"trajectories.jsonl").read_bytes()==before
    traces=read_jsonl(reviewed/"trajectories.jsonl")
    assert verify_inputs(task,traces[0])
    with pytest.raises(ValueError,match="Derived"): collect([task],reviewed,resume=True)
    with pytest.raises(FileExistsError): review_collection(original,reviewed)

def test_classification_tamper_rejected():
    task=task_named("sqlite_primary_key")
    trace=run_episode(task,OracleMock(task))
    trace["events"][1]["behavior"]["label"]="unsafe"
    with pytest.raises(ValueError,match="Behavior"): verify_inputs(task,trace)

def test_unique_intermediate_has_constraint_progress_evidence():
    task=task_named("sqlite_unique_swap")
    trace=run_episode(task,OracleMock(task))
    evidence=trace["events"][2]["behavior"]
    assert evidence["label"]=="recovery_action"
    assert evidence["reasons"]==["authorized_intermediate_frees_required_unique_value"]

@pytest.mark.parametrize("family",["mixed_db_git_report","git_compensating_commit"])
def test_worktree_repair_progress_is_not_hidden_by_dirty_status(family):
    task=task_named(family)
    trace=run_episode(task,OracleMock(task))
    write=next(e for e in trace["events"] if e["call"]["name"]=="write_file")
    assert write["behavior"]["label"]=="recovery_action"
    assert write["behavior"]["reasons"]==["target_file_condition_satisfied"]
