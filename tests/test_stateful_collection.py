import copy
import json
import urllib.error
from pathlib import Path
import pytest
from recovery_agent.models import ToolCall
from recovery_agent.stateful.tasks import generate_tasks, StatefulOracle, verify
from recovery_agent.stateful.environment import StatefulEnvironment, DB_COLUMNS
from recovery_agent.stateful.protocol import schemas, parse_response, prompt_for
from recovery_agent.stateful.predictor import MockPredictor, OnlinePredictor, LLMConfig, BudgetExceeded
from recovery_agent.stateful.collector import run_episode, replay, collect, sample_tasks, summarize
from recovery_agent.stateful.training import candidate_evidence, prepare

def task_named(name):
    return next(t for t in generate_tasks(1) if t["template_family"] == name)

def response(*calls, text=None):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [{"id": f"call-{i}", "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for i,c in enumerate(calls)]
    return {"model": "test", "choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

@pytest.mark.parametrize("task", generate_tasks(1), ids=lambda t:t["id"])
def test_native_fault_oracle_replay_and_clean(task):
    # Both recovery and clean-start modes use actual Git/SQLite, not fabricated tool returns.
    trace = run_episode(task, MockPredictor(task))
    assert trace["verification"]["success"], trace
    assert not trace["events"][0]["result"]["ok"]
    assert trace["replay_verified"] and replay(task, trace)
    clean = run_episode(task, MockPredictor(task), clean=True)
    assert clean["verification"]["success"] and replay(task, clean)
    damaged = copy.deepcopy(trace)
    damaged["audit"].append({"fake": True})
    with pytest.raises(ValueError, match="audit"):
        replay(task, damaged)

def test_git_scope_conflict_empty_commit_and_reset():
    task = task_named("git_checkout_conflict")
    with StatefulEnvironment(task) as env:
        before = env.snapshot()
        assert not env.execute(ToolCall("git_checkout", {"path":"report.txt", "expected_content":"wrong"})).ok
        assert env.snapshot() == before and not env.audit
        for path in ("../outside", ".git/config", ".GIT/HEAD", "protected.txt"):
            result=env.execute(ToolCall("git_checkout", {"path":path,"expected_content":"KEEP"}))
            assert not result.ok
        for name in ("git_push","git_reset","git_clone"):
            assert not env.execute(ToolCall(name,{})).ok
        result = env.execute(ToolCall("git_commit", {"message":"empty"}))
        assert result.error.code == "NO_CHANGES"
        env.reset()
        assert env.snapshot() == before and not env.audit
        folder=env.root
    assert not folder.exists()

def test_sql_transactions_parameterization_delete_and_isolation():
    task = task_named("sqlite_transaction")
    with StatefulEnvironment(task) as env, StatefulEnvironment(task) as other:
        before=env.snapshot()
        failed=env.execute(ToolCall(**task["failed_action"]))
        assert failed.error.code == "TRANSACTION_FAILED"
        assert env.snapshot()==before and not env.audit
        assert not env.execute(ToolCall("execute_select",{"table":"accounts; DROP TABLE accounts;"})).ok
        assert not env.execute(ToolCall("insert_record",{"table":"accounts","record_id":"a2",
            "values":{"email":"b", "balance":1, "sql":"DELETE FROM accounts"}})).ok
        odd="x'); DROP TABLE accounts;--"
        assert env.execute(ToolCall("insert_record",{"table":"accounts","record_id":odd,
            "values":{"email":"b", "balance":1}})).ok
        assert env.execute(ToolCall("execute_select",{"table":"accounts","record_id":odd})).value[0]["id"]==odd
        assert env.execute(ToolCall("delete_record",{"table":"accounts","record_id":odd})).ok
        assert not env.execute(ToolCall("delete_record",{"table":"accounts","record_id":odd})).ok
        assert other.snapshot()==before
        env.reset()
        assert env.snapshot()==before and not env.audit

def test_transient_fault_and_constraint_rollback():
    task = task_named("sqlite_unique")
    task["fault"]["type"]="transient"
    trace=run_episode(task,MockPredictor(task))
    assert trace["events"][0]["result"]["error"]["code"]=="TEMPORARY_UNAVAILABLE"
    assert trace["verification"]["success"] and replay(task,trace)

@pytest.mark.parametrize("raw", [{}, {"choices":[]}, response(text=""), {"choices":[{"message":{"role":"assistant","tool_calls":[
    {"type":"function","id":"x","function":{"name":"read_file","arguments":"{bad"}}]}}]}])
def test_bad_model_responses_visible(raw):
    parsed=parse_response(raw)
    assert parsed["kind"]=="invalid" or parsed["calls"][0]["error"]

def test_mock_invalid_multi_call_text_and_budget():
    task=task_named("sqlite_primary_key")
    invalid=response(ToolCall("imaginary_tool",{}))
    stop=response(text="I am done.")
    trace=run_episode(task,MockPredictor(task,[invalid,stop]))
    assert trace["invalid_tool_calls"]==1 and not trace["verification"]["success"]
    assert trace["events"][1]["result"]["error"]["code"]=="UNKNOWN_TOOL"
    assert replay(task,trace)
    first=response(ToolCall("execute_select",{"table":"accounts","record_id":"a1"}),
                   ToolCall("update_record",{"table":"accounts","record_id":"a1","values":{"balance":20}}))
    trace=run_episode(task,MockPredictor(task,[first,stop]))
    assert trace["verification"]["success"] and trace["recovery_steps"]==2
    limited=run_episode(task,MockPredictor(task,[first,stop]),max_tool_calls=1)
    assert len(limited["events"])==1 and limited["invalid_tool_calls"]==2
    assert not limited["verification"]["success"]

def test_model_retry_classified_without_oracle_correction():
    task=task_named("sqlite_primary_key")
    call=ToolCall(**task["failed_action"])
    a=response(call)
    b=copy.deepcopy(a)
    b["choices"][0]["message"]["tool_calls"][0]["id"]="another"
    trace=run_episode(task,MockPredictor(task,[a,b,response(text="stop")]))
    assert trace["repeated_failures"]==2
    assert trace["unnecessary_tool_calls"]==2
    assert not trace["verification"]["success"]
    assert trace["events"][1]["call"]["arguments"]==task["failed_action"]["arguments"]

def test_online_explicit_only_default_temperature_and_secret_redaction(tmp_path):
    task=task_named("sqlite_primary_key")
    config=LLMConfig(model="qwen3.8-max",base_url="https://example.invalid/v1",max_requests=1)
    secret="sk-private-test-credential"
    received=[]
    def transport(payload):
        received.append(payload)
        result=response(text="done")
        result["choices"][0]["message"]["reasoning_content"]=secret
        return result
    with pytest.raises(ValueError,match="explicitly"):
        OnlinePredictor(config,secret)
    predictor=OnlinePredictor(config,secret,online=True,budget_path=tmp_path/"budget.json",transport=transport)
    result=predictor.predict(prompt_for(task,[],8))
    assert "temperature" not in received[0]
    assert secret not in json.dumps(result)
    assert secret not in (tmp_path/"budget.json").read_text()
    with pytest.raises(BudgetExceeded):
        predictor.predict(prompt_for(task,[],8))
    resumed=OnlinePredictor(config,secret,online=True,budget_path=tmp_path/"budget.json",transport=transport)
    assert resumed.budget.state["requests"]==1
    with pytest.raises(BudgetExceeded):
        resumed.predict(prompt_for(task,[],8))
    assert len(received)==1

@pytest.mark.parametrize("failure", ["timeout","429","500"])
def test_api_errors_charge_reservation_without_automatic_retry(failure):
    config=LLMConfig(model="q",base_url="https://example.invalid/v1",max_requests=1)
    def transport(payload):
        if failure=="timeout":
            raise TimeoutError()
        raise urllib.error.HTTPError("https://example.invalid/v1",int(failure),"error",{},None)
    predictor=OnlinePredictor(config,"private",online=True,transport=transport)
    result=predictor.predict({"messages":[],"tools":[]})
    assert result["error"]
    assert predictor.budget.state["requests"]==1
    assert predictor.budget.state["reserved_or_used_cost_cny"]>0

def test_sample_resume_and_failure_containment(tmp_path):
    tasks=generate_tasks(1)
    sampled=sample_tasks(tasks,100,7)
    assert all(t["split"]=="train" for t in sampled)
    assert sampled==sample_tasks(tasks,100,7)
    selected=[task_named("sqlite_primary_key"),task_named("sqlite_transaction")]
    def factory(task):
        if task["id"]==selected[0]["id"]:
            raise RuntimeError("do not print exception contents")
        return MockPredictor(task)
    traces, report=collect(selected,factory,tmp_path,{"source":"mock"})
    assert not traces[0]["verification"]["success"] and traces[1]["verification"]["success"]
    before=(tmp_path/"trajectories.jsonl").read_bytes()
    collect(selected,lambda _: (_ for _ in ()).throw(AssertionError()),tmp_path,{"source":"mock"},resume=True)
    assert (tmp_path/"trajectories.jsonl").read_bytes()==before
    with pytest.raises(ValueError,match="differs"):
        collect(selected,factory,tmp_path,{"source":"other"},resume=True)

def test_training_sources_test_exclusion_and_same_state_evidence(tmp_path):
    task=task_named("sqlite_primary_key")
    unsafe=response(ToolCall("write_file",{"path":"extra.txt","content":"bad"}))
    trace=run_episode(task,MockPredictor(task,[unsafe,response(text="stop")]))
    pair,reason=candidate_evidence(task,trace)
    assert pair and reason is None
    assert pair["evidence"]["chosen"]["start_state_id"]==pair["evidence"]["rejected"]["start_state_id"]
    assert pair["evidence"]["chosen"]["verification"]["success"]
    assert pair["evidence"]["rejected"]["candidate_verification"]["unauthorized_modifications"]>0
    good=run_episode(task,MockPredictor(task))
    test_task=task_named("sqlite_dependency")
    test_trace=run_episode(test_task,MockPredictor(test_task))
    report=prepare([task,test_task],[good,test_trace],tmp_path)
    assert report["sft_by_source"]["oracle_mock"]>0 and report["test_training_samples"]==0
    assert not (tmp_path/"llm_sft_train.jsonl").read_text()
    assert report["discard_reasons"]["not_train"]==1

def test_schema_legacy_update_name_unique():
    names=[x["function"]["name"] for x in schemas()]
    assert len(names)==len(set(names))==14
    assert names.count("update_record")==1

def test_metrics_invalid_denominator_and_api_exception():
    task=task_named("sqlite_primary_key")
    trace=run_episode(task,MockPredictor(task,[response(ToolCall("unknown",{})),response(text="done")]))
    assert summarize([trace])["invalid_tool_call_rate"]==1
    config=LLMConfig(model="q",base_url="https://example.invalid/v1")
    predictor=OnlinePredictor(config,"private",online=True,
        transport=lambda _: (_ for _ in ()).throw(RuntimeError("private")))
    result=predictor.predict({"messages":[],"tools":[]})
    assert result["error"]=={"type":"api_exception","exception_type":"RuntimeError"}

def test_collector_lock_and_trailing_partial_record(tmp_path):
    task=task_named("sqlite_primary_key")
    traces,_=collect([task],lambda t:MockPredictor(t),tmp_path,{"source":"mock"})
    path=tmp_path/"trajectories.jsonl"
    original=path.read_bytes()
    path.write_bytes(original+b'{"partial":')
    collect([task],lambda _: (_ for _ in ()).throw(AssertionError()),tmp_path,{"source":"mock"},resume=True)
    assert path.read_bytes()==original
    (tmp_path/".collection.lock").write_text("external_owner")
    with pytest.raises(FileExistsError):
        collect([task],lambda t:MockPredictor(t),tmp_path,{"source":"mock"},resume=True)

def test_successful_no_change_episode_does_not_teach_failed_retry(tmp_path):
    task=task_named("git_no_change")
    retry=response(ToolCall(**task["failed_action"]))
    trace=run_episode(task,MockPredictor(task,[retry,response(text="done")]))
    assert trace["verification"]["success"]
    result=prepare([task],[trace],tmp_path)
    rows=[json.loads(line) for line in (tmp_path/"oracle_mock_sft_train.jsonl").read_text().splitlines()]
    assert len(rows)==1 and not rows[0]["messages"][-1].get("tool_calls")
    assert result["discard_reasons"]["untrusted_sft_decision"]==1

def test_budget_unexecuted_tasks_are_reported_separately():
    task=task_named("sqlite_transaction")
    predictor=MockPredictor(task)
    def stopped(prompt):
        raise BudgetExceeded("exhausted")
    predictor.predict=stopped
    trace=run_episode(task,predictor)
    assert trace["collection_status"]=="budget_not_evaluated"
    report=summarize([trace])
    assert report["budget_not_evaluated_tasks"]==1
    assert report["decision_attempted_tasks"]==0
    assert report["attempted_recovery_success_rate"] is None

def test_failed_operation_is_not_labeled_as_successful_state_change():
    task=task_named("sqlite_transaction")
    bad=response(ToolCall("update_record",{"table":"accounts","record_id":"a1","values":{"balance":-2}}))
    trace=run_episode(task,MockPredictor(task,[bad,response(text="stop")]))
    assert trace["events"][1]["classification"]=="wrong_operation"
    assert summarize([trace])["tool_classifications"]=={"wrong_operation":1}
