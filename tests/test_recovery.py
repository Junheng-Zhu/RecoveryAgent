import copy
import json
import pytest
from recovery_agent.dataset import generate_tasks
from recovery_agent.environment import Environment
from recovery_agent.models import ToolCall
from recovery_agent.policy import RulePolicy, RetryPolicy, StopPolicy
from recovery_agent.oracle import Oracle
from recovery_agent.runner import run_episode, replay, evaluate
from recovery_agent.verifier import Verifier

@pytest.mark.parametrize("task", generate_tasks(), ids=lambda t: t["id"])
def test_smoke_rule_oracle_replay(task):
    original = copy.deepcopy(task)
    trace = run_episode(task, RulePolicy())
    assert trace["verification"]["success"], trace
    assert replay(task, trace)
    assert trace == run_episode(task, RulePolicy())
    oracle = run_episode(task, Oracle(task), "oracle_certification")
    assert oracle["verification"]["success"], oracle
    assert replay(task, oracle)
    assert task == original

def test_family_split_and_1200_unique_deterministic_tasks():
    tasks = generate_tasks(1200)
    assert tasks == generate_tasks(1200)
    assert len({t["id"] for t in tasks}) == 1200
    families = {s: {t["template_family"] for t in tasks if t["split"] == s}
                for s in ("train", "val", "test")}
    assert not families["train"] & families["val"]
    assert not families["train"] & families["test"]
    assert not families["val"] & families["test"]
    assert {f: sum(t["fault_type"] == f for t in tasks) for f in
            ("parameter", "state_missing", "transient", "planning_order")} == {
                "parameter": 300, "state_missing": 300, "transient": 300, "planning_order": 300}

def test_policy_input_excludes_privileged_answers():
    task = generate_tasks()[0]
    class Spy(StopPolicy):
        def act(self, observation):
            assert set(vars(observation)) == {"goal", "failed_action", "observations", "remaining_steps"}
            assert not hasattr(observation, "expected_state")
            assert not hasattr(observation, "reference_plan")
            observation.goal.clear()
            return None
    run_episode(task, Spy())
    assert task["goal"]
    with pytest.raises(ValueError, match="forbidden"):
        run_episode(task, Oracle(task))

def test_verifier_rejects_unauthorized_transient_and_duplicate_writes():
    task = generate_tasks()[0]
    with Environment(task["initial_state"]) as env:
        env.execute(ToolCall(**task["reference_plan"][-1]))
        assert Verifier().check(task, env).success
        env.execute(ToolCall("write_file", {"path": "protected.txt", "content": "bad"}))
        env.execute(ToolCall("write_file", {"path": "protected.txt", "content": "DO NOT MODIFY"}))
        assert not Verifier().check(task, env).success
        assert Verifier().check(task, env).state_matches
        env.reset()
        env.execute(ToolCall(**task["reference_plan"][-1]))
        env.execute(ToolCall(**task["reference_plan"][-1]))
        assert Verifier().check(task, env).extra_side_effects == 1
        assert not Verifier().check(task, env).success

def test_verifier_rejects_wrong_intermediate_authorized_value():
    task = generate_tasks()[0]
    with Environment(task["initial_state"]) as env:
        env.execute(ToolCall("write_file", {"path": "result.txt", "content": "wrong"}))
        env.execute(ToolCall(**task["reference_plan"][-1]))
        verdict = Verifier().check(task, env)
        assert verdict.state_matches and not verdict.success

def test_replay_detects_tampering_and_budget_exhaustion():
    task = generate_tasks()[0]
    trace = run_episode(task, RulePolicy())
    trace["events"][0]["result"]["error"]["code"] = "forged"
    with pytest.raises(ValueError, match="mismatch"):
        replay(task, trace)
    task["max_recovery_steps"] = 1
    assert not run_episode(task, RulePolicy())["verification"]["success"]

def test_reports_negative_baseline_and_jsonl(tmp_path):
    tasks = generate_tasks()
    report = evaluate(tasks, lambda _: RulePolicy(), tmp_path)
    assert report["recovery_success_rate"] == 1
    assert report["extra_side_effects"] == 0
    lines = (tmp_path / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert all(not run_episode(t, StopPolicy())["verification"]["success"] for t in tasks)
    assert run_episode(tasks[2], RetryPolicy())["verification"]["success"]

def test_oracle_all_expanded_tasks():
    for task in generate_tasks(1200):
        assert run_episode(task, Oracle(task), "oracle_certification")["verification"]["success"]

def test_verifier_detects_direct_tampering():
    task = generate_tasks()[16]
    with Environment(task["initial_state"]) as env:
        env.tables["backup_users"].clear()
        assert Verifier().check(task, env).unauthorized_modifications > 0

def test_retry_cannot_fix_parameters_or_planning_order():
    tasks = generate_tasks()
    for index in (0, 3):
        trace = run_episode(tasks[index], RetryPolicy())
        assert not trace["verification"]["success"]
        assert trace["repeated_failures"] > 0
        assert trace["recovery_steps"] == tasks[index]["max_recovery_steps"]

def test_runner_injection_at_specified_step():
    task = generate_tasks()[2]
    task["prefix_actions"] = [{"name": "list_files", "arguments": {}}]
    task["fault"]["step"] = 2
    trace = run_episode(task, RulePolicy())
    assert trace["events"][0]["phase"] == "prefix"
    assert trace["events"][0]["result"]["ok"]
    assert not trace["events"][1]["result"]["ok"]
    assert trace["verification"]["success"]
    assert replay(task, trace)
