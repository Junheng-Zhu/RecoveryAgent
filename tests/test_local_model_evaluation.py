"""Mock text outputs exercise the existing native Runtime; this is not GPU/model validation."""
import json
from pathlib import Path
import pytest
from recovery_agent.training.local_model import LocalConfig, LocalPredictor, hermes_message
from recovery_agent.training.evaluation import select_tasks, comparison_settings, benchmark_summary
from recovery_agent.hardening.tasks import validate_frozen
from recovery_agent.hardening.protocol import schema_hash
from recovery_agent.hardening.runner import run_episode
from recovery_agent.stateful.protocol import parse_response
from recovery_agent.stateful.predictor import BudgetExceeded

ROOT = Path(__file__).resolve().parents[1]

class FixedEngine:
    """A fixed CPU text source, not a tokenizer, pretrained model or simulated tool runtime."""
    model_id = "cpu-test-fixed-hermes"
    context_limit = 40960
    def __init__(self, texts):
        self.texts = iter(texts)
    def set_seed(self, seed):
        self.seed = seed
    def prompt_ids(self, prompt):
        self.prompt = prompt
        return [1, 2, 3]
    def generate(self, ids, config):
        text = next(self.texts)
        return {"text": text, "raw_text": text, "generated_token_ids": [4, 5],
                "finish_reason": "stop", "latency_ms": 0.0}

def tasks():
    return validate_frozen(ROOT/"data/v0.3b_seed42_final", schema_hash())[0]

def stage_task():
    return next(t for t in tasks() if t["id"] == "0.3-B-git_stage_missing-000")

def text_call(name, arguments):
    return "<tool_call>"+json.dumps({"name": name, "arguments": arguments})+"</tool_call>"

def test_base_adapter_configuration():
    assert LocalConfig().model_mode == "base"
    assert LocalConfig(model_mode="adapter", adapter="/data/checkpoints/adapter").model_mode == "adapter"
    with pytest.raises(ValueError):
        LocalConfig(adapter="unexpected")
    with pytest.raises(ValueError):
        LocalConfig(model_mode="adapter")
    with pytest.raises(ValueError):
        LocalConfig(max_tokens=0)

def test_strict_tool_json_grouping_and_no_repairs():
    text = text_call("git_status", {})+"\n"+text_call("git_diff", {})
    message = hermes_message(text, 1)
    parsed = parse_response({"choices": [{"message": message, "finish_reason": "tool_calls"}]})
    assert [c["call"]["name"] for c in parsed["calls"]] == ["git_status", "git_diff"]
    assert len({c["call"]["call_id"] for c in parsed["calls"]}) == 2
    assert hermes_message("Recovery complete.", 2)["content"] == "Recovery complete."
    with pytest.raises(ValueError):
        hermes_message("<tool_call>{bad}</tool_call>", 3)
    with pytest.raises(ValueError):
        hermes_message("<tool_call>{}", 3)
    message = hermes_message(text_call("git_add", "[not-an-object]"), 4)
    parsed = parse_response({"choices": [{"message": message, "finish_reason": "tool_calls"}]})
    assert parsed["calls"][0]["call"] is None  # never rewritten as a valid/Oracle action

def test_local_predictor_runtime_recovery_chain():
    task = stage_task()
    engine = FixedEngine([text_call("git_add", {"path": "case_000/report.txt"}),
                          text_call("git_commit", {"message": "publish"}), "Recovery complete."])
    trace = run_episode(task, LocalPredictor(engine, LocalConfig(max_tokens=16)),
                        stop_protocol="termination-v0.3c-2")
    assert trace["verification"]["success"] and trace["replay_verified"]
    assert trace["termination_reason"] == "assistant_text" and trace["recovery_steps"] == 2
    assert trace["decisions"][0]["raw_response"]["raw_generation"]["text"].startswith("<tool_call>")
    assert not {"expected", "reference_plan", "audit", "verification"} & engine.prompt.keys()

def test_ever_success_and_mutation_ledger_metrics():
    task = stage_task()
    engine = FixedEngine([text_call("git_add", {"path": "case_000/report.txt"}),
                          text_call("git_commit", {"message": "publish"}),
                          text_call("git_add", {"path": "case_000/report.txt"}), "Recovery complete."])
    trace = run_episode(task, LocalPredictor(engine, LocalConfig(max_tokens=16)),
                        stop_protocol="termination-v0.3c-2")
    report, _ = benchmark_summary([task], [trace])
    metrics = report["metrics"]
    assert metrics["ever_verified_success_rate"]["value"] == 1
    assert metrics["final_recovery_success_rate"]["value"] == 0
    assert metrics["autonomous_completion_rate"]["value"] == 0
    assert metrics["mutation_budget_exceeded_rate"]["value"] == 1
    assert metrics["unauthorized_mutation_rate"]["value"] == 0
    assert metrics["post_success_tool_calls"]["total"] == 1

def test_budget_invalid_output_and_split_isolation():
    predictor = LocalPredictor(FixedEngine(["<tool_call>{bad}</tool_call>"]), LocalConfig(max_tokens=16, max_requests=1))
    received = predictor.predict({"messages": [], "tools": []})
    assert received["format_error"] and parse_response(received["raw_response"])["calls"][0]["call"] is None
    with pytest.raises(BudgetExceeded):
        predictor.predict({"messages": [], "tools": []})
    frozen = tasks()
    assert len(select_tasks(frozen, "train_smoke")) == 3
    assert {t["domain"] for t in select_tasks(frozen, "train_smoke")} == {"git", "sqlite", "mixed"}
    assert [len(select_tasks(frozen, split)) for split in ("iid_validation", "iid_test", "ood")] == [13, 13, 18]
    for split in ("iid_validation", "iid_test", "ood"):
        assert all(t["split"] != "train" for t in select_tasks(frozen, split))
    base = LocalConfig()
    sft = LocalConfig(model_mode="adapter", adapter="/data/checkpoints/adapter")
    selected = select_tasks(frozen, "iid_test")
    assert comparison_settings(selected, base, 12, 16, "termination-v0.3c-2", False, 42) == \
           comparison_settings(selected, sft, 12, 16, "termination-v0.3c-2", False, 42)
