"""Six focused CPU tests; real-template integration skips explicitly if official files are unavailable."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from recovery_agent.data_pipeline.common import read_jsonl
from recovery_agent.hardening.analysis import annotate
from recovery_agent.hardening.runner import replay
from recovery_agent.sft_preparation import trusted_decisions, validate_causal_prompts, target_for
from recovery_agent.qwen3_tokenizer import (
    Qwen3Adapter, TokenizerUnavailable, TemplateIncompatible, load_tokenizer, validate_messages)

ROOT = Path(__file__).resolve().parents[1]

def recorded(domain, teacher=False):
    """Use a few existing, fixed recovery records; no generator, API or mutated source files."""
    directory = ROOT/"outputs"/("v0.3b_teacher10" if teacher else "v0.3b_mock_verified")
    tasks = {t["id"]: t for t in read_jsonl(directory/"tasks.jsonl")}
    trace = next(t for t in read_jsonl(directory/"trajectories.jsonl")
                 if tasks[t["task_id"]]["domain"] == domain and t["split"] == "train"
                 and t["verification"]["success"])
    task = tasks[trace["task_id"]]
    replay(task, trace)
    annotate(task, trace)
    validate_causal_prompts(task, trace)
    return task, trace

def sample(domain, kind):
    task, trace = recorded(domain)
    rows, _ = trusted_decisions(task, trace, "fixed-reference", "fixed-test")
    return next(s for s in reversed(rows) if s["target_kind"] == kind and
                sum(m["role"] == "tool" for m in s["messages"][:-1]) >= 2)

@pytest.fixture(scope="module")
def official():
    """This is the actual AutoTokenizer/template; never a synthetic tokenizer."""
    path = os.environ.get("QWEN3_TOKENIZER_PATH", str(ROOT/"assets/tokenizers/Qwen3-1.7B"))
    try:
        tokenizer = load_tokenizer(path)
    except TokenizerUnavailable as exc:
        pytest.skip(f"REAL TOKENIZER INTEGRATION BLOCKED: {exc}")
    return Qwen3Adapter(tokenizer)

def test_sources_and_untrusted_decisions():
    task, oracle = recorded("sqlite")
    rows, _ = trusted_decisions(task, oracle, "oracle", "reference")
    assert rows and all(s["source"] == "oracle_reference" for s in rows)
    teacher_task, teacher = recorded("git", teacher=True)
    teacher_rows, _ = trusted_decisions(teacher_task, teacher, "teacher", "real")
    assert teacher_rows and all(s["source"] == "teacher" for s in teacher_rows)
    assert all("mock" not in s["model_id"] for s in teacher_rows)
    failed = copy.deepcopy(teacher)
    failed["verification"]["success"] = False
    assert trusted_decisions(teacher_task, failed, "failed", "real")[0] == []
    unsafe = copy.deepcopy(oracle)
    for event in unsafe["events"]:
        if event["phase"] == "recovery":
            event["behavior"]["label"] = "unsafe"
    retained, exclusions = trusted_decisions(task, unsafe, "unsafe", "reference")
    assert all(s["target_kind"] == "final_answer" for s in retained)
    assert exclusions["untrusted:unsafe"] > 0

def test_multi_tool_group_ids_and_order():
    task, trace = recorded("git")
    decision = copy.deepcopy(trace["decisions"][0])
    # Fixed grouping serialization test, not a claimed real model measurement.
    decision["raw_response"]["choices"][0]["message"]["tool_calls"] = [
        {"id": "fixed-a", "type": "function", "function": {"name": "git_status", "arguments": "{}"}},
        {"id": "fixed-b", "type": "function", "function": {"name": "git_diff", "arguments": "{}"}}]
    decision["parsed"]["kind"] = "tools"
    target = target_for(decision)
    messages = [{"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Diagnose Git."}, target,
                {"role": "tool", "name": "git_status", "tool_call_id": "fixed-a", "content": "{}"},
                {"role": "tool", "name": "git_diff", "tool_call_id": "fixed-b", "content": "{}"},
                {"role": "assistant", "content": "Recovery complete."}]
    bindings = validate_messages(messages, decision["prompt"]["tools"])
    assert [c["id"] for c in target["tool_calls"]] == ["fixed-a", "fixed-b"]
    assert [b["response_message_index"] for b in bindings] == [3, 4]
    assert all(b["group_size"] == 2 for b in bindings)
    bad = copy.deepcopy(messages)
    bad[3]["tool_call_id"] = "fixed-b"
    with pytest.raises(ValueError, match="ID/name/order"):
        validate_messages(bad, decision["prompt"]["tools"])

def test_real_assistant_only_mask(official):
    row = sample("git", "tool_call")
    encoded = official.encode(row)
    start, end = encoded["target_token_span"]
    assert encoded["labels"][:start] == [-100]*start
    assert encoded["labels"][start:end] == encoded["input_ids"][start:end]
    assert encoded["input_ids"][end-1] == official.tokenizer.eos_token_id
    assert all(x == -100 for x in encoded["labels"][end:])
    assert official.tokenizer.decode(encoded["input_ids"][start:end]).count("<tool_call>") >= 1
    assert all(label == -100 or label == token
               for token, label in zip(encoded["input_ids"], encoded["labels"]))

def test_real_tool_call_and_final_answer_labels(official):
    git = sample("git", "tool_call")
    sqlite = sample("sqlite", "final_answer")
    for row in (git, sqlite):
        encoded = official.encode(row)
        supervised = [x for x in encoded["labels"] if x != -100]
        assert len(supervised) > 1 and supervised[-1] == official.tokenizer.eos_token_id
        text = official.tokenizer.decode(supervised)
        if row["target_kind"] == "tool_call":
            assert "<tool_call>" in text and "</tool_call>" in text
        else:
            assert row["messages"][-1]["content"] in text and "<tool_response>" not in text
    assert official.capabilities["tool_call_groups"] and not official.capabilities["call_ids_rendered"]

def test_causality_hidden_fields_and_split_guard():
    task, trace = recorded("sqlite")
    rows, _ = trusted_decisions(task, trace, "trusted", "reference")
    assert rows
    for row in rows:
        decision = trace["decisions"][row["decision_index"]]
        assert row["messages"][:-1] == decision["prompt"]["messages"]
        assert row["tools"] == decision["prompt"]["tools"]
        assert not {"expected", "reference_plan", "audit", "verification"} & row["messages"][0].keys()
        # Target results exist in evidence hashes only, never appended as future tool messages.
        assert row["messages"][-1]["role"] == "assistant"
    modified = copy.deepcopy(trace)
    modified["decisions"][0]["prompt"]["messages"].append(
        {"role": "user", "content": "PRIVATE_VERIFIER_FUTURE_RESULT"})
    with pytest.raises(ValueError, match="hidden fields or future"):
        validate_causal_prompts(task, modified)
    evaluation = copy.deepcopy(task)
    evaluation.update(split="test", eval_group="iid_test")
    assert trusted_decisions(evaluation, trace, "test", "reference")[0] == []

def test_explicit_tokenizer_template_and_truncation_failures(official, tmp_path):
    with pytest.raises(TokenizerUnavailable):
        load_tokenizer(tmp_path/"missing-official-tokenizer")
    with pytest.raises(TemplateIncompatible):
        Qwen3Adapter(SimpleNamespace(chat_template=""))
    # Wrong role/EOS rendering is a rejected capability probe, not a substitute integration.
    incompatible = SimpleNamespace(chat_template="ignored",
        apply_chat_template=lambda *a, **kw: "NO_TOOL_FORMAT" if not kw.get("tokenize") else [1])
    with pytest.raises(TemplateIncompatible, match="support"):
        Qwen3Adapter(incompatible)
    row = sample("sqlite", "final_answer")
    small = Qwen3Adapter(official.tokenizer, max_seq_length=8)
    preserved = small.encode(row)
    assert preserved["over_max_seq_length"] and not preserved["truncated"]
    assert len(preserved["input_ids"]) > 8
    with pytest.raises(ValueError, match="truncation refused"):
        small.encode(row, enforce_limit=True)
