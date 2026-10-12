import copy
import json
import pytest
from dataclasses import asdict
from recovery_agent.models import ToolCall
from recovery_agent.runner import run_episode
from recovery_agent.data_pipeline.common import canonical, read_jsonl
from recovery_agent.data_pipeline.dataset import DEFAULT_CONFIG, generate_dataset, check_config
from recovery_agent.data_pipeline.serialization import (
    ConversationalPolicy, build_prompt, public_result, parse_action, tool_schemas, stop_message)
from recovery_agent.data_pipeline.trajectories import collect_trajectory, build_sft, decisions_from_trace
from recovery_agent.data_pipeline.preferences import (
    branch_rollout, build_preferences, verify_pair, preference_reason, candidates)
from recovery_agent.data_pipeline.quality import validate_data, validate_directory
from recovery_agent.data_pipeline.pipeline import run_pipeline, compare_directories

# 保留正式切分和环境配置，只缩小任务数以加快集成测试。
def small_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["counts"] = {"train": 16, "val": 8, "test": 16}
    return config

# 按基础模板、故障和环境布局选取代表任务，覆盖每种组合而不重复运行全部实例。
def representatives():
    tasks, _ = generate_dataset()
    selected = {}
    for task in tasks:
        key = (task["base_template"], task["fault_type"], task["environment_config"]["id"])
        selected.setdefault(key, task)
    return list(selected.values())

# 验证 v0.2 的 800/200/200 规模、固定 seed 重现、故障均衡及同机制模板别名不跨 split。
def test_exact_versioned_split_counts_and_no_alias_leakage():
    tasks, config = generate_dataset()
    assert tasks == generate_dataset()[0]
    assert len(tasks) == 1200 and len({t["id"] for t in tasks}) == 1200
    assert {s: sum(t["split"] == s for t in tasks) for s in ("train", "val", "test")} == {
        "train": 800, "val": 200, "test": 200}
    for split in ("train", "val", "test"):
        rows = [t for t in tasks if t["split"] == split]
        assert len({t["fault_type"] for t in rows}) == 4
        assert {sum(t["fault_type"] == f for t in rows) for f in
                ("parameter", "state_missing", "transient", "planning_order")} == {len(rows) // 4}
    assert {t["split"] for t in tasks if t["base_template"] in ("copy_note", "copy_receipt")} == {"train"}
    assert {t["split"] for t in tasks if t["base_template"].startswith(("sum", "ratio"))} == {"test"}
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    assert generate_dataset(changed)[0] != tasks

# 验证族、环境 ID/布局重叠和非法数量或 seed 被配置校验拒绝。
@pytest.mark.parametrize("change", ["family", "environment", "layout", "count", "seed"])
def test_invalid_configs_rejected(change):
    config = small_config()
    if change == "family":
        config["families"]["val"] = ["copy"]
    elif change == "environment":
        config["environments"]["val"][0]["id"] = "layout_a"
    elif change == "layout":
        config["environments"]["val"][0] = {**config["environments"]["train"][0], "id": "different_id"}
    elif change == "count":
        config["counts"]["train"] = 15
    else:
        config["seed"] = "42"
    with pytest.raises(ValueError):
        check_config(config)

# 验证轨迹成功、停止标签、test 导出排除，以及每一步训练 Prompt 与在线适配器完全一致。
@pytest.mark.parametrize("task", representatives(), ids=lambda t: t["id"])
def test_oracle_collection_causal_sft_and_online_contract(task):
    trajectory = collect_trajectory(task)
    assert trajectory["verification"]["success"]
    assert trajectory["replay_verified"]
    assert trajectory["decisions"][-1]["oracle_action"] == stop_message()
    samples, skipped = build_sft([trajectory])
    if task["split"] == "test":
        assert not samples and skipped == {"test_excluded": 1}
    else:
        assert len(samples) == trajectory["recovery_steps"] + 1
        for sample in samples:
            parse_action(sample["messages"][-1])
            assert sample["tools"] == tool_schemas()
    decisions = iter(trajectory["decisions"])
    # 测试替身逐步比对公开 Prompt 后返回记录的动作，仅验证协议一致性，不模拟模型性能。
    def predictor(prompt):
        decision = next(decisions)
        assert prompt == decision["prompt"]
        return decision["oracle_action"]
    result = run_episode(task, ConversationalPolicy(predictor))
    assert result["verification"]["success"]
    assert result["events"] == trajectory["trace"]["events"]
    with pytest.raises(StopIteration):
        next(decisions)

# 扰动隐藏 seed、参考计划和未来结果，验证当前决策 Prompt 不受这些不可见信息影响。
def test_hidden_seed_answer_and_future_result_not_in_inputs():
    task = generate_dataset()[0][0]
    trajectory = collect_trajectory(task)
    changed = copy.deepcopy(task)
    changed["expected_state"] = {"secret": "LEAK_MARKER"}
    changed["fault"]["seed"] = 998877
    changed["reference_plan"] = [{"hidden": "LEAK_MARKER"}]
    changed["goal"]["expected_state"] = "LEAK_MARKER"
    trace = copy.deepcopy(trajectory["trace"])
    trace["events"][0]["result"]["error"]["details"].update(
        fault_token="LEAK_MARKER", seed=998877, expected_state="LEAK_MARKER")
    first = decisions_from_trace(changed, trace)[0]["prompt"]
    assert first == trajectory["decisions"][0]["prompt"]
    assert "LEAK_MARKER" not in canonical(first)
    assert "fault_token" not in canonical(first)
    assert "expected_state" not in canonical(first)
    trace["events"][-1]["result"]["value"] = "FUTURE_MARKER"
    assert decisions_from_trace(task, trace)[0]["prompt"] == first
    assert "FUTURE_MARKER" not in canonical(first)

# 验证偏好分支可重新执行，并将续程重复写入导致的伪偏好丢弃；篡改证据也应报错。
def test_preferences_same_state_actual_execution_and_ambiguous_discard():
    tasks, _ = generate_dataset(small_config())
    task = next(t for t in tasks if t["template_family"] == "record_update"
                and t["fault_type"] == "transient")
    trajectory = collect_trajectory(task)
    pairs, dropped = build_preferences(task, trajectory)
    assert pairs and all(verify_pair(task, trajectory, pair) for pair in pairs)
    decision = trajectory["decisions"][0]
    prefix = trajectory["trace"]["events"][:decision["event_index"]]
    suffix = trajectory["trace"]["events"][decision["event_index"] + 1:]
    chosen = branch_rollout(task, prefix, decision["oracle_action"], suffix)
    retry = dict(candidates(task, decision))["retry_failed"]
    rejected = branch_rollout(task, prefix, retry, suffix)
    # The retry itself succeeds; a duplicate write in the fixed suffix is not evidence
    # that the candidate's side effect is irreparably wrong.
    assert rejected["candidate_verification"]["success"]
    assert rejected["verification"]["extra_side_effects"] == 1
    assert preference_reason(chosen, rejected) == (None, "ambiguous_recoverable_failure")
    forged = copy.deepcopy(pairs[0])
    forged["evidence"]["rejected"]["verification"]["success"] = True
    with pytest.raises(ValueError, match="evidence"):
        verify_pair(task, trajectory, forged)
    forged = copy.deepcopy(pairs[0])
    forged["prompt"]["messages"][1]["content"] = "{}"
    with pytest.raises(ValueError, match="prompt"):
        verify_pair(task, trajectory, forged)

# 验证 test 轨迹不生成 SFT 样本或偏好候选对。
def test_test_never_enters_training_builders():
    task = next(t for t in generate_dataset()[0] if t["split"] == "test")
    trajectory = collect_trajectory(task)
    assert build_sft([trajectory]) == ([], {"test_excluded": 1})
    assert build_preferences(task, trajectory) == ([], {"test_excluded": 1})

# 验证端到端生成、落盘重新读取、两次逐字节一致，以及拒绝覆盖和检测文件篡改。
def test_one_command_pipeline_roundtrip_reproduction_and_corruption(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    report, manifest, quality = run_pipeline(first, small_config())
    assert report["quality"]["passed"] and quality["trajectory_replays_verified"] == 40
    assert quality["test_training_samples"] == 0
    assert quality["preference_pairs_execution_verified"] > 0
    run_pipeline(second, small_config())
    assert compare_directories(first, second)["identical"]
    assert validate_directory(first)["passed"]
    assert not (first / "sft/test.jsonl").exists()
    assert not (first / "preferences/test.jsonl").exists()
    with pytest.raises(FileExistsError):
        run_pipeline(first, small_config())
    path = first / "sft/train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_directory(first)

# 验证质量检查拒绝手工插入的未来信息与重复样本。
def test_quality_rejects_future_result_and_duplicate_sample():
    tasks, config = generate_dataset(small_config())
    trajectories = [collect_trajectory(t) for t in tasks]
    samples, _ = build_sft(trajectories)
    assert validate_data(tasks, trajectories, samples, [], config)["passed"]
    bad = copy.deepcopy(samples)
    bad[0]["messages"].insert(-1, {"role": "user", "content": "future output"})
    with pytest.raises(ValueError, match="causal projection"):
        validate_data(tasks, trajectories, bad, [], config)
    with pytest.raises(ValueError):
        validate_data(tasks, trajectories, samples + [samples[0]], [], config)

# 验证空行、非法 JSON、非对象行和 NaN 都不能通过严格 JSONL 读取。
@pytest.mark.parametrize("text", ["[]\n", "{bad}\n", "\n", '{"x":NaN}\n'])
def test_invalid_jsonl_rejected(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(path)
