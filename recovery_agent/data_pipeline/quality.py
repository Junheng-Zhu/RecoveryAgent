import json
from collections import Counter
from pathlib import Path
from ..runner import replay
from .common import read_jsonl, fingerprint, canonical, VERSION
from .dataset import generate_dataset
from .trajectories import decisions_from_trace, build_sft
from .preferences import verify_pair
from .serialization import parse_action, tool_schemas

# 将失败的质量条件转为带说明的 ValueError，使验证流程立即停止。
def require(condition, message):
    if not condition:
        raise ValueError(message)

# 检查 ID 或内容指纹是否重复，避免重复样本被计入训练数量。
def unique(values, message):
    require(len(values) == len(set(values)), message)

# 校验角色顺序、调用 ID 与结果配对，以及工具 Schema 是否对应当前运行时。
# 最终 assistant 监督动作允许尚无 tool result；历史调用必须已有匹配结果。
def check_conversation(messages, tools):
    require(tools == tool_schemas(), "Tool schemas differ from runtime contract")
    require(len(messages) >= 4 and messages[0]["role"] == "system" and
            messages[1]["role"] == "user", "Invalid conversation prefix")
    pending, ids = set(), set()
    for message in messages:
        role = message.get("role")
        require(role in ("system", "user", "assistant", "tool"), "Unknown message role")
        if role == "assistant" and message.get("tool_calls"):
            require(not pending and len(message["tool_calls"]) == 1, "Unmatched tool calls")
            call = message["tool_calls"][0]
            require(call["type"] == "function" and call["id"] not in ids, "Invalid call ID")
            require(isinstance(json.loads(call["function"]["arguments"]), dict), "Arguments not object")
            ids.add(call["id"])
            pending.add(call["id"])
        elif role == "tool":
            require(message["tool_call_id"] in pending, "Orphan tool result")
            pending.remove(message["tool_call_id"])
            json.loads(message["content"])
        else:
            require(not pending, "Conversation skips a tool result")
    # The final assistant target is allowed to leave exactly one pending tool call.
    require(not pending or (messages[-1]["role"] == "assistant" and
                            bool(messages[-1].get("tool_calls"))), "Incomplete history")

# 按保存配置重新生成任务，检查切分、重复、轨迹重放、公开因果投影及偏好执行证据。
# SFT 必须是轨迹的精确前缀投影，不靠关键词搜索推断是否泄漏答案。
def validate_data(tasks, trajectories, samples, pairs, config):
    expected, _ = generate_dataset(config)
    require(tasks == expected, "Tasks/config/version do not reproduce")
    require(all(t["schema_version"] == VERSION for t in tasks), "Bad task version")
    unique([t["id"] for t in tasks], "Duplicate task IDs")
    unique([fingerprint({k: v for k, v in t.items() if k not in ("id", "split")})
            for t in tasks], "Duplicate task payloads")
    by_id = {t["id"]: t for t in tasks}
    groups = {}
    for split in ("train", "val", "test"):
        rows = [t for t in tasks if t["split"] == split]
        require(len(rows) == config["counts"][split], "Incorrect split count")
        groups[split] = (
            {t["template_family"] for t in rows},
            {t["environment_config"]["id"] for t in rows},
            {fingerprint(t["environment_config"]) for t in rows},
        )
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        require(all(not left & right for left, right in zip(groups[a], groups[b])), "Split leakage")
    require(len(trajectories) == len(tasks), "Missing trajectories")
    unique([t["task_id"] for t in trajectories], "Duplicate trajectories")
    by_trace = {t["task_id"]: t for t in trajectories}
    replay_count = 0
    for row in trajectories:
        task = by_id[row["task_id"]]
        require(row["split"] == task["split"] and row["template_family"] == task["template_family"],
                "Trajectory metadata mismatch")
        require(row["trace"]["task_id"] == task["id"] and row["trace"]["goal"] == task["goal"] and
                row["trace"]["split"] == task["split"], "Raw trace provenance mismatch")
        require(row["verification"] == row["trace"]["verification"] and
                row["recovery_steps"] == row["trace"]["recovery_steps"], "Trajectory verdict mismatch")
        require(row["replay_verified"] is True and row["verification"]["success"] is True,
                "Unsuccessful Oracle trajectory")
        replay(task, row["trace"])
        replay_count += 1
        require(row["decisions"] == decisions_from_trace(task, row["trace"]),
                "Decision input or target leakage/tampering")
    # 重新构建训练投影再精确比对，覆盖偷偷插入未来结果等字段级泄漏。
    expected_samples, _ = build_sft(trajectories)
    require(samples == expected_samples, "SFT is not an exact causal projection of trajectories")
    unique([s["sample_id"] for s in samples], "Duplicate SFT IDs")
    unique([fingerprint({"messages": s["messages"], "tools": s["tools"]}) for s in samples],
           "Duplicate SFT content")
    train_prompts, val_prompts = set(), set()
    for sample in samples:
        task = by_id[sample["task_id"]]
        require(task["split"] in ("train", "val") and sample["split"] == task["split"], "Test used in SFT")
        require(sample["loss_scope"] == "final_assistant_only", "Unsafe history supervision")
        check_conversation(sample["messages"], sample["tools"])
        parse_action(sample["messages"][-1])
        prompt = {"messages": sample["messages"][:-1], "tools": sample["tools"]}
        require(fingerprint(prompt) == sample["prompt_id"], "Prompt fingerprint mismatch")
        (train_prompts if sample["split"] == "train" else val_prompts).add(sample["prompt_id"])
    require(not train_prompts & val_prompts, "SFT prompt overlap across splits")
    unique([p["sample_id"] for p in pairs], "Duplicate preference IDs")
    unique([fingerprint({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]})
            for p in pairs], "Duplicate preference content")
    verified_pairs = 0
    preference_prompts = {"train": set(), "val": set()}
    for pair in pairs:
        task = by_id[pair["task_id"]]
        require(task["split"] in ("train", "val") and pair["split"] == task["split"],
                "Test used in preferences")
        # 重新执行而非仅读取 success 标记；伪造 Prompt 或分支证据会被检测。
        verify_pair(task, by_trace[task["id"]], pair)
        require(pair["chosen"] != pair["rejected"], "Identical candidate pair")
        preference_prompts[pair["split"]].add(pair["prompt_id"])
        verified_pairs += 1
    require(not preference_prompts["train"] & preference_prompts["val"], "Preference split leakage")
    return {"passed": True, "tasks": len(tasks), "trajectory_replays_verified": replay_count,
            "sft_samples": dict(Counter(s["split"] for s in samples)),
            "preference_pairs": dict(Counter(p["split"] for p in pairs)),
            "preference_pairs_execution_verified": verified_pairs,
            "duplicate_samples": 0, "split_leakage": False,
            "causal_inputs_verified": len(samples),
            "leakage_validation": "exact public projection; excludes hidden seeds/tokens, expected state, plans and future results",
            "tools_schema_verified": True, "test_training_samples": 0}

# 重新读取落盘 JSONL 与 manifest，先核对配置和文件哈希，再执行完整数据质量检查。
# 显式拒绝 test SFT/Preference 文件，确保测试数据不进入训练导出。
def validate_directory(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["dataset_version"] == VERSION, "Unsupported manifest version")
    require(manifest["config_hash"] == fingerprint(manifest["config"]), "Config hash mismatch")
    require(json.loads((directory / "config.json").read_text(encoding="utf-8")) ==
            manifest["config"], "Persisted config differs")
    # Verify every generated payload, including JSONL syntax and format via loaders below.
    for relative, expected_hash in manifest["artifacts"].items():
        import hashlib
        actual = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        require(actual == expected_hash, f"Artifact hash mismatch: {relative}")
    tasks = []
    trajectories = []
    samples = []
    pairs = []
    for split in ("train", "val", "test"):
        rows = read_jsonl(directory / "tasks" / f"{split}.jsonl")
        require(all(t["split"] == split for t in rows), "Task file split mismatch")
        tasks += rows
        rows = read_jsonl(directory / "trajectories" / f"{split}.jsonl")
        require(all(t["split"] == split for t in rows), "Trajectory file split mismatch")
        trajectories += rows
        if split != "test":
            rows = read_jsonl(directory / "sft" / f"{split}.jsonl")
            require(all(t["split"] == split for t in rows), "SFT file split mismatch")
            samples += rows
            rows = read_jsonl(directory / "preferences" / f"{split}.jsonl")
            require(all(t["split"] == split for t in rows), "Preference file split mismatch")
            pairs += rows
    require(not (directory / "sft/test.jsonl").exists() and
            not (directory / "preferences/test.jsonl").exists(), "Unexpected test training file")
    report = validate_data(tasks, trajectories, samples, pairs, manifest["config"])
    report["artifacts_hashed"] = len(manifest["artifacts"])
    return report
