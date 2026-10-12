import copy
from collections import Counter
from ..oracle import Oracle
from ..runner import run_episode, replay
from .common import VERSION
from .serialization import build_prompt, public_goal, public_result, visible_history, action_message, stop_message, prompt_id

# 按每个恢复决策的历史前缀构造公开输入，当前 Oracle 动作只作为标签。
# 当前动作的结果另存为 tool_result，不放入该步 Prompt；最后追加显式停止目标。
def decisions_from_trace(task, trace):
    decisions = []
    recovery_count = 0
    for index, event in enumerate(trace["events"]):
        if event["phase"] != "recovery":
            continue
        # 只截取当前动作之前的历史；不能把当前或之后的 Tool Result 放入输入。
        prefix = trace["events"][:index]
        prompt = build_prompt(task["goal"], prefix, task["max_recovery_steps"] - recovery_count)
        last = public_result(prefix[-1]["result"])
        decisions.append({
            "decision_index": recovery_count, "event_index": index,
            "goal": public_goal(task["goal"]), "visible_history": visible_history(prefix),
            "error_feedback": last["error"], "tools": prompt["tools"],
            "prompt": prompt, "prompt_id": prompt_id(prompt),
            "oracle_action": action_message(event["call"]),
            "tool_result": public_result(event["result"]),
        })
        recovery_count += 1
    prompt = build_prompt(task["goal"], trace["events"], task["max_recovery_steps"] - recovery_count)
    decisions.append({
        "decision_index": recovery_count, "event_index": len(trace["events"]),
        "goal": public_goal(task["goal"]), "visible_history": visible_history(trace["events"]),
        "error_feedback": public_result(trace["events"][-1]["result"])["error"],
        "tools": prompt["tools"], "prompt": prompt, "prompt_id": prompt_id(prompt),
        "oracle_action": stop_message(), "tool_result": None,
    })
    return decisions

# 调用现有 Oracle 在真实模拟环境执行，独立 Verifier 后立即 Replay，再提取逐步决策。
# 原始 trace 属于特权审计数据，训练时只能使用对应的公开投影。
def collect_trajectory(task):
    trace = run_episode(task, Oracle(task), "oracle_certification")
    checked = replay(task, trace)
    return {
        "schema_version": VERSION, "task_id": task["id"], "split": task["split"],
        "template_family": task["template_family"],
        "environment_config_id": task["environment_config"]["id"],
        "fault_type": task["fault_type"], "trace": trace,
        "replay_verified": checked, "verification": copy.deepcopy(trace["verification"]),
        "recovery_steps": trace["recovery_steps"],
        "decisions": decisions_from_trace(task, trace) if trace["verification"]["success"] else [],
    }

# 仅导出 train/val 中成功且已重放的轨迹，为每个动作及停止决策生成独立样本。
# 最后一条 assistant 是监督目标；历史失败动作不应参与正例损失计算。
def build_sft(trajectories):
    samples, skipped = [], Counter()
    for trajectory in trajectories:
        if trajectory["split"] == "test":
            skipped["test_excluded"] += 1
            continue
        if not trajectory["verification"]["success"] or not trajectory["replay_verified"]:
            skipped["unsuccessful_or_unverified"] += 1
            continue
        for decision in trajectory["decisions"]:
            samples.append({
                "schema_version": VERSION,
                "sample_id": f"{trajectory['task_id']}:{decision['decision_index']}",
                "task_id": trajectory["task_id"], "split": trajectory["split"],
                "decision_index": decision["decision_index"], "prompt_id": decision["prompt_id"],
                "messages": copy.deepcopy(decision["prompt"]["messages"]) + [copy.deepcopy(decision["oracle_action"])],
                "tools": copy.deepcopy(decision["tools"]),
                # 训练 collator 必须据此只监督最后一条 assistant；本模块不执行训练。
                "loss_scope": "final_assistant_only",
            })
    return samples, dict(skipped)

# 按 split 统计成功率、重放数、恢复调用/决策长度以及任务级副作用。
def trajectory_statistics(trajectories):
    result = {}
    for split in ("train", "val", "test"):
        rows = [t for t in trajectories if t["split"] == split]
        n = len(rows)
        result[split] = {
            "count": n, "successful": sum(t["verification"]["success"] for t in rows),
            "success_rate": sum(t["verification"]["success"] for t in rows) / n if n else 0,
            "replay_verified": sum(t["replay_verified"] for t in rows),
            "recovery_length_distribution": dict(sorted(Counter(t["recovery_steps"] for t in rows).items())),
            "decision_length_distribution": dict(sorted(Counter(len(t["decisions"]) for t in rows).items())),
            "extra_side_effects": sum(t["verification"]["extra_side_effects"] for t in rows),
            "unauthorized_modifications": sum(t["verification"]["unauthorized_modifications"] for t in rows),
        }
    return result
