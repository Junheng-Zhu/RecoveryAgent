import copy
from collections import Counter
from dataclasses import asdict
from ..environment import Environment
from ..faults import FaultInjector
from ..models import ToolCall
from ..verifier import Verifier
from .common import VERSION, fingerprint
from .serialization import action_message, stop_message, parse_action

# 对状态、账本、调用步数和故障/前置条件做联合摘要，核对分支真正从同一环境状态开始。
def state_fingerprint(env):
    return fingerprint({"state": env.snapshot(), "audit": env.audit, "step": env.step,
                        "fault_fired": env.injector.fired,
                        "needs_precondition": env.needs_precondition})

# 在新环境重放共同历史，再执行候选及预算内的相同参考续程，记录实际执行证据。
# 候选后的即时 Verifier 与最终 Verifier 分开保存，防止把续程造成的违规误归因于候选。
def branch_rollout(task, prefix, candidate, suffix):
    fault = task["fault"]
    with Environment(task["initial_state"],
                     FaultInjector(fault["type"], fault["seed"], fault["step"]),
                     task["goal"]) as env:
        for event in prefix:
            result = env.execute(ToolCall(**event["call"]))
            if result.to_dict() != event["result"]:
                raise ValueError("Branch prefix replay mismatch")
        start_id = state_fingerprint(env)
        audit_start = len(env.audit)
        call = parse_action(candidate, strict=False)
        events, candidate_audit = [], []
        # 初始判定用于停止候选；执行工具候选后会立即更新，不混入后续动作的影响。
        candidate_verification = asdict(Verifier().check(task, env))
        if call is not None:
            result = env.execute(call)
            events.append({"call": asdict(call), "result": result.to_dict()})
            candidate_audit = copy.deepcopy(env.audit[audit_start:])
            candidate_verification = asdict(Verifier().check(task, env))
            # 候选本身占用一步恢复预算，续程不得超出原 episode 的剩余调用上限。
            remaining = task["max_recovery_steps"] - sum(e["phase"] == "recovery" for e in prefix) - 1
            for event in suffix[:max(0, remaining)]:
                result = env.execute(ToolCall(**event["call"]))
                events.append({"call": copy.deepcopy(event["call"]), "result": result.to_dict()})
        verdict = asdict(Verifier().check(task, env))
        return {"start_state_id": start_id, "candidate": copy.deepcopy(candidate),
                "continuation": copy.deepcopy(suffix), "executed_events": events,
                "candidate_result": events[0]["result"] if events else None,
                "candidate_audit": candidate_audit, "candidate_verification": candidate_verification,
                "verification": verdict, "final_state": env.snapshot(), "audit": copy.deepcopy(env.audit),
                "stopped": call is None, "total_recovery_steps": sum(e["phase"] == "recovery" for e in prefix) + len(events)}

# 从同一决策构造停止、原调用重试、未请求写入与只读替代候选，不读取隐藏答案。
# 候选名称仅用于统计，是否形成偏好必须由实际执行和 Verifier 决定。
def candidates(task, decision):
    call_id = f"recovery-{decision['decision_index'] + 1}"
    failed = {**copy.deepcopy(task["failed_action"]), "call_id": call_id}
    # No answer/state lookup: mutations are generic actions valid for these tools.
    unsafe = {"name": "write_file", "arguments": {
        "path": task["environment_config"]["file_prefix"] + "unauthorized.txt",
        "content": "unrequested write"}, "call_id": call_id}
    noop = {"name": "list_files", "arguments": {}, "call_id": call_id}
    return [("premature_stop", stop_message()), ("retry_failed", action_message(failed)),
            ("unrequested_write", action_message(unsafe)), ("read_only_alternative", action_message(noop))]

# 仅接受 chosen 完成任务、rejected 停止未完成或候选自身造成不可消除账本违规的偏好。
# 双方成功或失败仍可修复时不强行排序；固定续程失败不足以证明候选动作更差。
def preference_reason(chosen, rejected):
    if chosen["start_state_id"] != rejected["start_state_id"]:
        raise ValueError("Preference branches did not start at the same state")
    if not chosen["verification"]["success"]:
        return None, "chosen_rollout_failed"
    if rejected["verification"]["success"]:
        return None, "both_succeed_no_strict_preference"
    if rejected["stopped"]:
        return "chosen_finishes_rejected_stops_unsatisfied", None
    # 只把候选已经造成的账本违规作为可靠负例，不把固定续程引入的违规算到候选头上。
    v = rejected["candidate_verification"]
    if v["unauthorized_modifications"] or v["extra_side_effects"]:
        return "rejected_has_irreversible_audit_violation", None
    # A failed fixed continuation does not prove an alternative action is bad:
    # a different repair could still work. Conservatively discard it.
    return None, "ambiguous_recoverable_failure"

# 对 train/val 的已验证轨迹构建同 Prompt 的 chosen/rejected 分支，并统计丢弃原因。
# 默认只选第一个恢复决策；test 任务不会进入偏好候选构建。
def build_preferences(task, trajectory, max_decisions=1):
    if task["split"] == "test":
        return [], {"test_excluded": 1}
    if not trajectory["verification"]["success"] or not trajectory["replay_verified"]:
        return [], {"unsuccessful_or_unverified": 1}
    pairs, dropped = [], Counter()
    for decision in trajectory["decisions"][:max_decisions]:
        index = decision["event_index"]
        prefix = trajectory["trace"]["events"][:index]
        suffix = trajectory["trace"]["events"][index + 1:]
        chosen = branch_rollout(task, prefix, decision["oracle_action"], suffix)
        for kind, candidate in candidates(task, decision):
            if candidate == decision["oracle_action"]:
                dropped["identical_actions"] += 1
                continue
            rejected = branch_rollout(task, prefix, candidate, suffix)
            reason, discard = preference_reason(chosen, rejected)
            if discard:
                dropped[discard] += 1
                continue
            pairs.append({
                "schema_version": VERSION,
                "sample_id": f"{task['id']}:{decision['decision_index']}:{kind}",
                "task_id": task["id"], "split": task["split"],
                "decision_index": decision["decision_index"],
                "prompt_id": decision["prompt_id"], "prompt": copy.deepcopy(decision["prompt"]),
                "chosen": [copy.deepcopy(decision["oracle_action"])], "rejected": [candidate],
                "candidate_kind": kind, "preference_reason": reason,
                "evidence": {"prefix_event_count": index, "chosen": chosen, "rejected": rejected},
            })
    return pairs, dict(dropped)

# 从原任务重新执行两个候选分支，逐项核对 Prompt、标签、执行证据与偏好依据。
# 不能只信任样本中已经保存的 success 字段或偏好标签。
def verify_pair(task, trajectory, pair):
    decision = trajectory["decisions"][pair["decision_index"]]
    if pair["prompt"] != decision["prompt"] or pair["prompt_id"] != decision["prompt_id"]:
        raise ValueError("Preference prompt mismatch")
    if len(pair["chosen"]) != 1 or len(pair["rejected"]) != 1:
        raise ValueError("Expected single-action completions")
    if pair["chosen"][0] != decision["oracle_action"]:
        raise ValueError("Chosen is not the collected supervision target")
    index = decision["event_index"]
    prefix = trajectory["trace"]["events"][:index]
    suffix = trajectory["trace"]["events"][index + 1:]
    chosen = branch_rollout(task, prefix, pair["chosen"][0], suffix)
    rejected = branch_rollout(task, prefix, pair["rejected"][0], suffix)
    if pair["evidence"] != {"prefix_event_count": index, "chosen": chosen, "rejected": rejected}:
        raise ValueError("Preference execution evidence mismatch")
    reason, discard = preference_reason(chosen, rejected)
    if discard or reason != pair["preference_reason"]:
        raise ValueError("Unverified preference")
    return True
