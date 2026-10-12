import copy
import json
from dataclasses import asdict
from pathlib import Path
from .environment import Environment
from .faults import FaultInjector
from .models import ToolCall
from .policy import PolicyInput
from .verifier import Verifier

# 运行一个在线恢复 episode：先执行前置动作与失败调用，再在固定预算内询问策略。
# Oracle 仅能在认证模式运行；给普通策略的 PolicyInput 不包含任务验收答案。
def run_episode(task, policy, mode="evaluation"):
    if mode not in ("evaluation", "oracle_certification"):
        raise ValueError("Invalid run mode")
    from .oracle import Oracle
    if isinstance(policy, Oracle) and mode != "oracle_certification":
        raise ValueError("Oracle is forbidden during policy evaluation")
    fault = task["fault"]
    injector = FaultInjector(fault["type"], fault["seed"], fault["step"])
    with Environment(task["initial_state"], injector, task["goal"]) as env:
        policy.reset()
        observations, events = [], []
        for index, action in enumerate(task.get("prefix_actions", [])):
            prefix = ToolCall(**action, call_id=f"prefix-{index + 1}")
            prefix_result = env.execute(prefix)
            if not prefix_result.ok:
                raise ValueError("Prefix action must succeed")
            observations.append(prefix_result)
            events.append({"phase": "prefix", "call": asdict(prefix),
                           "result": prefix_result.to_dict()})
        first = ToolCall(**task["failed_action"], call_id="initial")
        result = env.execute(first)
        if result.ok:
            raise ValueError("Task must start with a failed action")
        observations.append(result)
        events.append({"phase": "failure", "call": asdict(first), "result": result.to_dict()})
        steps, repeated = 0, 0
        failures = {(first.name, json.dumps(first.arguments, sort_keys=True), result.error.code)}
        for index in range(task["max_recovery_steps"]):
            # 只复制公开字段给策略，避免可变对象引用和特权验收信息泄漏。
            public = PolicyInput(copy.deepcopy(task["goal"]), copy.deepcopy(first),
                                 tuple(copy.deepcopy(observations)),
                                 task["max_recovery_steps"] - index)
            call = policy.act(public)
            if call is None:
                break
            call = copy.deepcopy(call)
            call.call_id = f"recovery-{index + 1}"
            result = env.execute(call)
            steps += 1
            if not result.ok:
                # 重复失败按工具名、规范化参数和错误码比较，不依赖诊断 token。
                signature = (call.name, json.dumps(call.arguments, sort_keys=True), result.error.code)
                repeated += int(signature in failures)
                failures.add(signature)
            observations.append(result)
            events.append({"phase": "recovery", "call": asdict(call), "result": result.to_dict()})
        verdict = Verifier().check(task, env)
        return {"schema_version": "0.1", "goal": copy.deepcopy(task["goal"]),
                "task_id": task["id"], "template_family": task["template_family"],
                "split": task["split"], "fault_type": task["fault_type"],
                "mode": mode, "seed": fault["seed"], "recovery_steps": steps,
                "repeated_failures": repeated, "verification": asdict(verdict),
                "events": events, "final_state": env.snapshot(), "audit": copy.deepcopy(env.audit)}

# 在全新环境按记录的调用顺序执行，逐项比较工具结果、最终状态、账本与 Verifier。
# 重放不再次调用 Policy，验证的是已记录调用序列的确定性。
def replay(task, trace):
    fault = task["fault"]
    with Environment(task["initial_state"],
                     FaultInjector(fault["type"], fault["seed"], fault["step"]),
                     task["goal"]) as env:
        for event in trace["events"]:
            result = env.execute(ToolCall(**event["call"]))
            if result.to_dict() != event["result"]:
                raise ValueError("Replay observation mismatch")
        if env.snapshot() != trace["final_state"] or env.audit != trace["audit"]:
            raise ValueError("Replay state/audit mismatch")
        if asdict(Verifier().check(task, env)) != trace["verification"]:
            raise ValueError("Replay verdict mismatch")
        return True

# 汇总恢复成功率、调用步数、重复失败率和副作用；起始失败调用不计入恢复步数。
# 成功任务为空时平均成功步数为 None，避免用 0 伪装未定义的指标。
def summarize(traces):
    n = len(traces)
    if not n:
        raise ValueError("No episodes to evaluate")
    successes = sum(t["verification"]["success"] for t in traces)
    steps = sum(t["recovery_steps"] for t in traces)
    return {"task_count": n, "recovery_success_rate": successes / n,
            "average_recovery_steps": steps / n,
            "average_steps_successful": sum(t["recovery_steps"] for t in traces
                if t["verification"]["success"]) / successes if successes else None,
            "repeated_failure_rate": sum(t["repeated_failures"] for t in traces) / steps if steps else 0,
            "extra_side_effects": sum(t["verification"]["extra_side_effects"] for t in traces),
            "unauthorized_modifications": sum(t["verification"]["unauthorized_modifications"] for t in traces)}

# 为每个任务创建策略并运行 episode，再输出 JSONL 轨迹和总体/分故障报告。
def evaluate(tasks, factory, output, mode="evaluation"):
    traces = [run_episode(task, factory(task), mode) for task in tasks]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = summarize(traces)
    report.update(mode=mode, schema_version="0.1",
                  by_fault={f: summarize([t for t in traces if t["fault_type"] == f])
                            for f in sorted({t["fault_type"] for t in traces})})
    (output / "trajectories.jsonl").write_text("".join(
        json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n" for t in traces), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
