import copy
import json
from dataclasses import asdict
from pathlib import Path
from .environment import Environment
from .faults import FaultInjector
from .models import ToolCall
from .policy import PolicyInput
from .verifier import Verifier

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
