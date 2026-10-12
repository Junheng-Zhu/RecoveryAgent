import copy
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from ..models import ToolCall
from ..data_pipeline.common import canonical, fingerprint, write_json
from .environment import StatefulEnvironment
from .tasks import VERSION, verify, state_id
from .protocol import prompt_for, parse_response
from .predictor import MockPredictor, BudgetExceeded, redact

READ_ONLY = {"list_files", "read_file", "query_table", "calculate", "git_status", "git_diff", "execute_select"}

def replay(task, trace, environment_factory=None, verifier=None):
    # Replay recorded calls, not model requests. Include failure calls and side-effect accounting.
    with (environment_factory or StatefulEnvironment)(task) as env:
        for event in trace["events"]:
            if state_id(env) != event["before_state_id"]:
                raise ValueError("Replay before-state mismatch")
            result = env.execute(ToolCall(**event["call"])).to_dict()
            if result != event["result"] or state_id(env) != event["after_state_id"]:
                raise ValueError("Replay result/state mismatch")
        if env.snapshot() != trace["final_state"] or env.audit != trace["audit"]:
            raise ValueError("Replay final state/audit mismatch")
        if (verifier or verify)(task, env) != trace["verification"]:
            raise ValueError("Replay verifier mismatch")
        return True

def run_episode(task, predictor, clean=False, max_tool_calls=8, max_requests=8,
                environment_factory=None, verifier=None, prompt_builder=None, replay_function=None):
    # Extend the legacy episode protocol with model decisions; execution stays sequential and bounded.
    # Optional extension hooks keep the original defaults and all v0.3-A call sites unchanged.
    _Environment=environment_factory or StatefulEnvironment
    _verify=verifier or verify
    _prompt=prompt_builder or prompt_for
    _replay=replay_function or replay
    start = time.perf_counter()
    events, decisions, feedback = [], [], None
    failures, seen = set(), set()
    calls, invalid, repeated, unnecessary, unsafe, attempts = 0, 0, 0, 0, 0, 0
    reason = "request_budget"
    with _Environment(task) as env:
        def execute(call, phase):
            nonlocal repeated, unnecessary, unsafe, invalid
            before = state_id(env)
            semantic_state = fingerprint({"state": env.snapshot(), "audit": env.audit})
            signature = (call.name, canonical(call.arguments))
            old_audit = len(env.audit)
            old_verdict = _verify(task, env)
            result = env.execute(call).to_dict()
            event = {"phase": phase, "call": asdict(call), "result": result,
                     "before_state_id": before, "after_state_id": state_id(env)}
            if phase == "recovery":
                duplicated = (signature, semantic_state) in seen
                seen.add((signature, semantic_state))
                code = result["error"]["code"] if result["error"] else None
                rep = not result["ok"] and (*signature, code) in failures
                repeated += int(rep)
                unnecessary += int(duplicated or rep)
                bad = code in ("INVALID_ARGUMENT", "UNKNOWN_TOOL")
                invalid += int(bad)
                new_verdict = _verify(task, env)
                ledger_violation = (new_verdict["unauthorized_modifications"] > old_verdict["unauthorized_modifications"] or
                                    new_verdict["extra_side_effects"] > old_verdict["extra_side_effects"])
                unsafe += int(code == "UNSAFE_ACTION" or ledger_violation)
                event["classification"] = "unsafe" if code == "UNSAFE_ACTION" or ledger_violation else (
                    "invalid" if bad else "unnecessary" if duplicated or rep else
                    "state_conflict" if not result["ok"] and code in ("REVISION_CONFLICT","FILE_CONFLICT","CONSTRAINT_CONFLICT") else
                    "wrong_operation" if not result["ok"] and code == "TRANSACTION_FAILED" else
                    "dependency_error" if not result["ok"] and code in ("STATE_DEPENDENCY","NO_CHANGES","NOT_FOUND") else
                    "execution_failure" if not result["ok"] else
                    "diagnostic" if call.name in READ_ONLY else "state_change")
            if not result["ok"]:
                failures.add((*signature, result["error"]["code"]))
            events.append(event)
            return result
        if not clean:
            initial = execute(ToolCall(**task["failed_action"], call_id="initial"), "failure")
            if initial["ok"]:
                raise ValueError("Recovery task did not start with a failed call")
        for index in range(max_requests):
            remaining = min(max_tool_calls, task["max_recovery_steps"]) - calls
            if remaining <= 0:
                reason = "tool_budget"
                break
            prompt = _prompt(task, events, remaining, feedback)
            if hasattr(predictor, "events"):
                predictor.events = copy.deepcopy(events)
            try:
                received = predictor.predict(prompt)
            except BudgetExceeded:
                reason = "global_api_budget"
                break
            decision = {"index": index, "prompt": prompt, "before_event_count": len(events), **received}
            if received["error"]:
                decision["parsed"] = {"kind": "api_error", "calls": []}
                decisions.append(decision)
                feedback = received["error"]
                if sum(bool(d.get("error")) for d in decisions) >= 2:
                    reason = "api_errors"
                    break
                continue
            parsed = parse_response(received["raw_response"])
            decision["parsed"] = parsed
            decisions.append(decision)
            feedback = None
            if parsed["kind"] == "stop":
                reason = "assistant_text"
                break
            if parsed["kind"] == "invalid":
                invalid += 1
                attempts += 1
                feedback = parsed["error"]
                continue
            requested = parsed["calls"]
            attempts += len(requested)
            if len(requested) > remaining:
                invalid += len(requested)
                feedback = "tool_batch_exceeds_remaining_budget"
                continue  # Reject whole batch; never partially execute an oversized model batch.
            all_ids = {e["call"]["call_id"] for e in events}
            if any(c["call"] and c["call"]["call_id"] in all_ids for c in requested):
                invalid += len(requested)
                feedback = "tool_call_id_reused"
                continue
            for candidate in requested:
                if candidate["error"]:
                    invalid += 1
                    feedback = candidate["error"]
                    continue
                calls += 1
                execute(ToolCall(**candidate["call"]), "recovery")
        verdict = _verify(task, env)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for d in decisions:
            for k in usage:
                value = d.get("usage", {}).get(k, 0)
                usage[k] += value if type(value) is int else 0
        trace = {"dataset_version": VERSION, "task_id": task["id"], "split": task["split"],
                 "template_family": task["template_family"], "domain": task["domain"],
                 "source": predictor.source, "mode": "clean" if clean else "recovery",
                 "model_id": next((d["model_id"] for d in decisions), predictor.config.model),
                 "config": predictor.config.public(), "events": events, "decisions": decisions,
                 "verification": verdict, "final_state": env.snapshot(), "audit": copy.deepcopy(env.audit),
                 "usage": usage, "termination_reason": reason,
                 "elapsed_ms": (time.perf_counter()-start)*1000,
                 "recovery_steps": calls, "model_tool_attempts": attempts, "invalid_tool_calls": invalid, "repeated_failures": repeated,
                 "unnecessary_tool_calls": unnecessary, "unsafe_actions": unsafe,
                 "extra_tool_calls": max(0, calls-len(task["reference_plan"])),
                 "api_errors": sum(bool(d.get("error")) for d in decisions),
                 "collection_status": "budget_not_evaluated" if not decisions and reason == "global_api_budget" else "completed"}
    trace["replay_verified"] = _replay(task, trace)
    return redact(trace, getattr(predictor, "secret", ""))

def sample_tasks(tasks, count, seed=42, split="train"):
    # Deterministic domain-balanced sample; online training collection never samples val/test.
    rows = [t for t in tasks if t["split"] == split]
    buckets = {}
    rng = random.Random(seed)
    for domain in sorted({t["domain"] for t in rows}):
        bucket = [t for t in rows if t["domain"] == domain]
        rng.shuffle(bucket)
        buckets[domain] = bucket
    chosen = []
    while buckets and len(chosen) < count:
        for domain in list(buckets):
            chosen.append(buckets[domain].pop())
            if not buckets[domain]:
                del buckets[domain]
            if len(chosen) == count:
                break
    return chosen

def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high]-values[low])*(index-low)

def summarize(traces):
    # Rates use recovery calls; malformed decisions count as invalid attempts, not executed writes.
    recoveries = [t for t in traces if t["mode"] == "recovery"]
    clean = [t for t in traces if t["mode"] == "clean"]
    calls = sum(t.get("recovery_steps", 0) for t in recoveries)
    bad = sum(t.get("invalid_tool_calls", 0) for t in recoveries)
    attempts = sum(t.get("model_tool_attempts", 0) for t in recoveries)
    successes = sum(t["verification"]["success"] for t in recoveries)
    latency = [d["latency_ms"] for t in recoveries for d in t.get("decisions", [])]
    attempted = [t for t in recoveries if t.get("decisions")]
    skipped = [t for t in recoveries if not t.get("decisions") and t["termination_reason"] == "global_api_budget"]
    return {
        "decision_attempted_tasks": len(attempted),
        "attempted_recovery_success_rate": sum(t["verification"]["success"] for t in attempted)/len(attempted) if attempted else None,
        "budget_not_evaluated_tasks": len(skipped),
        "model_request_count": sum(len(t.get("decisions", [])) for t in traces),
        "extra_tool_calls": sum(t.get("extra_tool_calls", 0) for t in recoveries),
        "tool_classifications": dict(Counter(e.get("classification") for t in recoveries
                                            for e in t.get("events",[]) if e["phase"]=="recovery")),
        "task_count": len(traces), "recovery_task_count": len(recoveries),
        "recovery_success_rate": successes/len(recoveries) if recoveries else None,
        "clean_task_count": len(clean),
        "clean_task_success_rate": sum(t["verification"]["success"] for t in clean)/len(clean) if clean else None,
        "invalid_tool_call_rate": bad/attempts if attempts else 0,
        "repeated_failure_rate": sum(t.get("repeated_failures", 0) for t in recoveries)/calls if calls else 0,
        "unnecessary_tool_call_rate": sum(t.get("unnecessary_tool_calls", 0) for t in recoveries)/calls if calls else 0,
        "unsafe_action_rate": sum(t.get("unsafe_actions", 0) for t in recoveries)/calls if calls else 0,
        "average_recovery_steps": calls/len(recoveries) if recoveries else None,
        "p50_latency_ms": percentile(latency, 0.5), "p95_latency_ms": percentile(latency, 0.95),
        "tokens_per_successful_recovery": sum(t.get("usage", {}).get("total_tokens", 0) for t in recoveries)/successes if successes else None,
        "total_usage": {k: sum(t.get("usage", {}).get(k, 0) for t in traces)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "replays_verified": sum(t.get("replay_verified", False) for t in traces),
        "termination_reasons": dict(Counter(t["termination_reason"] for t in traces)),
        "api_errors": sum(t.get("api_errors", 0) for t in traces)}

def _collect(tasks, factory, output, session, resume=False, clean=False, runner=None, dataset_version=None):
    # Append+fsync each completed trajectory; resume checks dataset/config before skipping IDs.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = {"dataset_version": dataset_version or VERSION, "task_digest": fingerprint(tasks), "session": session, "clean": clean}
    manifest, path = output/"session.json", output/"trajectories.jsonl"
    if manifest.exists():
        if not resume:
            raise FileExistsError("Collection exists; choose a new directory or --resume")
        if json.loads(manifest.read_text(encoding="utf-8")) != meta:
            raise ValueError("Resume configuration/task sample differs")
    else:
        write_json(manifest, meta)
    traces = []
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            # Only discard an incomplete trailing record; malformed complete records still fail.
            offset = data.rfind(b"\n")+1
            path.write_bytes(data[:offset])
            data = data[:offset]
        traces = [json.loads(line) for line in data.splitlines()]
    done = {t["task_id"] for t in traces}
    if len(done) != len(traces):
        raise ValueError("Duplicate completed task IDs")
    for task in tasks:
        if task["id"] in done:
            continue
        try:
            trace = (runner or run_episode)(task, factory(task), clean=clean,
                                max_tool_calls=session.get("max_tool_calls", 8),
                                max_requests=session.get("episode_requests", 8))
        except Exception as error:
            trace = {"task_id": task["id"], "split": task["split"], "mode": "clean" if clean else "recovery",
                     "verification": {"success": False}, "termination_reason": "collector_exception",
                     "exception_type": type(error).__name__, "replay_verified": False}
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(trace)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        traces.append(trace)
        print(f"saved {task['id']} success={trace['verification']['success']} reason={trace['termination_reason']}", flush=True)
    report = summarize(traces)
    write_json(output/"report.json", report)
    return traces, report

def collect(tasks, factory, output, session, resume=False, clean=False, runner=None, dataset_version=None):
    # One collector owns each output/budget directory; stale crash locks require explicit inspection.
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    lock=output/".collection.lock"
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(descriptor,str(os.getpid()).encode())
        os.close(descriptor)
        return _collect(tasks,factory,output,session,resume,clean,runner,dataset_version)
    finally:
        lock.unlink(missing_ok=True)
