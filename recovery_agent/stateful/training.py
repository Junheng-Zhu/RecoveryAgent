import copy
from dataclasses import asdict
from ..models import ToolCall
from ..data_pipeline.common import canonical, write_jsonl, write_json
from ..data_pipeline.serialization import action_message, stop_message
from .environment import StatefulEnvironment
from .tasks import StatefulOracle, verify, state_id
from .collector import replay

def branch(task, prefix, candidate):
    # Rebuild the exact decision state, execute the candidate, then a bounded reference continuation.
    with StatefulEnvironment(task) as env:
        events = copy.deepcopy(prefix)
        for e in prefix:
            if env.execute(ToolCall(**e["call"])).to_dict() != e["result"]:
                raise ValueError("Training branch prefix mismatch")
        initial = state_id(env)
        oracle = StatefulOracle(task)
        oracle.next_action(events)  # Both branches replace the same first reference action.
        executed = []
        if candidate:
            call = ToolCall(**candidate)
            result = env.execute(call).to_dict()
            executed.append({"call": asdict(call), "result": result})
            events.append(executed[-1])
        immediate = verify(task, env)
        if candidate:
            for _ in range(task["max_recovery_steps"]-1):
                action = oracle.next_action(events)
                if action is None:
                    break
                call = ToolCall(**action, call_id=f"branch-{len(executed)}")
                result = env.execute(call).to_dict()
                executed.append({"call": asdict(call), "result": result})
                events.append(executed[-1])
        return {"start_state_id": initial, "candidate": candidate, "executed": executed,
                "candidate_verification": immediate, "verification": verify(task, env),
                "final_state": env.snapshot(), "audit": copy.deepcopy(env.audit)}

def candidate_evidence(task, trace):
    # Prototype only: first real decision vs reference action. Do not fabricate rankings for ambiguity.
    if task["split"] != "train" or not trace.get("decisions"):
        return None, "not_train_or_no_decision"
    decision = trace["decisions"][0]
    parsed = decision["parsed"]
    if parsed["kind"] not in ("tools", "stop") or len(parsed["calls"]) > 1:
        return None, "unreliable_or_multi_action"
    rejected = parsed["calls"][0]["call"] if parsed["calls"] else None
    if parsed["calls"] and not rejected:
        return None, "malformed_candidate"
    prefix = trace["events"][:decision["before_event_count"]]
    chosen = StatefulOracle(task).next_action(prefix)
    if chosen and rejected and chosen == {k: rejected[k] for k in ("name", "arguments")}:
        return None, "identical_action"
    if chosen:
        chosen = {**chosen, "call_id": "reference-candidate"}
    try:
        good, bad = branch(task, prefix, chosen), branch(task, prefix, rejected)
    except (KeyError, TypeError, ValueError):
        return None, "ambiguous_reference_continuation"
    if good["start_state_id"] != bad["start_state_id"]:
        raise ValueError("Candidate states differ")
    if not good["verification"]["success"] or bad["verification"]["success"]:
        return None, "no_verified_strict_preference"
    immediate = bad["candidate_verification"]
    if rejected is not None and not (immediate["unauthorized_modifications"] or immediate["extra_side_effects"]):
        return None, "ambiguous_recoverable_failure"
    return {"task_id": task["id"], "split": "train", "prompt": decision["prompt"],
            "chosen_source": "oracle_reference", "rejected_source": trace["source"],
            "chosen": action_message(chosen) if chosen else stop_message(),
            "rejected": action_message(rejected) if rejected else decision["raw_response"]["choices"][0]["message"],
            "evidence": {"chosen": good, "rejected": bad}}, None

def prepare(tasks, traces, output):
    # Replay before exporting. Separate Oracle/mock labels from actual LLM samples and exclude val/test.
    from pathlib import Path
    from collections import Counter
    by_id, samples, evidence, reasons = {t["id"]: t for t in tasks}, [], [], Counter()
    for trace in traces:
        task = by_id[trace["task_id"]]
        if task["split"] != "train":
            reasons["not_train"] += 1
            continue
        if not trace.get("events"):
            reasons["missing_trace"] += 1
            continue
        replay(task, trace)
        pair, reason = candidate_evidence(task, trace)
        if pair:
            evidence.append(pair)
        else:
            reasons[reason] += 1
        if not trace["verification"]["success"] or trace["verification"]["extra_side_effects"] or trace["verification"]["unauthorized_modifications"]:
            reasons["untrusted_or_failed_sft"] += 1
            continue
        for d in trace["decisions"]:
            if d.get("error") or d["parsed"]["kind"] == "invalid":
                continue
            parsed = d["parsed"]
            if any(c["error"] for c in parsed["calls"]):
                continue
            if parsed["kind"] == "tools":
                actual = trace["events"][d["before_event_count"]:d["before_event_count"]+len(parsed["calls"])]
                if len(actual) != len(parsed["calls"]) or any(e["call"] != c["call"] for e, c in zip(actual, parsed["calls"])):
                    continue
                if any(not e["result"]["ok"] or e.get("classification") in ("unsafe","invalid","unnecessary") for e in actual):
                    reasons["untrusted_sft_decision"] += 1
                    continue
                target = {"role": "assistant", "content": None,
                          "tool_calls": [action_message(c["call"])["tool_calls"][0] for c in parsed["calls"]]}
            else:
                target = {"role": "assistant", "content": parsed["text"]}
            samples.append({"task_id": task["id"], "source": trace["source"], "split": "train",
                            "messages": copy.deepcopy(d["prompt"]["messages"])+[target],
                            "tools": d["prompt"]["tools"], "loss_scope": "final_assistant_only"})
    output = Path(output)
    for source in ("llm", "oracle_mock"):
        write_jsonl(output/f"{source}_sft_train.jsonl", [s for s in samples if s["source"] == source])
    write_jsonl(output/"candidate_evidence.jsonl", evidence)
    report = {"sft_by_source": dict(Counter(s["source"] for s in samples)), "verified_candidates": len(evidence),
              "discard_reasons": dict(reasons), "test_training_samples": 0, "training_executed": False}
    write_json(output/"report.json", report)
    return report
