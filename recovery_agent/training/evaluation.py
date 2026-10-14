"""Base/SFT benchmark orchestration over the unchanged collector, verifier and replay."""
import copy
from collections import Counter
from pathlib import Path
from ..data_pipeline.common import fingerprint, write_json, write_jsonl
from ..hardening.tasks import validate_frozen
from ..hardening.protocol import schema_hash, STOP_PROTOCOLS
from ..hardening.runner import run_episode, OracleMock
from ..hardening.mutation_diagnostics import diagnose_mutations
from ..stateful.collector import collect, sample_tasks, percentile
from ..gpu_workspace import versions
from .local_model import LocalEngine, LocalPredictor

def select_tasks(tasks, split, seed=42):
    """Train smoke is exactly three domain-balanced tasks; evaluation splits remain separate."""
    if split == "train_smoke":
        selected = sample_tasks(tasks, 3, seed, "train")
        if len(selected) != 3:
            raise ValueError("Three train smoke tasks are required")
    elif split in ("iid_validation", "iid_test", "ood"):
        selected = [t for t in tasks if t["eval_group"] == split]
    else:
        raise ValueError("Choose train_smoke / iid_validation / iid_test / ood")
    if not selected or any(t["eval_group"] != ("train" if split == "train_smoke" else split) for t in selected):
        raise ValueError("Evaluation split isolation failure")
    return selected

def comparison_settings(tasks, config, tools, requests, protocol, assisted, seed):
    """Base and Adapter share this signature; model/adapter weights are recorded separately."""
    return {"task_digest": fingerprint(tasks), "schema_hash": schema_hash(),
            "stop_protocol": protocol, "runtime_assisted": bool(assisted), "seed": seed,
            "max_tool_calls": tools, "max_requests": requests,
            "decoding": {k: config.public()[k] for k in
                         ("temperature", "max_tokens", "max_total_tokens", "dtype")}}

def benchmark_summary(tasks, traces):
    """New benchmark fields use explicit episode/attempt/request denominators and real replay evidence."""
    task_map = {t["id"]: t for t in tasks}
    diagnostics, missing = [], []
    for trace in traces:
        if trace.get("events") and trace.get("replay_verified"):
            row = diagnose_mutations(task_map[trace["task_id"]], trace)
            diagnostics.append(row)
        else:
            missing.append(trace["task_id"])
    def summarize(rows, cohort):
        n = len(cohort)
        reached = [r for r in rows if r["ever_full_success"]]
        autonomous = [r for r in rows if r["stop_regime"] == "model_autonomous"]
        decisions = [d for t in cohort for d in t.get("decisions", [])]
        attempts = sum(t.get("model_tool_attempts", 0) for t in cohort)
        def rate(num, den):
            return {"numerator": num, "denominator": den, "value": num/den if den else None}
        return {
            "episodes": n, "verified_episodes": len(rows), "missing_replay_evidence": n-len(rows),
            "final_recovery_success_rate": rate(sum(r["final_full_success"] for r in rows), n),
            "ever_verified_success_rate": rate(len(reached), len(rows)),
            "autonomous_completion_rate": rate(sum(r["autonomous_completion"] for r in autonomous), len(autonomous)),
            "unauthorized_mutation_rate": rate(sum("unauthorized_mutation" in r["diagnostic_labels"] for r in rows), len(rows)),
            "mutation_budget_exceeded_rate": rate(sum("mutation_budget_exceeded" in r["diagnostic_labels"] for r in rows), len(rows)),
            "invalid_tool_call_rate": rate(sum(t.get("invalid_tool_calls", 0) for t in cohort), attempts),
            "post_success_tool_calls": {
                "total": sum(r["post_success_tool_calls"] for r in rows),
                "ever_success_denominator": len(reached),
                "mean_among_ever_successful": sum(r["post_success_tool_calls"] for r in rows)/len(reached) if reached else None},
            "average_tool_calls": sum(t.get("recovery_steps", 0) for t in cohort)/n if n else None,
            "average_tokens": sum(t.get("usage", {}).get("total_tokens", 0) for t in cohort)/n if n else None,
            "average_request_latency_ms": sum(d["latency_ms"] for d in decisions)/len(decisions) if decisions else None,
            "p50_request_latency_ms": percentile([d["latency_ms"] for d in decisions], .5),
            "p95_request_latency_ms": percentile([d["latency_ms"] for d in decisions], .95),
            "average_episode_latency_ms": sum(t.get("elapsed_ms", 0) for t in cohort)/n if n else None,
            "termination_reasons": dict(Counter(t["termination_reason"] for t in cohort))}
    report = {"metrics": summarize(diagnostics, traces), "missing_evidence_tasks": missing,
              "by_mechanism": {}}
    for mechanism in sorted({t["mechanism_id"] for t in tasks}):
        ids = {t["id"] for t in tasks if t["mechanism_id"] == mechanism}
        report["by_mechanism"][mechanism] = summarize(
            [r for r in diagnostics if r["task_id"] in ids],
            [t for t in traces if t["task_id"] in ids])
    report["definitions"] = {
        "final": "Full unchanged Verifier: state plus cumulative side-effect ledger; failures/exceptions stay in episode denominator.",
        "ever": "At least one legal execution prefix passed full Verifier; denominator is replay-verified episodes, missing evidence listed separately.",
        "autonomous": "Successful assistant_text termination / model-autonomous replay-verified episodes; runtime-assisted episodes excluded.",
        "mutation_rates": "Episodes with actual audited unauthorized/quota-exceeding mutations / replay-verified episodes; permission-denied attempts alone are not writes.",
        "invalid": "Original collector invalid_tool_calls / original model_tool_attempts (including malformed/unexecuted attempts).",
        "post_success": "Recovery calls after first full verified success; total and mean among ever-successful episodes, not all automatically unnecessary.",
        "tokens": "Actual input plus completion token count per episode; mock zero usage is not measured model performance."}
    return report, diagnostics

def evaluate(workspace, config, split, output, execution="dry-run", seed=42,
             max_tool_calls=12, protocol="termination-v0.3c-2", assisted=False, resume=False, device=0):
    """Reuse the existing incremental collector; no new environment or execution semantics."""
    if protocol not in STOP_PROTOCOLS or max_tool_calls <= 0:
        raise ValueError("Invalid existing protocol/tool budget")
    tasks, manifest = validate_frozen(workspace.frozen, schema_hash())
    selected = select_tasks(tasks, split, seed)
    settings = comparison_settings(selected, config, max_tool_calls, config.max_requests, protocol, assisted, seed)
    if execution == "dry-run":
        return {"status": "EVALUATION_CONFIG_ONLY", "split": split, "tasks": len(selected),
                "model_config": config.public(), "comparison": settings, "paths": workspace.public(),
                "gpu_inference_verified": False}
    if execution not in ("mock", "gpu"):
        raise ValueError("Execution must be dry-run, mock or gpu")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Evaluation output exists; choose separate Base/Adapter directories")
    engine = LocalEngine(workspace, config, device) if execution == "gpu" else None
    def factory(task):
        if execution == "mock":
            return OracleMock(task)
        task_seed = seed ^ int(fingerprint(task["id"])[:8], 16)
        return LocalPredictor(engine, config, task_seed)
    session = {"execution": execution, "model_config": config.public(), "comparison": settings,
               "max_tool_calls": max_tool_calls, "episode_requests": config.max_requests,
               "dataset_manifest": manifest,
               "model_inventory": engine.preflight["model_inventory"] if engine else None,
               "adapter_sha256": engine.model_id if engine and config.adapter else None}
    runner = lambda task, predictor, clean, max_tool_calls, max_requests: run_episode(
        task, predictor, clean=clean, max_tool_calls=max_tool_calls, max_requests=max_requests,
        stop_protocol=protocol, stop_on_verified_success=assisted)
    traces, legacy = collect(selected, factory, output, session, resume=resume, runner=runner, dataset_version="0.3-B")
    write_jsonl(output/"tasks.jsonl", selected)
    report, diagnostics = benchmark_summary(selected, traces)
    report.update(execution=execution, model_performance_measured=execution == "gpu",
                  split=split, comparison_signature=fingerprint(settings), comparison=settings,
                  model_config=config.public(), dataset_manifest=manifest, versions=versions(),
                  legacy_collector_report=legacy,
                  gpu_preflight=engine.preflight if engine else None)
    write_json(output/"benchmark_report.json", report)
    write_jsonl(output/"termination_diagnostics.jsonl", diagnostics)
    return report
