"""Read-only provenance audit and trusted causal SFT preparation for v0.4-A."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from .data_pipeline.common import canonical, fingerprint, read_jsonl, write_json, write_jsonl
from .data_pipeline.serialization import action_message
from .hardening.analysis import annotate
from .hardening.mutation_diagnostics import diagnose_mutations
from .hardening.protocol import prompt_for, schema_hash
from .hardening.runner import replay
from .hardening.tasks import validate_frozen
from .stateful.protocol import parse_response
from .qwen3_tokenizer import validate_messages

UNTRUSTED = {"unsafe", "invalid", "redundant", "non_progress", "uncertain"}
BAD_MUTATIONS = {"redundant_state_mutation", "unauthorized_mutation",
                 "mutation_budget_exceeded", "uncertain"}

def protocol_version(trace):
    return trace.get("termination_protocol", {}).get("prompt_version", "legacy")

def validate_causal_prompts(task, trace):
    """Rebuild existing prompts, including parser feedback and the recorded stop protocol."""
    feedback = None
    for decision in trace.get("decisions", []):
        count = decision["before_event_count"]
        if not isinstance(count, int) or not 0 <= count <= len(trace["events"]):
            raise ValueError("Invalid event prefix")
        prefix = trace["events"][:count]
        remaining = min(trace["collection_limits"]["max_tool_calls"], task["max_recovery_steps"])
        remaining -= sum(e["phase"] == "recovery" for e in prefix)
        expected = prompt_for(task, prefix, remaining, feedback, protocol_version(trace))
        if decision["prompt"] != expected:
            raise ValueError("Unverified prompt, hidden fields or future information")
        parsed = decision["parsed"]
        if not decision.get("error") and parse_response(decision["raw_response"]) != parsed:
            raise ValueError("Raw model response differs from parsed decision")
        feedback = None
        if decision.get("error"):
            feedback = decision["error"]
        elif parsed["kind"] == "invalid":
            feedback = parsed["error"]
        elif parsed["kind"] == "tools":
            if len(parsed["calls"]) > remaining:
                feedback = "tool_batch_exceeds_remaining_budget"
            elif any(c["call"] and c["call"]["call_id"] in {e["call"]["call_id"] for e in prefix}
                     for c in parsed["calls"]):
                feedback = "tool_call_id_reused"
            else:
                for call in parsed["calls"]:
                    if call["error"]:
                        feedback = call["error"]
    return True

def target_for(decision):
    """Keep actual model grouping, content and IDs; never substitute a reference action."""
    message = decision["raw_response"]["choices"][0]["message"]
    target = {"role": "assistant", "content": message.get("content")}
    if decision["parsed"]["kind"] == "tools":
        target["tool_calls"] = copy.deepcopy(message["tool_calls"])
    return target

def trusted_decisions(task, trace, trajectory_id, directory, mutation_labels=None):
    """Conservatively reject failed episodes; verify each retained decision separately."""
    drops, samples = Counter(), []
    if task["split"] != "train" or task["eval_group"] != "train":
        return [], Counter({"non_train_episode": len(trace["decisions"])})
    if trace["mode"] != "recovery":
        return [], Counter({"clean_episode": len(trace["decisions"])})
    if not trace["verification"]["success"]:
        return [], Counter({"failed_episode": len(trace["decisions"])})
    if trace.get("termination_protocol", {}).get("stop_on_verified_success", False):
        return [], Counter({"runtime_assisted_episode": len(trace["decisions"])})
    mutation_labels = mutation_labels or {}
    for decision in trace["decisions"]:
        parsed = decision["parsed"]
        if decision.get("error") or parsed["kind"] not in ("tools", "stop"):
            drops["invalid_decision"] += 1
            continue
        count = decision["before_event_count"]
        actual = []
        if parsed["kind"] == "tools":
            calls = parsed["calls"]
            actual = trace["events"][count:count+len(calls)]
            if (any(c["error"] or not c["call"] for c in calls) or len(actual) != len(calls) or
                any(e["call"] != c["call"] or e["phase"] != "recovery"
                    for e, c in zip(actual, calls))):
                drops["unexecuted_or_malformed"] += 1
                continue
            rejected = []
            for i, event in enumerate(actual, count):
                label = event["behavior"]["label"]
                if not event["result"]["ok"]:
                    rejected.append("tool_failed")
                if label in UNTRUSTED:
                    rejected.append(label)
                rejected += sorted(set(mutation_labels.get(i, [])) & BAD_MUTATIONS)
            if rejected:
                drops["untrusted_decision"] += 1
                for label in set(rejected):
                    drops[f"untrusted:{label}"] += 1
                continue
            # Queries after complete verified success are not necessary recovery supervision.
            if actual[0]["behavior"]["verifier_before"]["success"]:
                drops["post_success_call"] += 1
                continue
        elif (trace["termination_reason"] != "assistant_text" or
              decision is not trace["decisions"][-1]):
            drops["unverified_stop"] += 1
            continue
        target = target_for(decision)
        prompt = decision["prompt"]
        messages = copy.deepcopy(prompt["messages"])+[target]
        try:
            bindings = validate_messages(messages, prompt["tools"])
        except (ValueError, KeyError, TypeError):
            drops["invalid_conversation"] += 1
            continue
        samples.append({
            "sample_id": f"{trajectory_id}:{decision['index']}",
            "source": "teacher" if trace["source"] == "llm" else "oracle_reference",
            "source_transport": trace["source"], "source_directory": directory,
            "model_id": decision.get("model_id", trace["model_id"]),
            "configured_model_id": trace["model_id"], "task_id": task["id"],
            "protocol_version": protocol_version(trace), "trajectory_id": trajectory_id,
            "decision_index": decision["index"], "split": "train", "eval_group": task["eval_group"],
            "dataset_version": task["dataset_version"], "preparation_version": "v0.4-A",
            "mechanism_id": task["mechanism_id"], "domain": task["domain"],
            "schema_hash": trace["schema_hash"], "prompt_hash": fingerprint(prompt),
            "messages": messages, "tools": copy.deepcopy(prompt["tools"]),
            "loss_scope": "final_assistant_only", "tool_bindings": bindings,
            "evidence": {"replay_verified": True, "final_verifier_success": True,
                         "target_call_ids": [e["call"]["call_id"] for e in actual],
                         "target_behavior_labels": [e["behavior"]["label"] for e in actual],
                         "target_result_hashes": [fingerprint(e["result"]) for e in actual]},
            "trajectory_recovery_steps": trace["recovery_steps"],
            "target_kind": "tool_call" if actual else "final_answer"})
    return samples, drops

def sample_key(sample):
    """Normalize call IDs for duplicate detection only; never alter exported conversations."""
    messages = copy.deepcopy(sample["messages"])
    ids = {}
    for message in messages:
        for call in message.get("tool_calls", []):
            ids.setdefault(call["id"], f"call-{len(ids)}")
            call["id"] = ids[call["id"]]
            fn = call["function"]
            if isinstance(fn["arguments"], str):
                fn["arguments"] = json.loads(fn["arguments"])
        if message["role"] == "tool":
            message["tool_call_id"] = ids[message["tool_call_id"]]
    return fingerprint({"messages": messages, "tools": sample["tools"]})

def source_directories(root):
    """Discover only the sources requested for this stage, including derived evidence reports."""
    return [root/"outputs"/name for name in (
        "v0.3b_sft_reference_final", "v0.3b_teacher10",
        "v0.3c_teacher10_new_stop", "v0.3c2_paired_teacher10")] + \
        sorted((root/"outputs").glob("v03c2_eval_*")) + \
        sorted((root/"outputs").glob("v03c2_retry_*"))

def audit_sources(root, progress=print):
    """Read frozen data and replay recorded actions; never construct or invoke a predictor."""
    root = Path(root)
    dataset = root/"data/v0.3b_seed42_final"
    frozen, manifest = validate_frozen(dataset, schema_hash())
    by_id = {t["id"]: t for t in frozen}
    groups = Counter(t["eval_group"] for t in frozen)
    train_mechanisms = {t["mechanism_id"] for t in frozen if t["eval_group"] == "train"}
    ood_mechanisms = {t["mechanism_id"] for t in frozen if t["eval_group"] == "ood"}
    if train_mechanisms & ood_mechanisms:
        raise ValueError("Train/OOD mechanism leakage")
    oracle, teacher, inventory, episodes = [], [], [], []
    drops = Counter()
    hashes, checked_paths = {}, set()
    paths = source_directories(root)
    reference = paths[0]
    oracle_evidence = root/"outputs/v0.3b_mock_verified"
    derived = read_jsonl(reference/"oracle_reference_sft_train.jsonl")
    derived_keys = {fingerprint({"messages": s["messages"], "tools": s["tools"]}) for s in derived}
    matched_derived = set()
    all_directories = [dataset]+paths+[oracle_evidence]
    before_hashes = {}
    for directory in all_directories:
        for path in directory.rglob("*") if directory.exists() else []:
            if path.is_file():
                relative = str(path.relative_to(root))
                before_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                checked_paths.add(path)
    for directory in paths+[oracle_evidence]:
        entry = {"directory": str(directory.relative_to(root)), "exists": directory.exists()}
        if not directory.exists():
            entry["status"] = "missing"
            inventory.append(entry)
            continue
        if not (directory/"trajectories.jsonl").exists():
            entry.update(status="derived_only_not_an_independent_source",
                         jsonl_rows={p.name: len(read_jsonl(p)) for p in directory.glob("*.jsonl")})
            inventory.append(entry)
            continue
        local_tasks = read_jsonl(directory/"tasks.jsonl")
        local = {t["id"]: t for t in local_tasks}
        if len(local) != len(local_tasks) or any(by_id.get(t["id"]) != t for t in local_tasks):
            raise ValueError(f"Task identity/split differs from frozen manifest: {directory.name}")
        traces = read_jsonl(directory/"trajectories.jsonl")
        entry.update(status="execution_evidence", episodes=len(traces), replay_verified=0,
                     successful=0, trustworthy_decisions=0, exclusion_reasons={})
        entry_drops = Counter()
        for position, original in enumerate(traces):
            trace = copy.deepcopy(original)
            task = local[trace["task_id"]]
            if (trace["split"] != task["split"] or trace["eval_group"] != task["eval_group"] or
                trace["schema_hash"] != schema_hash() or trace["dataset_version"] != task["dataset_version"]):
                raise ValueError("Trajectory split/version/schema provenance mismatch")
            if directory == oracle_evidence:
                if trace["source"] != "oracle_mock":
                    raise ValueError("Reference evidence is not Oracle-scripted")
            elif trace["source"] != "llm":
                raise ValueError("Requested Teacher source is not real LLM evidence")
            replay(task, trace)
            annotate(task, trace)
            validate_causal_prompts(task, trace)
            drift = sum(old.get("behavior") != new.get("behavior")
                        for old, new in zip(original["events"], trace["events"]))
            mutation = None
            labels = {}
            if trace["source"] == "llm":
                mutation = diagnose_mutations(task, trace)
                labels = {e["event_index"]: e["labels"] for e in mutation["mutation_evidence"]}
            trajectory_id = fingerprint({"directory": entry["directory"],
                                         "position": position, "trace": original})
            kept, excluded = trusted_decisions(task, trace, trajectory_id, entry["directory"], labels)
            if directory == oracle_evidence:
                retained = []
                for sample in kept:
                    key = fingerprint({"messages": sample["messages"], "tools": sample["tools"]})
                    if key not in derived_keys:
                        excluded["not_in_requested_reference_export"] += 1
                        continue
                    matched_derived.add(key)
                    retained.append(sample)
                kept = retained
            (oracle if directory == oracle_evidence else teacher).extend(kept)
            entry_drops.update(excluded)
            entry["replay_verified"] += 1
            entry["successful"] += int(trace["verification"]["success"])
            entry["trustworthy_decisions"] += len(kept)
            episodes.append({"trajectory_id": trajectory_id, "source_directory": entry["directory"],
                             "source": trace["source"], "model_id": trace["model_id"],
                             "protocol_version": protocol_version(trace), "task_id": task["id"],
                             "split": task["split"], "eval_group": task["eval_group"],
                             "final_verifier": trace["verification"], "replay_verified": True,
                             "recomputed_behavior_drift_events": drift,
                             "behavior_counts": trace["behavior_counts"],
                             "diagnostic_labels": mutation["diagnostic_labels"] if mutation else [],
                             "trusted_decisions": len(kept), "exclusion_reasons": dict(excluded),
                             "recovery_steps": trace["recovery_steps"],
                             "initial_error_type": trace["events"][0]["result"]["error"]["code"]
                             if trace["events"] and trace["events"][0]["result"]["error"] else "none"})
            progress(f"Audited {directory.name} [{position+1}/{len(traces)}]: {task['id']} -> {len(kept)} trusted")
        entry["exclusion_reasons"] = dict(entry_drops)
        drops.update(entry_drops)
        inventory.append(entry)
    seen, duplicates = {}, Counter()
    for name, rows in (("oracle_reference", oracle), ("teacher", teacher)):
        retained = []
        for sample in rows:
            key = sample_key(sample)
            if (name, key) in seen:
                duplicates[name] += 1
                continue
            seen[name, key] = sample["sample_id"]
            retained.append(sample)
        rows[:] = retained
    for path in checked_paths:
        relative = str(path.relative_to(root))
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if hashes != before_hashes:
        raise ValueError("Source files changed during preparation")
    samples = oracle+teacher
    if any(s["task_id"] not in by_id or by_id[s["task_id"]]["eval_group"] != "train"
           for s in samples):
        raise ValueError("Evaluation task in SFT output")
    report = {
        "preparation_version": "v0.4-A", "offline_execution_only": True,
        "dataset_manifest": manifest, "frozen_task_counts": dict(groups),
        "source_inventory": inventory, "episodes": episodes,
        "source_sha256": hashes, "original_sources_unchanged": True,
        "oracle_reference_export_rows": len(derived),
        "oracle_reference_export_matched": len(matched_derived),
        "unmatched_reference_rows": len(derived_keys-matched_derived),
        "trustworthy_decisions": {"oracle_reference": len(oracle), "teacher": len(teacher)},
        "independent_train_tasks": {
            "oracle_reference": len({s["task_id"] for s in oracle}),
            "teacher": len({s["task_id"] for s in teacher}),
            "combined": len({s["task_id"] for s in samples})},
        "real_teacher_episodes": sum(e["source"] == "llm" for e in episodes),
        "real_teacher_independent_tasks": len({e["task_id"] for e in episodes if e["source"] == "llm"}),
        "source_domain_distribution": {n: dict(Counter(s["domain"] for s in rows))
                                      for n, rows in (("oracle_reference", oracle), ("teacher", teacher))},
        "mechanism_coverage": {n: dict(Counter(s["mechanism_id"] for s in rows))
                               for n, rows in (("oracle_reference", oracle), ("teacher", teacher))},
        "target_kind_distribution": {n: dict(Counter(s["target_kind"] for s in rows))
                                    for n, rows in (("oracle_reference", oracle), ("teacher", teacher))},
        "trajectory_length_distribution": dict(Counter(e["recovery_steps"] for e in episodes)),
        "initial_error_distribution": dict(Counter(e["initial_error_type"] for e in episodes)),
        "duplicate_samples_removed": dict(duplicates), "exclusion_reasons": dict(drops),
        "cross_split_leakage": 0, "invalid_exported_conversations": 0,
        "causal_prompts_checked": True, "hidden_answer_leak_found": False,
        "trust_policy": "Successful full-Verifier episodes only; per-decision replay/behavior evidence; uncertain, failures, redundant mutations and post-success calls excluded. Historical assistant actions remain masked.",
        "history_policy": "Recorded model-visible history is authoritative. Historical calls were serialized individually by the existing Runtime; target raw multi-call groups and FIFO call_id mappings are retained.",
        "legacy_v03a_policy": "Nested legacy_v03a derivatives are counted in inventory below but not mixed with the frozen v0.3-B task/schema cohort."}
    legacy = reference/"legacy_v03a"
    if legacy.exists():
        report["legacy_v03a_derived_rows_excluded"] = {
            str(p.relative_to(reference)): len(read_jsonl(p)) for p in legacy.rglob("*sft*.jsonl")}
    return oracle, teacher, report


def length_summary(values):
    """Nearest-rank percentiles; empty cohorts never masquerade as measured zero-token data."""
    import math
    values = sorted(values)
    return {"count": len(values),
            **{name: values[max(0, math.ceil(q*len(values))-1)] if values else None
               for name, q in (("p50", .50), ("p95", .95), ("p99", .99))},
            "max": max(values) if values else None}

def encode_cohorts(oracle, teacher, adapter):
    """Encode every trusted sample without truncation, and select real Git/SQLite alignment examples."""
    stats, examples = {}, []
    for name, rows in (("oracle_reference", oracle), ("teacher", teacher)):
        for sample in rows:
            sample.update(adapter.encode(sample))
        total = sum(s["token_length"] for s in rows)
        supervision = sum(s["assistant_supervised_tokens"] for s in rows)
        stats[name] = {
            "lengths": length_summary([s["token_length"] for s in rows]),
            "over_max_seq_length": sum(s["over_max_seq_length"] for s in rows),
            "supervised_token_counts": length_summary([s["assistant_supervised_tokens"] for s in rows]),
            "total_tokens": total, "assistant_supervised_tokens": supervision,
            "assistant_supervised_ratio": supervision/total if total else None,
            "per_domain": {domain: length_summary([s["token_length"] for s in rows if s["domain"] == domain])
                           for domain in sorted({s["domain"] for s in rows})},
            "long_samples": [{"sample_id": s["sample_id"], "token_length": s["token_length"]}
                             for s in rows if s["over_max_seq_length"]]}
    for domain, kind in (("git", "tool_call"), ("sqlite", "final_answer")):
        candidates = [s for s in oracle+teacher if s["domain"] == domain and s["target_kind"] == kind
                      and sum(m["role"] == "tool" for m in s["messages"][:-1]) >= 2]
        if not candidates:
            raise ValueError(f"No verified multi-round {domain} example")
        sample = min(candidates, key=lambda s: s["token_length"])
        examples.append(adapter.alignment_example(sample, sample))
    combined = oracle+teacher
    stats["combined"] = {"lengths": length_summary([s["token_length"] for s in combined]),
                         "over_max_seq_length": sum(s["over_max_seq_length"] for s in combined)}
    return stats, examples

def write_readiness_report(output, audit, token_stats, status, blocker=None, mixed=False):
    """Summarize measured evidence and unresolved training choices without claiming model gains."""
    counts = audit["trustworthy_decisions"]
    lines = ["# RecoveryAgent v0.4-A Training Readiness", "",
             f"Status: **{status}**", "",
             f"- Oracle trusted decisions: {counts['oracle_reference']}",
             f"- Real Teacher trusted decisions: {counts['teacher']}",
             f"- Independent Train tasks: {audit['independent_train_tasks']}",
             f"- Real Teacher episodes/tasks audited: {audit['real_teacher_episodes']}/{audit['real_teacher_independent_tasks']}",
             f"- Matched reference rows: {audit['oracle_reference_export_matched']}/{audit['oracle_reference_export_rows']}",
             f"- Cross-split leakage: {audit['cross_split_leakage']}",
             f"- Source files unchanged: {audit['original_sources_unchanged']}", "",
             "## Trust and causality", "", audit["trust_policy"], "", audit["history_policy"], "",
             "Final-failed episodes are excluded conservatively; partial-action salvage is not attempted. "
             "Successful episodes still undergo per-call filtering. Diagnostic calls require existing positive "
             "behavior evidence; uncertain calls are excluded. Final answers require an autonomous successful stop. "
             "Privileged evidence is metadata only, never messages/tools. Historical failed calls remain visible "
             "as observed context but receive no loss.", "",
             "Same task under several prompts counts once in independent task coverage. IID validation/test "
             "may share mechanisms with Train but have separate frozen variants; OOD mechanisms are disjoint.", "",
             "## Official tokenizer and labels", ""]
    if blocker:
        lines += [f"BLOCKER: {blocker}", "",
                  "Conversational JSONL is available. Token statistics and masks are unvalidated; no substitute "
                  "tokenizer or hand-written chat template was used.", ""]
    else:
        lines += [
            "AutoTokenizer loads the repository's unmodified Qwen3 template with enable_thinking=False. "
            "The official template renders tool schema in System and Tool Responses as user/tool_response. "
            "It does not render call_id: original IDs, group order and parameter/response bindings remain "
            "in messages and tool_bindings and are validated FIFO before rendering.", "",
            "No native generation-mask markers exist in this pinned template. Full conversation IDs are "
            "checked against independent encoding of the rendered template. The official generation-prompt "
            "IDs must be an exact prefix of the full conversation. Only the final verified assistant "
            "content/tool calls and its im_end EOS receive labels; System/User/Tool/history and the "
            "generation header/empty thinking prefill are -100. No string-length offsets are used.", "",
            "Every retained decision is an individual causal multi-round prefix, not an episode with all "
            "assistant actions supervised. Future results of the target action are excluded.", "",
            f"Encoding configuration: max_seq_length={token_stats['max_seq_length']}, truncation=False.", "",
            f"Official model config native context: {token_stats.get('model_context_length')} tokens; "
            f"over native context: {token_stats['cohorts']['combined'].get('over_model_context')}. "
            "Tokenizer model_max_length is recorded separately and is not treated as the native model limit.", "",
            "| Source | Samples | P50 | P95 | P99 | Max | Over limit | Supervised tokens | Ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name in ("oracle_reference", "teacher"):
            row = token_stats["cohorts"][name]
            lengths = row["lengths"]
            ratio = row["assistant_supervised_ratio"]
            if ratio is not None:
                lines.append(f"| {name} | {lengths['count']} | {lengths['p50']} | {lengths['p95']} | "
                             f"{lengths['p99']} | {lengths['max']} | {row['over_max_seq_length']} | "
                             f"{row['assistant_supervised_tokens']} | {ratio:.4f} |")
            else:
                lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a | 0 | 0 | n/a |")
        lines += ["", "Percentiles use nearest rank. Long sequences are retained and listed, never truncated "
                  "or silently dropped. 8192 is a configurable candidate training length, not a measured "
                  "GPU-memory optimum. Examine mask_examples.json for full Git/SQLite token/label alignment.", ""]
    lines += ["## Dataset coverage and exclusions", "", chr(96)*3+"json",
              json.dumps({"domains": audit["source_domain_distribution"],
                          "target_kinds": audit["target_kind_distribution"],
                          "duplicates_removed": audit["duplicate_samples_removed"],
                          "exclusions": audit["exclusion_reasons"]}, indent=2), chr(96)*3, "",
              f"Mixed export: {'explicitly enabled with duplicate removal' if mixed else 'disabled; sources remain separate'}.", "",
              "## Before LoRA-SFT", "",
              "1. Review trusted Teacher coverage and exclusion evidence; 33 episodes of repeated tasks "
              "are not 33 independent tasks. Oracle/Mock results are not real-model performance.",
              "2. Choose Oracle/Teacher sampling weights explicitly. Existing cohorts have limited independent "
              "mechanism/variant coverage; no samples are fabricated to meet a quota.",
              "3. Decide context handling from the measured distribution. Keep final targets and tool groups intact.",
              "4. On the training host, pin the same tokenizer revision/template hashes, verify the training "
              "collator honors exported labels, and run a small CPU/GPU batch preflight before any training.",
              "5. Keep IID validation, IID test and OOD out of training and hyperparameter-negative mining.",
              "", "No weights, paid API requests, training code or gradient updates are part of this stage.", ""]
    (output/"readiness_report.md").write_text("\n".join(lines), encoding="utf-8")

def prepare(root, output, tokenizer_id, max_seq_length=8192, audit_only=False,
            local_files_only=True, revision=None, mixed=False):
    """Produce a fresh, separately versioned output; a tokenizer blocker preserves the audit."""
    from .qwen3_tokenizer import load_tokenizer, Qwen3Adapter, tokenizer_manifest
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be new/empty; previous artifacts are never overwritten")
    if any(output.resolve().is_relative_to(p.resolve()) for p in source_directories(root)+[
            root/"data", root/"outputs/v0.3b_mock_verified"]):
        raise ValueError("Output must be separate from frozen sources")
    output.mkdir(parents=True, exist_ok=True)
    oracle, teacher, audit = audit_sources(root)
    write_json(output/"source_audit.json", audit)
    blocker, status = None, "BLOCKED"
    stats = {"status": "unvalidated", "max_seq_length": max_seq_length, "truncation": False}
    examples = []
    try:
        if audit_only:
            raise RuntimeError("Audit-only requested; real tokenizer encoding was not performed")
        tokenizer = load_tokenizer(tokenizer_id, local_files_only=local_files_only, revision=revision)
        adapter = Qwen3Adapter(tokenizer, max_seq_length=max_seq_length)
        cohorts, examples = encode_cohorts(oracle, teacher, adapter)
        metadata = tokenizer_manifest(tokenizer, tokenizer_id)
        model_context = metadata["model_config_max_position_embeddings"]
        for name, rows in (("oracle_reference", oracle), ("teacher", teacher), ("combined", oracle+teacher)):
            cohorts[name]["over_model_context"] = sum(s["token_length"] > model_context for s in rows) if model_context else None
        stats.update(status="verified", tokenizer=metadata, model_context_length=model_context,
                     template=adapter.capabilities, cohorts=cohorts,
                     per_sample={s["sample_id"]: {"length": s["token_length"],
                                                 "supervised": s["assistant_supervised_tokens"]}
                                 for s in oracle+teacher})
        status = "CPU_DATA_READY" if not cohorts["combined"]["over_max_seq_length"] else "CONTEXT_REVIEW_REQUIRED"
    except Exception as exc:
        blocker = f"{type(exc).__name__}: {exc}"
        stats.update(status="blocked", blocker=blocker)
        encoded_fields = ("input_ids", "attention_mask", "labels", "target_token_span",
                          "token_length", "assistant_supervised_tokens", "over_max_seq_length", "truncated")
        for sample in oracle+teacher:
            for key in encoded_fields:
                sample.pop(key, None)
        examples = []
    write_jsonl(output/"oracle_sft.jsonl", oracle)
    write_jsonl(output/"teacher_sft.jsonl", teacher)
    if mixed:
        combined, seen = [], set()
        for sample in oracle+teacher:
            key = sample_key(sample)
            if key not in seen:
                seen.add(key)
                combined.append(sample)
        write_jsonl(output/"mixed_sft.jsonl", combined)
    write_json(output/"token_stats.json", stats)
    write_json(output/"mask_examples.json", {"status": stats["status"], "examples": examples})
    write_readiness_report(output, audit, stats, status, blocker, mixed)
    print(f"READINESS {status}: Oracle={len(oracle)}, Teacher={len(teacher)}; {output}", flush=True)
    if blocker:
        print(f"BLOCKER {blocker}", flush=True)
    return 0 if status == "CPU_DATA_READY" else 2
