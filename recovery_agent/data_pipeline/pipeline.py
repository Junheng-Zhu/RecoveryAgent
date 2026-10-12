import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from .common import VERSION, write_json, write_jsonl, fingerprint
from .dataset import generate_dataset, task_statistics, DEFAULT_CONFIG
from .trajectories import collect_trajectory, build_sft, trajectory_statistics
from .preferences import build_preferences
from .quality import validate_data, validate_directory

# 一键串联任务生成、Oracle 收集、偏好分支、SFT 导出、质量验证和版本化落盘。
# 拒绝覆盖非空目录；保存后再次读取并验证，以覆盖序列化和文件完整性问题。
def run_pipeline(output, config=None, preference_decisions=1, progress=None):
    if type(preference_decisions) is not int or preference_decisions < 1:
        raise ValueError("preference_decisions must be a positive integer")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be empty; use a new version directory for reproducibility")
    output.mkdir(parents=True, exist_ok=True)
    tasks, config = generate_dataset(config)
    trajectories, pairs, discarded = [], [], Counter()
    for number, task in enumerate(tasks, 1):
        trajectory = collect_trajectory(task)
        trajectories.append(trajectory)
        # Test is collected only for environment certification/replay, never candidate construction.
        # 测试集只收集认证轨迹；不执行偏好候选构建，也不参与训练导出。
        if task["split"] == "test":
            discarded["test_excluded"] += 1
        else:
            generated, reasons = build_preferences(task, trajectory, preference_decisions)
            pairs += generated
            discarded.update(reasons)
        if progress and (number % 50 == 0 or number == len(tasks)):
            progress(f"Collected and branched {number}/{len(tasks)} tasks")
    samples, sft_skipped = build_sft(trajectories)
    if progress:
        progress("Validating all causal inputs, replaying successful trajectories and preference branches")
    quality = validate_data(tasks, trajectories, samples, pairs, config)
    for split in ("train", "val", "test"):
        write_jsonl(output / "tasks" / f"{split}.jsonl", [t for t in tasks if t["split"] == split])
        write_jsonl(output / "trajectories" / f"{split}.jsonl",
                    [t for t in trajectories if t["split"] == split])
        if split != "test":
            write_jsonl(output / "sft" / f"{split}.jsonl", [s for s in samples if s["split"] == split])
            write_jsonl(output / "preferences" / f"{split}.jsonl", [p for p in pairs if p["split"] == split])
    report = {"version": VERSION, "task_statistics": task_statistics(tasks),
              "trajectory_statistics": trajectory_statistics(trajectories),
              "sft_statistics": {"sample_counts": dict(Counter(s["split"] for s in samples)),
                                 "discard_reasons": sft_skipped,
                                 "terminal_targets": sum(not s["messages"][-1].get("tool_calls") for s in samples)},
              "preference_statistics": {"pair_counts": dict(Counter(p["split"] for p in pairs)),
                                       "candidate_types": dict(Counter(p["candidate_kind"] for p in pairs)),
                                       "accepted_reasons": dict(Counter(p["preference_reason"] for p in pairs)),
                                       "discard_reasons": dict(discarded),
                                       "decisions_per_task": preference_decisions,
                                       "rule": "verified chosen success vs terminal failure or immutable audit violation"},
              "quality": quality}
    write_json(output / "report.json", report)
    write_json(output / "quality_report.json", quality)
    write_json(output / "config.json", config)
    # 摘要覆盖实际落盘字节；manifest 自身另在复现比较时逐字节检查。
    artifacts = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = {"dataset_version": VERSION, "config": config, "config_hash": fingerprint(config),
                "pipeline_config": {"preference_decisions": preference_decisions},
                "artifacts": artifacts}
    write_json(output / "manifest.json", manifest)
    # Re-read persisted payloads: JSONL serialization and artifact hashes must pass too.
    saved_quality = validate_directory(output)
    if progress:
        progress("Persisted artifacts re-read and independently verified")
    return report, manifest, saved_quality

# 先比较配置与产物 manifest，再逐文件比较字节，验证完整流水线复现一致。
def compare_directories(first, second):
    first, second = Path(first), Path(second)
    a = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    b = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    if a != b:
        raise ValueError("Reproduction manifest differs")
    for name in ["manifest.json", *a["artifacts"]]:
        if (first / name).read_bytes() != (second / name).read_bytes():
            raise ValueError(f"Reproduction bytes differ: {name}")
    return {"identical": True, "artifacts_compared": len(a["artifacts"]) + 1}
