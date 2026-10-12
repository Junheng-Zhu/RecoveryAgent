"""Freeze a local acceptance summary after all offline validation commands finish."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    # This report reads validated artifacts only; it never executes models or tools.
    paths={"mock":"outputs/v0.3b_mock_verified","clean":"outputs/v0.3b_clean_verified",
           "online":"outputs/v0.3b_online_review"}
    quality={name:read(path+"/quality_report.json") for name,path in paths.items()}
    if not all(q["passed"] for q in quality.values()): raise ValueError("Data validation incomplete")
    logs=ROOT/"outputs/v0.3b_pytest_acceptance.log"
    text=logs.read_text(encoding="utf-16")
    match=re.search(r"(\d+) passed in ([^\r\n]+)",text)
    if not match or " FAILED " in text: raise ValueError("Full pytest acceptance is incomplete")
    hashes=read("outputs/v0.3b_original_artifact_hashes.json")
    changed=[name for name,h in hashes.items() if digest(ROOT/name)!=h]
    if changed: raise ValueError("Original frozen artifact changed")
    dataset=read("data/v0.3b_seed42_final/manifest.json")
    online=read(paths["online"]+"/report.json")
    budget=read("outputs/v0.3b_online_smoke/budget.json")["state"]
    exports={"reference":read("outputs/v0.3b_sft_reference_final/report.json"),
             "real_llm":read("outputs/v0.3b_llm_trusted_export/report.json"),
             "legacy_real_llm":read("outputs/v0.3b_sft_reference_final/legacy_v03a/report.json")}
    exports_manifest={}
    for directory in ("outputs/v0.3b_sft_reference_final","outputs/v0.3b_llm_trusted_export"):
        for p in (ROOT/directory).rglob("*"):
            if p.is_file(): exports_manifest[str(p.relative_to(ROOT))]=digest(p)
    snapshot=ROOT/"data/v0.3b_seed42_final"
    repeated=ROOT/"data/v0.3b_reproduced"
    reproducible={p.name:p.read_bytes() for p in snapshot.iterdir()}=={p.name:p.read_bytes() for p in repeated.iterdir()}
    if not reproducible: raise ValueError("Snapshot regeneration differs")
    summary={"dataset_version":"0.3-B","date":"2026-10-12","pytest":{"passed":int(match[1]),"duration":match[2],"log":str(logs)},
             "dataset_manifest":dataset,"dataset_byte_reproduction":reproducible,"quality":quality,
             "exports":exports,"export_artifact_hashes":exports_manifest,
             "online_new_episodes":online["task_count"],"online_requests":budget["requests"],
             "online_model":"qwen3.8-max-0902","online_usage":online["total_usage"],
             "online_successes":quality["online"]["successful"],"online_termination":online["termination_reasons"],
             "online_conservative_cost_cny":budget["reserved_or_used_cost_cny"],
             "online_resume_new_requests":0,"online_batch_executed":False,
             "original_artifacts_verified":len(hashes),"original_artifacts_changed":changed,
             "core_extension":"stateful/collector.py optional extension hooks; original defaults retained",
             "gpu":{"baseline_smoke_ready":True,"formal_lora_sft_ready":False,"training_executed":False,
                    "blockers":["Only one new real train episode, SQLite only","Teacher bulk collection remains user-triggered",
                                "Tokenizer/chat-template and final-assistant label mask require GPU-stage validation",
                                "No reliably verified real DPO pairs yet"]}}
    output=ROOT/"outputs/v0.3b_acceptance.json"
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(output),"pytest_passed":int(match[1]),
                      "new_real_episodes":summary["online_new_episodes"]},indent=2))
if __name__=="__main__": main()
