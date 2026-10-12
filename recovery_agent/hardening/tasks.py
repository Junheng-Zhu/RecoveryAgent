import copy
import json
from ..stateful.tasks import generate_tasks as legacy_tasks, action, StatefulOracle
from ..data_pipeline.common import canonical, fingerprint, write_json, write_jsonl, read_jsonl

VERSION="0.3-B"
NEW_FAMILIES=("git_wrong_branch","git_stale_index","git_multifile_revision","git_compensating_commit",
              "sqlite_delete_dependency","sqlite_unique_swap","sqlite_relink","mixed_db_git_report")
OOD_FAMILIES={"git_compensating_commit","sqlite_relink"}

def new_task(family,variant,seed):
    # Each family changes recovery mechanics, not only resource names or values.
    old,new=f"draft-{variant}",f"approved-{variant}"
    files={"report.txt":old,"protected.txt":"KEEP"}
    account={"email":f"a{variant}@example.invalid","balance":10+variant}
    db={"accounts":{"a1":account},"entries":{}}
    git=family.startswith("git") or family.startswith("mixed")
    t={"id":"","dataset_version":VERSION,"template_family":family,"domain":"mixed" if family.startswith("mixed") else "git" if git else "sqlite",
       "initial_state":{"files":files,"tables":{}},"sqlite_initial":{"accounts":{},"entries":{}} if git else copy.deepcopy(db),
       "goal":{"instruction":"","operation":"repair","target":"report.txt"},
       "fault":{"type":"native","step":1,"seed":seed+variant},"git_paths":["report.txt"],
       "max_recovery_steps":12,"allowed_resources":{},"reference_plan":[],
       "variant_id":variant,"origin":"new_mechanism"}
    expected={"files":copy.deepcopy(files),"tables":{},"sqlite":copy.deepcopy(t["sqlite_initial"]),
              "git":{"message":"publish","commit_count":2,"clean":True,"branch":"main"} if git else None}
    if family in ("git_wrong_branch","git_stale_index"):
        t["dirty_files"]={"report.txt":new}
        expected["files"]["report.txt"]=new
        t["failed_action"]=action("git_commit",message="publish")
        t["allowed_resources"]={"git:index":1,"git:head":1}
        if family=="git_wrong_branch":
            t.update(initial_branch="feature",branches=["main","feature"],commit_branch="main")
            t["allowed_resources"]["git:branch"]=1
            t["goal"]["instruction"]="Publish the current report.txt on main, preserving all other files. Inspect branch before committing."
            t["reference_plan"]=[action("git_status"),action("git_switch",branch="main"),
                                 action("git_add",path="report.txt"),action("git_commit",message="publish")]
        else:
            t.update(staged_paths=["report.txt"],require_fresh_index=True)
            # Stage v1, then replace only the working tree with v2.
            t["post_stage_files"]={"report.txt":new+"-latest"}
            expected["files"]["report.txt"]=new+"-latest"
            t["goal"]["instruction"]="Publish the latest working-tree report.txt; the index may contain an older version. Refresh it before commit."
            t["reference_plan"]=[action("git_status"),action("git_diff",path="report.txt"),
                                 action("git_add",path="report.txt"),action("git_commit",message="publish")]
    elif family=="git_multifile_revision":
        a,b=canonical({"revision":variant}),canonical({"revision":variant+1})
        files.update({"report.txt":a,"metadata.json":a})
        expected["files"]=copy.deepcopy(files)
        expected["files"].update({"report.txt":b,"metadata.json":b})
        t.update(dirty_files={"report.txt":b},staged_paths=["report.txt"],git_paths=["report.txt","metadata.json"],
                 require_fresh_index=True,revision_dependency=["report.txt","metadata.json"],
                 semantic_json_files=["report.txt","metadata.json"])
        t["failed_action"]=action("git_commit",message="publish")
        t["goal"]["instruction"]="Read report.txt, make metadata.json carry exactly its revision, and commit both required files as publish."
        t["reference_plan"]=[action("read_file",path="report.txt"),
                             action("write_file",path="metadata.json",content="$revision_json"),
                             action("git_add",path="metadata.json"),action("git_commit",message="publish")]
        t["allowed_resources"]={"file:metadata.json":1,"git:index":1,"git:head":1}
    elif family=="git_compensating_commit":
        files["backup.txt"]=new
        expected["files"]=copy.deepcopy(files)
        expected["files"]["report.txt"]=new
        expected["git"].update(message="repair",commit_count=3)
        t.update(bootstrap_commit={"files":{"report.txt":"obsolete"},"message":"stale"},forward_only=True)
        t["failed_action"]=action("git_checkout",path="report.txt",expected_content="obsolete")
        t["goal"]["instruction"]="An incorrect report was already committed. Read backup.txt and publish it in report.txt with a new repair commit; preserve prior history."
        t["reference_plan"]=[action("read_file",path="backup.txt"),action("write_file",path="report.txt",content="$current"),
                             action("git_add",path="report.txt"),action("git_commit",message="repair")]
        t["allowed_resources"]={"file:report.txt":1,"git:index":1,"git:head":1}
    elif family=="sqlite_delete_dependency":
        entry={"account_id":"a1","amount":5}
        t["sqlite_initial"]["entries"]["e1"]=entry
        expected["sqlite"]={"accounts":{},"entries":{}}
        t["failed_action"]=action("delete_record",table="accounts",record_id="a1")
        t["goal"]["instruction"]="Delete entries/e1, then accounts/a1; leave no dependent orphan and preserve unrelated state."
        t["reference_plan"]=[action("execute_select",table="entries"),action("delete_record",table="entries",record_id="e1"),
                             action("delete_record",table="accounts",record_id="a1")]
        t["allowed_resources"]={"sqlite:entries:e1":1,"sqlite:accounts:a1":1}
    elif family=="sqlite_unique_swap":
        second={"email":f"b{variant}@example.invalid","balance":20+variant}
        t["sqlite_initial"]["accounts"]["a2"]=second
        expected["sqlite"]=copy.deepcopy(t["sqlite_initial"])
        expected["sqlite"]["accounts"]["a1"]["email"]=second["email"]
        expected["sqlite"]["accounts"]["a2"]["email"]=account["email"]
        t["failed_action"]=action("update_record",table="accounts",record_id="a1",values={"email":second["email"]})
        t["goal"]["instruction"]="Swap the emails of accounts/a1 and a2, preserving balances and IDs. An unused temporary email is permitted while resolving UNIQUE."
        t["reference_plan"]=[action("execute_select",table="accounts"),
                             action("update_record",table="accounts",record_id="a1",values={"email":f"temporary{variant}@example.invalid"}),
                             action("update_record",table="accounts",record_id="a2",values={"email":account["email"]}),
                             action("update_record",table="accounts",record_id="a1",values={"email":second["email"]})]
        t["allowed_resources"]={"sqlite:accounts:a1":2,"sqlite:accounts:a2":1}
        t["temporary_email_resources"]=["sqlite:accounts:a1"]
    elif family=="sqlite_relink":
        t["sqlite_initial"]["entries"]["e1"]={"account_id":"a1","amount":5}
        second={"email":f"b{variant}@example.invalid","balance":account["balance"]}
        expected["sqlite"]={"accounts":{"a2":second},"entries":{"e1":{"account_id":"a2","amount":5}}}
        t["failed_action"]=action("update_record",table="entries",record_id="e1",values={"account_id":"a2"})
        t["goal"]["instruction"]=f"Create accounts/a2 with email {second['email']} and the current a1 balance, relink entries/e1 to a2, then delete a1; preserve entry amount."
        t["reference_plan"]=[action("execute_select",table="accounts",record_id="a1"),
                             action("insert_record",table="accounts",record_id="a2",values={"email":second["email"],"balance":"$balance"}),
                             action("update_record",table="entries",record_id="e1",values={"account_id":"a2"}),
                             action("delete_record",table="accounts",record_id="a1")]
        t["allowed_resources"]={"sqlite:accounts:a2":1,"sqlite:entries:e1":1,"sqlite:accounts:a1":1}
    else:
        files["report.txt"]=canonical({"total":0})
        t["sqlite_initial"]=copy.deepcopy(db)
        t["sqlite_initial"]["entries"]={"e1":{"account_id":"a1","amount":5+variant},
                                        "e2":{"account_id":"a1","amount":7+variant}}
        expected["sqlite"]=copy.deepcopy(t["sqlite_initial"])
        expected["files"]=copy.deepcopy(files)
        expected["files"]["report.txt"]=canonical({"total":12+2*variant})
        t["semantic_json_files"]=["report.txt"]
        t["failed_action"]=action("git_commit",message="publish")
        t["goal"]["instruction"]="Query SQLite entries, sum their amounts, write report.txt as a JSON object with only total, then stage and commit as publish. Do not modify the database."
        t["reference_plan"]=[action("execute_select",table="entries"),action("calculate",expression="$entries_expression"),
                             action("write_file",path="report.txt",content="$total_json"),
                             action("git_add",path="report.txt"),action("git_commit",message="publish")]
        t["allowed_resources"]={"file:report.txt":1,"git:index":1,"git:head":1}
    t["expected"]=expected
    return t

def namespace(task,variant):
    # Variant isolation changes resource namespaces, not the claimed recovery mechanism.
    import re
    mappings={p:f"case_{variant:03d}/"+p for p in task["initial_state"]["files"]}
    mappings.update({k:f"{k}_{variant:03d}" for k in ("a1","a2","e1","e2")})
    def walk(value):
        if isinstance(value,dict):
            return {walk(k):walk(v) for k,v in value.items()}
        if isinstance(value,list):
            return [walk(v) for v in value]
        if isinstance(value,str):
            if value in mappings: return mappings[value]
            if value.startswith("file:"): return "file:"+mappings.get(value[5:],value[5:])
            if value.startswith("sqlite:"):
                parts=value.split(":");parts[-1]=mappings.get(parts[-1],parts[-1]);return ":".join(parts)
        return value
    task=walk(task)
    text=task["goal"]["instruction"]
    for old,new in sorted(mappings.items(),key=lambda x:-len(x[0])):
        text=re.sub(r"(?<![\w@])"+re.escape(old)+r"(?![\w@])",lambda _:new,text)
    task["goal"]["instruction"]=text+f" Workspace namespace: case_{variant:03d}/."
    task["goal"]["operation"]=task["domain"]
    return task

def generate_dataset(variants=6,seed=42):
    # Mechanism OOD is whole-family held out; IID uses disjoint variant IDs within seen families.
    if variants<6:
        raise ValueError("At least 6 variants required for train/IID/test separation")
    rows=[]
    for variant in range(variants):
        for legacy in legacy_tasks(1,seed):
            family=legacy["template_family"]
            t=next(x for x in legacy_tasks(variant+1,seed) if x["template_family"]==family and x["id"].endswith(f"-{variant:03d}"))
            t=copy.deepcopy(t)
            t.update(dataset_version=VERSION,variant_id=variant,origin="legacy_mechanism")
            if t["expected"]["git"]:
                t["expected"]["git"]["branch"]="main"
            rows.append(t)
        rows.extend(new_task(f,variant,seed) for f in NEW_FAMILIES)
    rows=[namespace(t,t["variant_id"]) for t in rows]
    for task in rows:
        family,v=task["template_family"],task["variant_id"]
        heldout=family in OOD_FAMILIES or family=="sqlite_dependency"
        group="ood" if heldout else "iid_validation" if v%6==4 else "iid_test" if v%6==5 else "train"
        task.update(split="train" if group=="train" else "val" if group=="iid_validation" else "test",
                    eval_group=group,mechanism_id=family,id=f"{VERSION}-{family}-{v:03d}")
    config={"dataset_version":VERSION,"seed":seed,"variants":variants,"iid_variant_rule":"mod6=4 validation;mod6=5 test;else train",
            "ood_families":sorted(OOD_FAMILIES|{"sqlite_dependency"})}
    return rows,config

class Oracle(StatefulOracle):
    # Resolve actions from actual earlier observations; Oracle is never a predictor input.
    def next_action(self,events):
        if self.index>=len(self.task["reference_plan"]):
            return None
        call=copy.deepcopy(self.task["reference_plan"][self.index])
        self.index+=1
        last=events[-1]["result"]["value"] if events else None
        def resolve(value):
            if isinstance(value,dict):
                return {k:resolve(v) for k,v in value.items()}
            if value=="$head": return last["head"]
            if value=="$current": return last
            if value=="$revision_json": return canonical({"revision":json.loads(last)["revision"]})
            if value=="$balance": return last[0]["balance"]
            if value=="$entries_expression": return " + ".join(str(r["amount"]) for r in last)
            if value=="$total_json": return canonical({"total":last})
            return value
        call["arguments"]=resolve(call["arguments"])
        return call

def freeze(output,variants=6,seed=42,schema_hash=""):
    # New immutable directory; dataset regeneration must produce byte-identical payloads.
    from pathlib import Path
    import hashlib
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Dataset directory is already populated")
    tasks,config=generate_dataset(variants,seed)
    output.mkdir(parents=True,exist_ok=True)
    write_jsonl(output/"tasks.jsonl",tasks)
    for group in ("train","iid_validation","iid_test","ood"):
        write_jsonl(output/f"{group}.jsonl",[t for t in tasks if t["eval_group"]==group])
    write_json(output/"config.json",config)
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    manifest={"config":config,"schema_hash":schema_hash,"task_digest":fingerprint(tasks),"artifacts":artifacts}
    write_json(output/"manifest.json",manifest)
    return tasks,manifest

def validate_frozen(output,schema_hash):
    from pathlib import Path
    import hashlib
    output=Path(output)
    manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_hash"]!=schema_hash:
        raise ValueError("Tool schema changed")
    for name,digest in manifest["artifacts"].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Dataset artifact hash mismatch")
    config=manifest["config"]
    tasks,again=generate_dataset(config["variants"],config["seed"])
    if tasks!=read_jsonl(output/"tasks.jsonl") or again!=config or fingerprint(tasks)!=manifest["task_digest"]:
        raise ValueError("Dataset does not reproduce")
    train=[t for t in tasks if t["split"]=="train"]
    ids={t["id"] for t in train}
    train_families={t["mechanism_id"] for t in train}
    if ids & {t["id"] for t in tasks if t["split"]!="train"}:
        raise ValueError("Instance leakage")
    if train_families & {t["mechanism_id"] for t in tasks if t["eval_group"]=="ood"}:
        raise ValueError("OOD mechanism leakage")
    return tasks,manifest
