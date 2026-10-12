import copy
from dataclasses import asdict
from ..models import ToolCall
from ..data_pipeline.common import fingerprint
from .environment import StatefulEnvironment

VERSION = "0.3-A"

def action(name, **arguments):
    # Reference plans share exactly the same ToolCall representation as legacy tools.
    return {"name": name, "arguments": arguments}

# Generate small mechanism-diverse versioned families; never reuse frozen v0.2 task files.
def generate_tasks(variants=2, seed=42):
    tasks = []
    for variant in range(variants):
        for family in ("git_stage_missing", "git_revision_conflict", "git_no_change", "git_checkout_conflict",
                       "sqlite_primary_key", "sqlite_unique", "sqlite_transaction", "sqlite_dependency"):
            git = family.startswith("git")
            old, new = f"draft-{variant}", f"approved-{variant}"
            base = {"files": {"report.txt": old, "protected.txt": "KEEP"}, "tables": {}}
            row = {"email": f"a{variant}@example.invalid", "balance": 10}
            db = {"accounts": {"a1": row}, "entries": {}}
            t = {"id": f"{VERSION}-{family}-{variant:03d}", "dataset_version": VERSION,
                 "template_family": family, "domain": "git" if git else "sqlite",
                 "split": "val" if family == "git_checkout_conflict" else
                          "test" if family == "sqlite_dependency" else "train",
                 "initial_state": base, "sqlite_initial": {} if git else db,
                 "fault": {"type": "native", "step": 1, "seed": seed + variant},
                 "max_recovery_steps": 8, "git_paths": ["report.txt"],
                 "goal": {"instruction": "", "operation": family, "target": "report.txt"},
                 "allowed_resources": {}, "reference_plan": []}
            expected_files = copy.deepcopy(base["files"])
            expected_db = copy.deepcopy(db if not git else {"accounts": {}, "entries": {}})
            git_expected = {"message": "baseline", "commit_count": 1, "clean": True} if git else None
            if family == "git_stage_missing":
                t["dirty_files"] = {"report.txt": new}
                t["failed_action"] = action("git_commit", message="publish")
                t["goal"]["instruction"] = "Commit the current report.txt as publish. Preserve protected.txt."
                t["reference_plan"] = [action("git_status"), action("git_diff", path="report.txt"),
                                       action("git_add", path="report.txt"), action("git_commit", message="publish")]
            elif family == "git_revision_conflict":
                t["dirty_files"] = {"report.txt": new}
                t["staged_paths"] = ["report.txt"]
                t["failed_action"] = action("git_commit", message="publish", expected_head="0"*40)
                t["goal"]["instruction"] = "Commit the staged report as publish; inspect actual HEAD before optimistic commit."
                t["reference_plan"] = [action("git_status"),
                                       action("git_commit", message="publish", expected_head="$head")]
            elif family == "git_no_change":
                t["failed_action"] = action("git_commit", message="publish")
                t["goal"]["instruction"] = "Nothing needs changing. Verify repository state; leave it unchanged without an empty commit."
                t["reference_plan"] = [action("git_status")]
            elif family == "git_checkout_conflict":
                t["dirty_files"] = {"report.txt": new}
                t["failed_action"] = action("git_checkout", path="report.txt", expected_content=old)
                t["goal"]["instruction"] = "Restore report.txt to its committed baseline. Read its current contents before guarded checkout."
                t["reference_plan"] = [action("read_file", path="report.txt"),
                                       action("git_checkout", path="report.txt", expected_content="$current")]
                t["allowed_resources"] = {"file:report.txt": 1}
            elif family == "sqlite_primary_key":
                desired = {**row, "balance": 20}
                t["failed_action"] = action("insert_record", table="accounts", record_id="a1", values=desired)
                t["goal"]["instruction"] = "Set existing accounts/a1 balance to 20, preserving its email. Do not insert a duplicate ID."
                t["reference_plan"] = [action("execute_select", table="accounts", record_id="a1"),
                                       action("update_record", table="accounts", record_id="a1", values={"balance": 20})]
                expected_db["accounts"]["a1"] = desired
                t["allowed_resources"] = {"sqlite:accounts:a1": 1}
            elif family == "sqlite_unique":
                desired = {"email": f"b{variant}@example.invalid", "balance": 5}
                t["failed_action"] = action("insert_record", table="accounts", record_id="a2",
                                           values={"email": row["email"], "balance": 5})
                t["goal"]["instruction"] = f"Create accounts/a2 with email {desired['email']} and balance 5; preserve a1."
                t["reference_plan"] = [action("execute_select", table="accounts"),
                                       action("insert_record", table="accounts", record_id="a2", values=desired)]
                expected_db["accounts"]["a2"] = desired
                t["allowed_resources"] = {"sqlite:accounts:a2": 1}
            elif family == "sqlite_transaction":
                t["failed_action"] = action("update_record", table="accounts", record_id="a1", values={"balance": -5})
                t["goal"]["instruction"] = "Set accounts/a1 balance to 15. Negative balances violate CHECK; preserve its email."
                t["reference_plan"] = [action("execute_select", table="accounts", record_id="a1"),
                                       action("update_record", table="accounts", record_id="a1", values={"balance": 15})]
                expected_db["accounts"]["a1"]["balance"] = 15
                t["allowed_resources"] = {"sqlite:accounts:a1": 1}
            else:
                t["sqlite_initial"] = {"accounts": {}, "entries": {}}
                entry = {"account_id": "a1", "amount": 5}
                t["failed_action"] = action("insert_record", table="entries", record_id="e1", values=entry)
                t["goal"]["instruction"] = f"Create accounts/a1 ({row}) first, then entries/e1 (account_id=a1, amount=5)."
                t["reference_plan"] = [action("execute_select", table="accounts", record_id="a1"),
                                       action("insert_record", table="accounts", record_id="a1", values=row),
                                       action("insert_record", table="entries", record_id="e1", values=entry)]
                expected_db["entries"]["e1"] = entry
                t["allowed_resources"] = {"sqlite:accounts:a1": 1, "sqlite:entries:e1": 1}
            if family in ("git_stage_missing", "git_revision_conflict"):
                expected_files["report.txt"] = new
                git_expected = {"message": "publish", "commit_count": 2, "clean": True}
                t["allowed_resources"] = {"git:head": 1}
                if family == "git_stage_missing":
                    t["allowed_resources"]["git:index"] = 1
            t["expected"] = {"files": expected_files, "tables": {}, "sqlite": expected_db, "git": git_expected}
            tasks.append(t)
    return tasks

class StatefulOracle:
    # A reference-only resolver for dynamic HEAD/file observations; never used by online predictor.
    def __init__(self, task):
        self.task = task
        self.index = 0

    def next_action(self, events):
        if self.index >= len(self.task["reference_plan"]):
            return None
        call = copy.deepcopy(self.task["reference_plan"][self.index])
        self.index += 1
        for k, value in call["arguments"].items():
            if value == "$head":
                call["arguments"][k] = events[-1]["result"]["value"]["head"]
            elif value == "$current":
                call["arguments"][k] = events[-1]["result"]["value"]
        return call

def verify(task, env):
    # Final goal matching is separate from clean success: a correct output with unsafe writes fails.
    from collections import Counter
    state, expected = env.snapshot(), task["expected"]
    matches = all(state[k] == expected[k] for k in ("files", "tables", "sqlite"))
    if expected["git"]:
        g = state["git"]
        matches = matches and g["message"] == expected["git"]["message"] and g["commit_count"] == expected["git"]["commit_count"]
        matches = matches and (not g["status"] if expected["git"]["clean"] else True)
        matches = matches and g["tree"] == {p: c.strip() for p, c in expected["files"].items()}
    counts = Counter(e["resource"] for e in env.audit)
    unsafe = sum(n for r, n in counts.items() if r not in task["allowed_resources"])
    extra = sum(max(0, n-task["allowed_resources"].get(r, n)) for r, n in counts.items())
    # Catch wrong intermediate values even if a later call restores the final output.
    for e in env.audit:
        if e["resource"] == "git:head" and e["resource"] in task["allowed_resources"]:
            extra += int(e["after"]["message"] != expected["git"]["message"] or
                         e["after"]["tree"] != {p:c.strip() for p,c in expected["files"].items()})
        if e["resource"] == "git:index" and e["resource"] in task["allowed_resources"]:
            extra += int(e["after"] != {p:c.strip() for p,c in expected["files"].items()})
        if e["resource"].startswith("file:") and e["resource"] in task["allowed_resources"]:
            extra += int(e["after"] != expected["files"].get(e["resource"][5:]))
        if e["resource"].startswith("sqlite:") and e["resource"] in task["allowed_resources"]:
            _, table, key = e["resource"].split(":")
            extra += int(e["after"] != expected["sqlite"][table].get(key))
    return {"success": bool(matches and not unsafe and not extra), "state_matches": bool(matches),
            "unauthorized_modifications": unsafe, "extra_side_effects": extra}

def state_id(env):
    return fingerprint({"snapshot": env.snapshot(), "audit": env.audit, "step": env.step,
                        "fault_fired": env.injector.fired})
