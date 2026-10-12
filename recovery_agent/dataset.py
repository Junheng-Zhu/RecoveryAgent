import copy
import json
import random
from pathlib import Path
from .faults import FAULT_TYPES

FAMILIES = ("copy_note", "copy_receipt", "upper_name", "upper_label",
            "record_status", "record_quota", "sum_invoice", "sum_stock",
            "ratio_report", "ratio_summary")
SPLITS = {f: "train" for f in FAMILIES}
SPLITS.update(upper_label="val", sum_stock="val",
              record_quota="test", ratio_summary="test")

def make_task(family, fault, variant, seed):
    rng = random.Random(f"{seed}:{family}:{variant}")
    number = rng.randint(2, 10000) + variant * 10001
    initial = {"files": {"protected.txt": "DO NOT MODIFY"}, "tables": {}}
    goal = {"instruction": "", "target": "result.txt"}
    if family.startswith(("copy", "upper")):
        value = f"{family} item {number}"
        goal.update(operation="copy" if family.startswith("copy") else "uppercase",
                    source="input.txt", backup="backup.txt")
        initial["files"].update({"input.txt": value, "backup.txt": value})
        output = value if goal["operation"] == "copy" else value.upper()
        action = {"name": "write_file", "arguments": {"path": "result.txt", "content": output}}
        plan = []
        if fault == "state_missing":
            plan += [{"name": "read_file", "arguments": {"path": "backup.txt"}},
                     {"name": "write_file", "arguments": {"path": "input.txt", "content": value}}]
        plan += [{"name": "read_file", "arguments": {"path": "input.txt"}}, action]
        goal["instruction"] = f"{goal['operation']} input.txt to result.txt; restore from backup.txt if missing."
    elif family.startswith("record"):
        base = {"name": f"user-{number}", "status": "pending", "quota": number}
        desired = {"status": "ready"} if family.endswith("status") else {"quota": number + 3}
        goal.update(operation="record", table="users", record_id="u1",
                    values=desired, backup_table="backup_users")
        initial["tables"] = {"users": {"u1": base}, "backup_users": {"u1": copy.deepcopy(base)}}
        action = {"name": "update_record", "arguments": {
            "table": "users", "record_id": "u1", "values": desired}}
        plan = [{"name": "query_table", "arguments": {"table": "users", "record_id": "u1"}}]
        if fault == "state_missing":
            plan = [{"name": "query_table", "arguments": {"table": "backup_users", "record_id": "u1"}},
                    {"name": "update_record", "arguments": {
                        "table": "users", "record_id": "u1", "values": {**base, **desired}, "create": True}}]
        else:
            plan += [action]
        goal["instruction"] = f"Set users/u1 fields to {desired}; recover missing row from backup_users."
    else:
        expression = f"{number} + 3" if family.startswith("sum") else f"{number} / 2"
        output = str(number + 3 if family.startswith("sum") else number / 2)
        goal.update(operation="compute", expression=expression)
        initial["files"]["result.txt"] = "old"
        action = {"name": "write_file", "arguments": {"path": "result.txt", "content": output}}
        plan = [{"name": "calculate", "arguments": {"expression": expression}}, action]
        goal["instruction"] = f"Calculate {expression} and save str(result) in result.txt."
    expected = copy.deepcopy(initial)
    if goal["operation"] == "record":
        expected["tables"]["users"]["u1"].update(goal["values"])
        permitted = {"record:users:u1": [expected["tables"]["users"]["u1"]]}
    else:
        expected["files"]["result.txt"] = action["arguments"]["content"]
        permitted = {"file:result.txt": [action["arguments"]["content"]]}
        if fault == "state_missing" and goal["operation"] in ("copy", "uppercase"):
            permitted["file:input.txt"] = [initial["files"]["input.txt"]]
    failed_action = copy.deepcopy(action)
    if fault == "parameter":
        key = "table" if goal["operation"] == "record" else "path"
        failed_action["arguments"]["invalid_" + key] = failed_action["arguments"].pop(key)
    return {"id": f"{family}-{fault}-{variant:04d}", "template_family": family,
            "split": SPLITS[family], "goal": goal, "initial_state": initial,
            "failed_action": failed_action, "fault_type": fault,
            "fault": {"type": fault, "seed": seed + variant, "step": 1},
            "expected_state": expected, "max_recovery_steps": 8,
            "recoverable": True, "allowed_mutations": permitted,
            "reference_plan": plan, "schema_version": "0.1"}

def generate_tasks(count=40, seed=42):
    if count < 1:
        raise ValueError("count must be positive")
    return [make_task(FAMILIES[(i // 4) % 10], FAULT_TYPES[i % 4],
                      i // 40, seed) for i in range(count)]

def save_tasks(path, tasks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n"
                           for t in tasks), encoding="utf-8")

def load_tasks(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
