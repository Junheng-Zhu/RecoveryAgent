import copy
from collections import Counter
from ..dataset import make_task
from ..faults import FAULT_TYPES
from .common import VERSION, fingerprint

# Conservative grouping: aliases and parameter variants share one mechanism family.
FAMILY_BASES = {
    "copy": ("copy_note", "copy_receipt"),
    "record_update": ("record_status", "record_quota"),
    "uppercase": ("upper_name", "upper_label"),
    "arithmetic_report": ("sum_invoice", "sum_stock", "ratio_report", "ratio_summary"),
}
DEFAULT_CONFIG = {
    "version": VERSION, "seed": 42,
    "counts": {"train": 800, "val": 200, "test": 200},
    "families": {"train": ["copy", "record_update"], "val": ["uppercase"],
                 "test": ["arithmetic_report"]},
    "environments": {
        "train": [{"id": "layout_a", "file_prefix": "workspace/", "table_prefix": "app_"},
                  {"id": "layout_b", "file_prefix": "documents/", "table_prefix": "live_"}],
        "val": [{"id": "layout_c", "file_prefix": "staging/", "table_prefix": "stage_"},
                {"id": "layout_d", "file_prefix": "inbox/", "table_prefix": "inbox_"}],
        "test": [{"id": "layout_e", "file_prefix": "archive/", "table_prefix": "archive_"},
                 {"id": "layout_f", "file_prefix": "reports/", "table_prefix": "report_"}],
    },
}

# 验证版本、seed、数量以及族/环境配置，拒绝跨 split 重用同一族或环境布局。
# 数量须为 4 的倍数，以保证四类故障均衡；这里不执行工具或读取任务答案。
def check_config(config):
    if config["version"] != VERSION or type(config["seed"]) is not int:
        raise ValueError("Unsupported version or seed")
    seen_families, seen_envs = set(), set()
    for split in ("train", "val", "test"):
        count = config["counts"][split]
        families, envs = config["families"][split], config["environments"][split]
        if type(count) is not int or count <= 0 or count % 4:
            raise ValueError("Split counts must be positive multiples of four")
        if not families or not envs or len(set(families)) != len(families):
            raise ValueError("Empty or duplicate family configuration")
        if set(families) & seen_families or not set(families) <= FAMILY_BASES.keys():
            raise ValueError("Template family overlap or unknown family")
        if {e["id"] for e in envs} & seen_envs or len({e["id"] for e in envs}) != len(envs):
            raise ValueError("Environment configuration overlap")
        for env in envs:
            if not env["file_prefix"].endswith("/") or not env["table_prefix"]:
                raise ValueError("Invalid namespace")
        seen_families.update(families)
        seen_envs.update(e["id"] for e in envs)
    layouts = [(e["file_prefix"], e["table_prefix"])
               for envs in config["environments"].values() for e in envs]
    if len(set(layouts)) != len(layouts):
        raise ValueError("Duplicate environment layout")

# 将 v0.1 任务映射到实例独立的文件目录和记录 ID，并同步更新目标、计划和验收资源。
# 操作深拷贝后的任务，避免不同实例共享可变状态；环境配置保留用于切分审计。
def namespace_task(task, env, variant=0):
    task = copy.deepcopy(task)
    file_map = {name: env["file_prefix"] + f"case_{variant:04d}/" + name for name in task["initial_state"]["files"]}
    file_map.setdefault("result.txt", env["file_prefix"] + f"case_{variant:04d}/result.txt")
    table_map = {name: env["table_prefix"] + name for name in task["initial_state"]["tables"]}
    for key in ("initial_state", "expected_state"):
        state = task[key]
        state["files"] = {file_map[name]: value for name, value in state["files"].items()}
        state["tables"] = {table_map[name]: value for name, value in state["tables"].items()}
    g = task["goal"]
    # 记录 ID、备份行、计划参数和许可资源需要一起改名，否则验收会指向旧资源。
    if "record_id" in g:
        old_id, new_id = g["record_id"], f"r_{variant:04d}"
        g["record_id"] = new_id
        g["instruction"] = g["instruction"].replace("/" + old_id, "/" + new_id)
        for state_key in ("initial_state", "expected_state"):
            for rows in task[state_key]["tables"].values():
                rows[new_id] = rows.pop(old_id)
        for action in [task["failed_action"], *task["reference_plan"]]:
            if "record_id" in action["arguments"]:
                action["arguments"]["record_id"] = new_id
        task["allowed_mutations"] = {
            resource.rsplit(":", 1)[0] + ":" + new_id: values
            for resource, values in task["allowed_mutations"].items()}
    for key in ("target", "source", "backup"):
        if key in g:
            g[key] = file_map[g[key]]
    for key in ("table", "backup_table"):
        if key in g:
            g[key] = table_map[g[key]]
    # Replace the longest names first; no hidden answer is added to the request.
    for old, new in sorted({**file_map, **table_map}.items(), key=lambda item: -len(item[0])):
        import re
        g["instruction"] = re.sub(r"(?<![\w/])" + re.escape(old) + r"(?![\w])",
                                  lambda _: new, g["instruction"])
    for action in [task["failed_action"], *task["reference_plan"]]:
        a = action["arguments"]
        for key in ("path", "invalid_path"):
            if key in a:
                a[key] = file_map[a[key]]
        for key in ("table", "invalid_table"):
            if key in a:
                a[key] = table_map[a[key]]
    permitted = {}
    for resource, values in task["allowed_mutations"].items():
        parts = resource.split(":")
        if parts[0] == "file":
            resource = "file:" + file_map[parts[1]]
        else:
            resource = f"record:{table_map[parts[1]]}:{parts[2]}"
        permitted[resource] = values
    task["allowed_mutations"] = permitted
    task["environment_config"] = copy.deepcopy(env)
    return task

# 复用 v0.1 make_task，按固定配置生成 train/val/test 并附加 v0.2 版本元数据。
# 先按恢复机制归组模板别名，再生成变体，避免同一机制的别名跨 split 泄漏。
def generate_dataset(config=None):
    config = copy.deepcopy(config or DEFAULT_CONFIG)
    check_config(config)
    tasks = []
    for split in ("train", "val", "test"):
        families, envs = config["families"][split], config["environments"][split]
        for i in range(config["counts"][split]):
            block, fault_index = divmod(i, 4)
            family = families[block % len(families)]
            bases = FAMILY_BASES[family]
            variant = block // len(families)
            base = bases[variant % len(bases)]
            # Environment cycling is independent of family and fault.
            # 环境配置按实例循环，与当前故障类型无关，避免把布局当作故障提示。
            env = envs[(block // len(families)) % len(envs)]
            task = namespace_task(make_task(base, FAULT_TYPES[fault_index], variant,
                                            config["seed"]), env, variant)
            task.update(id=f"v02-{family}-{env['id']}-{FAULT_TYPES[fault_index]}-{variant:04d}",
                        template_family=family, base_template=base, split=split,
                        schema_version=VERSION, dataset_version=VERSION)
            tasks.append(task)
    return tasks, config

# 分别统计各 split 的任务数、故障、模板族、环境布局和参考计划长度。
# 参考计划长度是数据审计信息，不会作为模型 Prompt 输入。
def task_statistics(tasks):
    result = {}
    for split in ("train", "val", "test"):
        rows = [t for t in tasks if t["split"] == split]
        result[split] = {
            "count": len(rows),
            "fault_types": dict(sorted(Counter(t["fault_type"] for t in rows).items())),
            "template_families": dict(sorted(Counter(t["template_family"] for t in rows).items())),
            "environments": dict(sorted(Counter(t["environment_config"]["id"] for t in rows).items())),
            "reference_plan_lengths": dict(sorted(Counter(len(t["reference_plan"]) for t in rows).items())),
        }
    return result
