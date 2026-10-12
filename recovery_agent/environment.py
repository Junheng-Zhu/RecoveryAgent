import copy
import tempfile
from pathlib import Path
from .models import ToolResult, ToolError
from .tools import validate, safe_path, calculate

class Environment:
    # 深拷贝初始文件与数据库状态，并为当前任务创建独立临时目录。
    # context 仅供环境故障逻辑使用，不等于允许策略读取全部内部状态。
    def __init__(self, initial_state, injector=None, context=None):
        self.initial = copy.deepcopy(initial_state)
        self.injector = injector
        self.context = copy.deepcopy(context or {})
        self._temp = None
        self.reset()

    # 清理旧目录后重新加载初始状态，同时清空步数、修改账本和前置条件。
    # 注入器也重置，确保同一任务再次执行时故障触发位置保持一致。
    def reset(self):
        self.close()
        self._temp = tempfile.TemporaryDirectory(prefix="recovery-agent-")
        self.root = Path(self._temp.name)
        self.tables = copy.deepcopy(self.initial.get("tables", {}))
        for name, content in self.initial.get("files", {}).items():
            path = self.path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.step, self.audit = 0, []
        self.needs_precondition = False
        if self.injector:
            self.injector.reset()

    # 释放本任务的临时目录；重复调用不会重复清理已释放资源。
    def close(self):
        if self._temp:
            self._temp.cleanup()
            self._temp = None

    # 允许通过 with 管理环境生命周期，返回当前环境实例。
    def __enter__(self):
        return self

    # 无论任务成功还是抛出异常，都在退出 with 时清理临时文件。
    def __exit__(self, *args):
        self.close()

    # 先检查相对路径格式，再检查解析后的路径是否仍位于工作目录内。
    # 返回的是本任务目录中的真实路径，不允许通过盘符或父目录访问外部文件。
    def path(self, name):
        path = self.root / safe_path(name)
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ToolError("INVALID_ARGUMENT", "Path escapes workspace")
        return path

    # 读取完整文件状态，并深拷贝内存表，避免观察者修改返回值污染环境。
    # 相对路径和稳定文件顺序使状态比较不依赖随机临时目录名称。
    def snapshot(self):
        return {"files": {p.relative_to(self.root).as_posix(): p.read_text(encoding="utf-8")
                          for p in sorted(self.root.rglob("*")) if p.is_file()},
                "tables": copy.deepcopy(self.tables)}

    # 统一执行入口：计步、注入故障、校验参数、调用工具、包装结果。
    # 只有成功的 read/query/calculate 会解除写入前置条件；ToolError 转为错误观察。
    def execute(self, call):
        self.step += 1
        try:
            # 故障注入先于参数检查；因此起始失败也可以带有稳定的注入诊断。
            if self.injector:
                self.injector.before(self, call, self.step)
            validate(call)
            value = self._dispatch(call.name, call.arguments)
            if call.name in ("read_file", "query_table", "calculate"):
                self.needs_precondition = False
            return ToolResult(call.call_id, call.name, True, value)
        except ToolError as e:
            return ToolResult(call.call_id, call.name, False, error=e.observation)

    # 实现六个模拟工具的实际行为；读操作返回快照，写操作记录 before/after 账本。
    # 规划顺序故障未解除时拒绝修改；即使重复写入相同内容，也保留写操作事件。
    def _dispatch(self, name, a):
        if self.needs_precondition and name in ("write_file", "update_record"):
            raise ToolError("PRECONDITION_REQUIRED", "Read/query/calculate before mutation")
        if name == "list_files":
            return sorted(p for p in self.snapshot()["files"]
                          if p.startswith(a.get("prefix", "")))
        if name == "read_file":
            path = self.path(a["path"])
            if not path.is_file():
                raise ToolError("NOT_FOUND", "File is missing", path=a["path"])
            return path.read_text(encoding="utf-8")
        if name == "write_file":
            path = self.path(a["path"])
            if path.exists() and not path.is_file():
                raise ToolError("INVALID_ARGUMENT", "Target is a directory")
            if any(p.is_file() for p in path.parents if p != self.root):
                raise ToolError("INVALID_ARGUMENT", "Parent is a file")
            path.parent.mkdir(parents=True, exist_ok=True)
            before = path.read_text(encoding="utf-8") if path.exists() else None
            path.write_text(a["content"], encoding="utf-8")
            # 每次写操作都入账，包括同值重复写；Verifier 会检查过程中发生的副作用。
            self.audit.append({"resource": "file:" + safe_path(a["path"]),
                               "before": before, "after": a["content"]})
            return {"written": a["path"]}
        if name in ("query_table", "update_record"):
            if a["table"] not in self.tables:
                raise ToolError("NOT_FOUND", "Table is missing", table=a["table"])
            table = self.tables[a["table"]]
            if name == "query_table":
                if "record_id" not in a:
                    return copy.deepcopy(table)
                if a["record_id"] not in table:
                    raise ToolError("NOT_FOUND", "Record is missing", record_id=a["record_id"])
                return copy.deepcopy(table[a["record_id"]])
            key = a["record_id"]
            if key not in table and not a.get("create", False):
                raise ToolError("NOT_FOUND", "Record is missing", record_id=key)
            before = copy.deepcopy(table.get(key))
            # 更新采用字段合并，create 只控制是否允许创建缺失行，不替换已有整行。
            table.setdefault(key, {}).update(copy.deepcopy(a["values"]))
            self.audit.append({"resource": f"record:{a['table']}:{key}",
                               "before": before, "after": copy.deepcopy(table[key])})
            return copy.deepcopy(table[key])
        return calculate(a["expression"])
