import copy
import tempfile
from pathlib import Path
from .models import ToolResult, ToolError
from .tools import validate, safe_path, calculate

class Environment:
    def __init__(self, initial_state, injector=None, context=None):
        self.initial = copy.deepcopy(initial_state)
        self.injector = injector
        self.context = copy.deepcopy(context or {})
        self._temp = None
        self.reset()

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

    def close(self):
        if self._temp:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self, name):
        path = self.root / safe_path(name)
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ToolError("INVALID_ARGUMENT", "Path escapes workspace")
        return path

    def snapshot(self):
        return {"files": {p.relative_to(self.root).as_posix(): p.read_text(encoding="utf-8")
                          for p in sorted(self.root.rglob("*")) if p.is_file()},
                "tables": copy.deepcopy(self.tables)}

    def execute(self, call):
        self.step += 1
        try:
            if self.injector:
                self.injector.before(self, call, self.step)
            validate(call)
            value = self._dispatch(call.name, call.arguments)
            if call.name in ("read_file", "query_table", "calculate"):
                self.needs_precondition = False
            return ToolResult(call.call_id, call.name, True, value)
        except ToolError as e:
            return ToolResult(call.call_id, call.name, False, error=e.observation)

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
            table.setdefault(key, {}).update(copy.deepcopy(a["values"]))
            self.audit.append({"resource": f"record:{a['table']}:{key}",
                               "before": before, "after": copy.deepcopy(table[key])})
            return copy.deepcopy(table[key])
        return calculate(a["expression"])
