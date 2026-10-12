"""Isolated Git/SQLite extension; legacy tools and interfaces remain intact."""
import copy
import os
import sqlite3
import subprocess
from pathlib import Path
from ..environment import Environment
from ..models import ToolCall, ToolResult, ToolError
from ..tools import safe_path, validate

EXTRA_SCHEMAS = {
    "git_status": (set(), set()), "git_diff": (set(), {"path"}),
    "git_add": ({"path"}, set()),
    "git_commit": ({"message"}, {"expected_head"}),
    "git_checkout": ({"path"}, {"source", "expected_content"}),
    "execute_select": ({"table"}, {"record_id"}),
    "insert_record": ({"table", "record_id", "values"}, set()),
    "delete_record": ({"table", "record_id"}, set()),
}
DB_COLUMNS = {"accounts": {"email", "balance"}, "entries": {"account_id", "amount"}}

def validate_extra(call):
    # Only whitelist names, fields and typed values; identifiers never enter arbitrary SQL.
    if call.name not in EXTRA_SCHEMAS:
        validate(call)
        return
    required, optional = EXTRA_SCHEMAS[call.name]
    a = call.arguments
    if not isinstance(a, dict) or required - a.keys() or a.keys() - required - optional:
        raise ToolError("INVALID_ARGUMENT", "Invalid tool arguments")
    for key, value in a.items():
        if not isinstance(value, dict if key == "values" else str):
            raise ToolError("INVALID_ARGUMENT", "Wrong argument type", field=key)

class StatefulFaultInjector:
    # Native faults are exercised by the failed call; transient faults are pre-side-effect.
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.reset()

    def reset(self):
        self.fired = False

    def before(self, env, call, step):
        if self.fired or step != self.config.get("step", 1):
            return
        self.fired = True
        if self.config["type"] == "transient":
            raise ToolError("TEMPORARY_UNAVAILABLE", "Transient service failure", True)

class StatefulEnvironment(Environment):
    # Base reset builds the filesystem; Git and SQLite live only inside this instance.
    def __init__(self, task):
        self.task = copy.deepcopy(task)
        self.db = None
        self.git_ready = False
        super().__init__(task["initial_state"], StatefulFaultInjector(task["fault"]), task["goal"])

    def path(self, name):
        path = safe_path(name)
        if any(p.lower() == ".git" for p in Path(path).parts):
            raise ToolError("UNSAFE_ACTION", "Git metadata is not a tool-accessible path")
        return super().path(path)

    def reset(self):
        self.git_ready = False
        super().reset()
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(
            "CREATE TABLE accounts(id TEXT PRIMARY KEY,email TEXT NOT NULL UNIQUE,"
            "balance INTEGER NOT NULL CHECK(balance>=0));"
            "CREATE TABLE entries(id TEXT PRIMARY KEY,account_id TEXT NOT NULL REFERENCES accounts(id),"
            "amount INTEGER NOT NULL CHECK(amount>0));")
        for table, rows in self.task.get("sqlite_initial", {}).items():
            for key, row in rows.items():
                columns = list(row)
                self.db.execute(f"INSERT INTO {table}(id,{','.join(columns)}) VALUES "
                                f"({','.join('?' for _ in range(len(columns)+1))})", [key, *row.values()])
        self.db.commit()
        if self.task["domain"] == "git":
            self._git(["init", "-b", "main"])
            self._git(["config", "user.name", "RecoveryAgent"])
            self._git(["config", "user.email", "recovery@example.invalid"])
            self._git(["config", "core.autocrlf", "false"])
            self._git(["config", "core.hooksPath", os.devnull])
            self._git(["add", "--", "."])
            self._git(["commit", "-m", "baseline"], commit=True)
            self.git_ready = True
            for path, content in self.task.get("dirty_files", {}).items():
                self.path(path).write_text(content, encoding="utf-8")
            for path in self.task.get("staged_paths", []):
                self._git(["add", "--", safe_path(path)])
        self.audit = []

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        super().close()

    def _git(self, args, commit=False, allow_failure=False):
        # Argument vectors, private cwd, fixed identity/time and disabled hooks prevent host effects.
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_AUTHOR_NAME="RecoveryAgent", GIT_AUTHOR_EMAIL="recovery@example.invalid",
                   GIT_COMMITTER_NAME="RecoveryAgent", GIT_COMMITTER_EMAIL="recovery@example.invalid",
                   GIT_AUTHOR_DATE="2026-01-01T00:00:00+0000", GIT_COMMITTER_DATE="2026-01-01T00:00:00+0000")
        result = subprocess.run(["git", *args], cwd=self.root, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=10)
        if result.returncode and not allow_failure:
            raise ToolError("GIT_ERROR", (result.stderr or result.stdout).strip()[:1000])
        return result.stdout.strip()

    def _git_state(self):
        files = self._git(["ls-files"]).splitlines()
        staged = {p: self._git(["show", ":" + p]) for p in files}
        tree = {p: self._git(["show", "HEAD:" + p], allow_failure=True) for p in files}
        return {"head": self._git(["rev-parse", "HEAD"]),
                "message": self._git(["log", "-1", "--format=%s"]),
                "commit_count": int(self._git(["rev-list", "--count", "HEAD"])),
                "status": self._git(["status", "--porcelain"]),
                "index": staged, "tree": tree}

    def snapshot(self):
        # Exclude nondeterministic .git internals; index, HEAD and commit tree are semantic state.
        files = {p.relative_to(self.root).as_posix(): p.read_text(encoding="utf-8")
                 for p in sorted(self.root.rglob("*"))
                 if p.is_file() and ".git" not in p.relative_to(self.root).parts}
        sql = {t: {r["id"]: {k: r[k] for k in r.keys() if k != "id"}
                   for r in self.db.execute(f"SELECT * FROM {t} ORDER BY id")}
               for t in DB_COLUMNS} if self.db else {}
        return {"files": files, "tables": copy.deepcopy(self.tables), "sqlite": sql,
                "git": self._git_state() if self.git_ready else None}

    def execute(self, call):
        if call.name not in EXTRA_SCHEMAS and not (call.name == "update_record" and
                isinstance(call.arguments, dict) and call.arguments.get("table") in DB_COLUMNS):
            return super().execute(call)
        self.step += 1
        try:
            self.injector.before(self, call, self.step)
            validate_extra(call)
            value = self._stateful_dispatch(call.name, call.arguments)
            return ToolResult(call.call_id, call.name, True, value)
        except ToolError as e:
            return ToolResult(call.call_id, call.name, False, error=e.observation)
        except (OSError, subprocess.SubprocessError) as e:
            return ToolResult(call.call_id, call.name, False,
                              error=ToolError("EXECUTION_ERROR", type(e).__name__).observation)

    def _stateful_dispatch(self, name, a):
        if name.startswith("git_"):
            if not self.git_ready:
                raise ToolError("NOT_FOUND", "No Git repository in this task")
            before = self._git_state()
            if name == "git_status":
                return before
            if name == "git_diff":
                args = ["diff", "HEAD"]
                if "path" in a:
                    self.path(a["path"])
                    args += ["--", safe_path(a["path"])]
                return self._git(args)
            if name in ("git_add", "git_checkout"):
                self.path(a["path"])
                if a["path"] not in self.task["git_paths"]:
                    raise ToolError("UNSAFE_ACTION", "Path is outside permitted Git task scope")
            if name == "git_add":
                self._git(["add", "--", safe_path(a["path"])])
                after = self._git_state()
                self.audit.append({"resource": "git:index", "before": before["index"], "after": after["index"]})
                return after
            if name == "git_commit":
                if a.get("expected_head", before["head"]) != before["head"]:
                    raise ToolError("REVISION_CONFLICT", "HEAD changed; inspect git_status")
                if not self._git(["diff", "--cached", "--name-only"]):
                    raise ToolError("NO_CHANGES", "Nothing staged to commit")
                self._git(["commit", "-m", a["message"]], commit=True)
                after = self._git_state()
                self.audit.append({"resource": "git:head", "before": before, "after": after})
                return after
            source = a.get("source", "head")
            if source not in ("head", "index"):
                raise ToolError("INVALID_ARGUMENT", "source must be head or index")
            path = self.path(a["path"])
            old = path.read_text(encoding="utf-8") if path.exists() else None
            if "expected_content" not in a or a["expected_content"] != old:
                raise ToolError("FILE_CONFLICT", "Read current file; provide matching expected_content")
            args = ["restore", "--worktree"]
            if source == "head":
                args += ["--source=HEAD"]
            self._git([*args, "--", safe_path(a["path"])])
            new = path.read_text(encoding="utf-8")
            self.audit.append({"resource": "file:" + a["path"], "before": old, "after": new})
            return {"restored": a["path"]}
        table = a["table"]
        if table not in DB_COLUMNS:
            raise ToolError("UNSAFE_ACTION", "Table is not allowlisted")
        key = a.get("record_id")
        old = self.db.execute(f"SELECT * FROM {table} WHERE id=?", (key,)).fetchone() if key else None
        if name == "execute_select":
            rows = self.db.execute(f"SELECT * FROM {table}" + (" WHERE id=?" if key else "") +
                                   " ORDER BY id", (key,) if key else ())
            return [dict(r) for r in rows]
        if a.get("create"):
            raise ToolError("INVALID_ARGUMENT", "SQLite update does not support upsert; use insert_record")
        values = a.get("values", {})
        if values.keys() - DB_COLUMNS[table] or not values and name != "delete_record":
            raise ToolError("INVALID_ARGUMENT", "Unknown or empty column update")
        for field, value in values.items():
            expected = str if field in ("email", "account_id") else int
            if type(value) is not expected:
                raise ToolError("INVALID_ARGUMENT", "Wrong column type", field=field)
        if name in ("update_record", "delete_record") and old is None:
            raise ToolError("NOT_FOUND", "Record missing", record_id=key)
        try:
            with self.db:
                if name == "insert_record":
                    columns = list(values)
                    self.db.execute(f"INSERT INTO {table}(id,{','.join(columns)}) VALUES "
                                    f"({','.join('?' for _ in range(len(columns)+1))})", [key, *values.values()])
                elif name == "update_record":
                    self.db.execute(f"UPDATE {table} SET " + ",".join(f"{c}=?" for c in values) +
                                    " WHERE id=?", [*values.values(), key])
                else:
                    self.db.execute(f"DELETE FROM {table} WHERE id=?", (key,))
        except sqlite3.IntegrityError as error:
            code = ("STATE_DEPENDENCY" if "FOREIGN KEY" in str(error) else
                    "TRANSACTION_FAILED" if "CHECK" in str(error) else "CONSTRAINT_CONFLICT")
            raise ToolError(code, str(error), rolled_back=True)
        new = self.db.execute(f"SELECT * FROM {table} WHERE id=?", (key,)).fetchone()
        def content(row):
            return {k: row[k] for k in row.keys() if k != "id"} if row else None
        self.audit.append({"resource": f"sqlite:{table}:{key}",
                           "before": content(old), "after": content(new)})
        return dict(new) if new else {"deleted": key}
