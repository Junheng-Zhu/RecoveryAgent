import pytest
from recovery_agent.environment import Environment
from recovery_agent.models import ToolCall
from recovery_agent.dataset import generate_tasks

def call(env, name, **args):
    return env.execute(ToolCall(name, args))

def test_all_six_tools_and_reset():
    state = {"files": {"a.txt": "hello"}, "tables": {"users": {"a": {"n": 1}}}}
    with Environment(state) as env:
        assert call(env, "list_files").value == ["a.txt"]
        assert call(env, "read_file", path="a.txt").value == "hello"
        assert call(env, "write_file", path="nested/b.txt", content="world").ok
        assert call(env, "query_table", table="users", record_id="a").value == {"n": 1}
        assert call(env, "update_record", table="users", record_id="a", values={"n": 2}).ok
        assert call(env, "calculate", expression="(2 + 3) * 4 / 2").value == 10
        env.reset()
        assert env.snapshot() == state
        assert env.audit == [] and env.step == 0
        root = env.root
    assert not root.exists()
    assert state["tables"]["users"]["a"]["n"] == 1

@pytest.mark.parametrize("path", ["../evil", "/evil", "C:/evil", "a\\b", ""])
def test_path_escape_rejected(path):
    with Environment({"files": {}, "tables": {}}) as env:
        result = call(env, "write_file", path=path, content="x")
        assert not result.ok and not env.audit

@pytest.mark.parametrize("expression", ["__import__('os')", "1 / 0", "2 ** 100", "True", "1e309"])
def test_calculate_rejects_unsafe_inputs(expression):
    with Environment({"files": {}, "tables": {}}) as env:
        assert not call(env, "calculate", expression=expression).ok

def test_schema_errors_and_unknown_tools():
    with Environment({"files": {}, "tables": {}}) as env:
        for action in [ToolCall("bad", {}), ToolCall("read_file", {}),
                       ToolCall("write_file", {"path": "a", "content": 1}),
                       ToolCall("list_files", {"unexpected": "x"}),
                       ToolCall("update_record", {"table": "x", "record_id": "1", "values": {"x": float("nan")}})]:
            assert not env.execute(action).ok

def test_snapshots_and_reads_do_not_alias_database():
    task = generate_tasks()[16]
    with Environment(task["initial_state"]) as env:
        data = call(env, "query_table", table="users", record_id="u1").value
        data["name"] = "tampered"
        snapshot = env.snapshot()
        snapshot["tables"].clear()
        assert env.snapshot() == task["initial_state"]

def test_workspace_isolation():
    state = {"files": {"a": "x"}, "tables": {}}
    with Environment(state) as left, Environment(state) as right:
        assert left.root != right.root
        call(left, "write_file", path="a", content="changed")
        assert right.snapshot() == state

def test_directory_conflict_is_error_not_exception():
    with Environment({"files": {"a": "x"}, "tables": {}}) as env:
        assert not call(env, "write_file", path="a/b", content="y").ok
        call(env, "write_file", path="dir/file", content="x")
        assert not call(env, "write_file", path="dir", content="y").ok
