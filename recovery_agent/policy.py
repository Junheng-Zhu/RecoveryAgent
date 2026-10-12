import copy
from dataclasses import dataclass
from typing import Protocol
from .models import ToolCall, ToolResult

@dataclass(frozen=True)
class PolicyInput:
    goal: dict
    failed_action: ToolCall
    observations: tuple[ToolResult, ...]
    remaining_steps: int

class Policy(Protocol):
    def reset(self): ...
    def act(self, observation: PolicyInput) -> ToolCall | None: ...

class StopPolicy:
    def reset(self):
        pass
    def act(self, observation):
        return None

class RetryPolicy:
    def reset(self):
        self.done = False
    def act(self, observation):
        if len(observation.observations) > 1 and observation.observations[-1].ok:
            return None
        return copy.deepcopy(observation.failed_action)

class RulePolicy:
    """Executable baseline: derives actions only from public goal and tool observations."""
    def reset(self):
        self.phase = "discover"
        self.source_content = None

    def act(self, obs):
        g, last = obs.goal, obs.observations[-1]
        if self.phase == "done":
            return None if last.ok else self._final(g)
        if self.phase == "discover":
            self.phase = "read"
            return ToolCall("list_files", {})
        if self.phase == "read":
            self.phase = "consume"
            if g["operation"] in ("copy", "uppercase"):
                return ToolCall("read_file", {"path": g["source"]})
            if g["operation"] == "record":
                return ToolCall("query_table", {"table": g["table"], "record_id": g["record_id"]})
            return ToolCall("calculate", {"expression": g["expression"]})
        if self.phase == "consume":
            if not last.ok:
                if last.error.code == "NOT_FOUND" and g["operation"] in ("copy", "uppercase", "record"):
                    self.phase = "backup"
                    if g["operation"] == "record":
                        return ToolCall("query_table", {"table": g["backup_table"], "record_id": g["record_id"]})
                    return ToolCall("read_file", {"path": g["backup"]})
                self.phase = "read"
                return self.act(obs)
            self.source_content = last.value
            self.phase = "done"
            return self._final(g)
        if self.phase == "backup":
            if not last.ok:
                return None
            self.source_content = last.value
            if g["operation"] == "record":
                self.phase = "done"
                return ToolCall("update_record", {"table": g["table"], "record_id": g["record_id"],
                    "values": {**last.value, **g["values"]}, "create": True})
            self.phase = "restored"
            return ToolCall("write_file", {"path": g["source"], "content": last.value})
        if self.phase == "restored":
            if not last.ok:
                return None
            self.phase = "read"
            return self.act(obs)

    def _final(self, g):
        if g["operation"] == "record":
            return ToolCall("update_record", {"table": g["table"],
                "record_id": g["record_id"], "values": g["values"],
                "create": False})
        value = self.source_content
        if g["operation"] == "uppercase":
            value = value.upper()
        return ToolCall("write_file", {"path": g["target"], "content": str(value)})
