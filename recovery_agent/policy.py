import copy
from dataclasses import dataclass
from typing import Protocol
from .models import ToolCall, ToolResult

# 策略只能获得公开目标、失败动作、已观察结果及剩余预算，不包含期望状态或参考计划。
@dataclass(frozen=True)
class PolicyInput:
    goal: dict
    failed_action: ToolCall
    observations: tuple[ToolResult, ...]
    remaining_steps: int

class Policy(Protocol):
    # 策略接口：在每个新任务开始时清理策略自身的运行状态。
    def reset(self): ...
    # 策略接口：根据当前可见观察返回下一次 ToolCall，或返回 None 停止恢复。
    def act(self, observation: PolicyInput) -> ToolCall | None: ...

class StopPolicy:
    # 停止基线无跨步骤状态，保留统一接口以便 Runner 调用。
    def reset(self):
        pass
    # 立即停止，用于验证环境不会把未恢复的任务误判为成功。
    def act(self, observation):
        return None

class RetryPolicy:
    # 初始化重试策略的任务级状态；成功与否仍由最新工具结果判断。
    def reset(self):
        self.done = False
    # 没有成功结果时重复原失败调用，成功后停止；不主动修正参数或补齐依赖。
    def act(self, observation):
        if len(observation.observations) > 1 and observation.observations[-1].ok:
            return None
        return copy.deepcopy(observation.failed_action)

class RulePolicy:
    """Executable baseline: derives actions only from public goal and tool observations."""
    # 从资源发现阶段开始，并清空之前任务缓存的文件内容或数据库记录。
    def reset(self):
        self.phase = "discover"
        self.source_content = None

    # 使用本地阶段状态机处理发现、读取、缺失资源备份恢复和最终写入。
    # 动作只依据公开目标与已有工具结果产生，不读取 expected_state 或 Oracle 计划。
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

    # 根据任务类型构造最终更新：修改记录、复制/大写文本，或保存计算结果。
    # 文件输出依赖已经观察到的 source_content，不从环境内部直接提取答案。
    def _final(self, g):
        if g["operation"] == "record":
            return ToolCall("update_record", {"table": g["table"],
                "record_id": g["record_id"], "values": g["values"],
                "create": False})
        value = self.source_content
        if g["operation"] == "uppercase":
            value = value.upper()
        return ToolCall("write_file", {"path": g["target"], "content": str(value)})
