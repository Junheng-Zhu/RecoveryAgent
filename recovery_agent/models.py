from dataclasses import dataclass, field, asdict
from typing import Any

# 统一的工具调用请求：name 用于路由，arguments 保存参数，call_id 用于关联执行结果。
@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str = ""

# 可供策略观察的结构化错误；retryable 是提示，不能代替策略对副作用的判断。
@dataclass
class ErrorObservation:
    code: str
    message: str
    retryable: bool = False
    details: dict = field(default_factory=dict)

# 工具执行返回值；成功时读取 value，失败时读取 error，并保留请求的 call_id。
@dataclass
class ToolResult:
    call_id: str
    tool: str
    ok: bool
    value: Any = None
    error: ErrorObservation | None = None

    # 递归转换 dataclass 为普通字典，供轨迹序列化、结果比较和 Replay 使用。
    def to_dict(self):
        return asdict(self)

class ToolError(Exception):
    # 将工具层异常包装为 ErrorObservation，由 Environment 捕获后返回统一失败结果。
    def __init__(self, code, message, retryable=False, **details):
        self.observation = ErrorObservation(code, message, retryable, details)
        super().__init__(message)
