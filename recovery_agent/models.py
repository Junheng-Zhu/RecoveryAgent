from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str = ""

@dataclass
class ErrorObservation:
    code: str
    message: str
    retryable: bool = False
    details: dict = field(default_factory=dict)

@dataclass
class ToolResult:
    call_id: str
    tool: str
    ok: bool
    value: Any = None
    error: ErrorObservation | None = None

    def to_dict(self):
        return asdict(self)

class ToolError(Exception):
    def __init__(self, code, message, retryable=False, **details):
        self.observation = ErrorObservation(code, message, retryable, details)
        super().__init__(message)
