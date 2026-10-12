import hashlib
from .models import ToolError
from .tools import validate

FAULT_TYPES = ("parameter", "state_missing", "transient", "planning_order")

class FaultInjector:
    """One-shot faults at a specified 1-based tool step; stable across processes."""
    def __init__(self, fault_type, seed=0, step=1):
        if fault_type not in FAULT_TYPES or step < 1:
            raise ValueError("Invalid fault specification")
        self.fault_type, self.seed, self.step = fault_type, seed, step
        self.reset()

    def reset(self):
        self.fired = False

    def before(self, env, call, step):
        if self.fired or step != self.step:
            return
        self.fired = True
        token = hashlib.sha256(f"{self.seed}:{step}".encode()).hexdigest()[:12]
        if self.fault_type == "parameter":
            try:
                validate(call)
            except ToolError as error:
                error.observation.details["fault_token"] = token
                raise
            raise ToolError("INVALID_ARGUMENT", "Injected invalid parameter",
                            field=sorted(call.arguments)[0], fault_token=token)
        if self.fault_type == "transient":
            raise ToolError("TEMPORARY_UNAVAILABLE", "Try the operation again",
                            True, fault_token=token)
        if self.fault_type == "planning_order":
            env.needs_precondition = True
            raise ToolError("PRECONDITION_REQUIRED", "Read/query/calculate before mutation",
                            fault_token=token)
        goal = env.context
        if goal["operation"] in ("copy", "uppercase"):
            path = env.path(goal["source"])
            if path.exists():
                path.unlink()
            raise ToolError("NOT_FOUND", "Source file is missing",
                            path=goal["source"], fault_token=token)
        if goal["operation"] == "record":
            env.tables[goal["table"]].pop(goal["record_id"], None)
            raise ToolError("NOT_FOUND", "Record is missing",
                            record_id=goal["record_id"], fault_token=token)
        path = env.path(goal["target"])
        if path.exists():
            path.unlink()
        raise ToolError("NOT_FOUND", "Output state is missing",
                        path=goal["target"], fault_token=token)
