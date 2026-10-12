import copy
from .models import ToolCall

class Oracle:
    """Privileged reference-plan executor for certification and supervised targets only."""
    # 复制特权参考计划，避免执行指针或外部修改影响原始任务数据。
    def __init__(self, task):
        self.plan = copy.deepcopy(task["reference_plan"])
        self.reset()

    # 将参考计划执行位置归零，便于重复收集同一任务的轨迹。
    def reset(self):
        self.index = 0

    # 按顺序返回参考动作，计划耗尽时返回 None 表示停止。
    # Oracle 不调用模型；仅用于可恢复性验收和监督标签，不能作为普通评测输入。
    def act(self, observation):
        if self.index == len(self.plan):
            return None
        action = ToolCall(**self.plan[self.index])
        self.index += 1
        return action
