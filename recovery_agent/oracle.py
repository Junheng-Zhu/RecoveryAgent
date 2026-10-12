import copy
from .models import ToolCall

class Oracle:
    """Privileged reference-plan executor for certification and supervised targets only."""
    def __init__(self, task):
        self.plan = copy.deepcopy(task["reference_plan"])
        self.reset()

    def reset(self):
        self.index = 0

    def act(self, observation):
        if self.index == len(self.plan):
            return None
        action = ToolCall(**self.plan[self.index])
        self.index += 1
        return action
