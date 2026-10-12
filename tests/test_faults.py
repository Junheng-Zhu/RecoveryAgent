import pytest
from recovery_agent.dataset import generate_tasks
from recovery_agent.environment import Environment
from recovery_agent.faults import FaultInjector
from recovery_agent.models import ToolCall

# 将故障设在第 2 步，验证触发位置、Reset 后的错误结果和故障后状态一致。
@pytest.mark.parametrize("task", generate_tasks(), ids=lambda t: t["id"])
def test_fault_determinism_reset_and_one_shot(task):
    fault = task["fault"]
    with Environment(task["initial_state"], FaultInjector(fault["type"], fault["seed"], 2),
                     task["goal"]) as env:
        assert env.execute(ToolCall("list_files", {})).ok
        action = ToolCall(**task["failed_action"])
        first = env.execute(action)
        snapshot = env.snapshot()
        assert not first.ok
        assert env.injector.fired
        env.reset()
        env.execute(ToolCall("list_files", {}))
        assert env.execute(action).to_dict() == first.to_dict()
        assert env.snapshot() == snapshot

# 验证不同 seed 的诊断 token 不同，瞬时故障只触发一次，非法配置被拒绝。
def test_seed_changes_diagnostic_and_invalid_config():
    task = generate_tasks()[2]
    results = []
    for seed in (1, 2):
        with Environment(task["initial_state"], FaultInjector("transient", seed), task["goal"]) as env:
            results.append(env.execute(ToolCall(**task["failed_action"])).error.details["fault_token"])
            assert env.execute(ToolCall(**task["failed_action"])).ok
    assert results[0] != results[1]
    with pytest.raises(ValueError):
        FaultInjector("unknown")
    with pytest.raises(ValueError):
        FaultInjector("transient", step=0)
