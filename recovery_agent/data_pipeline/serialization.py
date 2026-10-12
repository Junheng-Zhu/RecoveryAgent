import copy
import json
from ..tools import SCHEMAS
from ..models import ToolCall
from .common import canonical, fingerprint

GOAL_KEYS = {"instruction", "operation", "target", "source", "backup",
             "table", "record_id", "values", "backup_table", "expression"}
DETAIL_KEYS = {"path", "record_id", "table", "field", "missing", "extra"}
SYSTEM = ("Recover the user's task using only the available tools and observed history. "
          "Do not modify unrelated resources. Call one tool at a time. "
          "When finished, respond with: Recovery complete.")

# 从运行时 SCHEMAS 构建标准 function JSON Schema，确保训练披露字段与工具校验一致。
def tool_schemas():
    descriptions = {
        "list_files": "List workspace files, optionally filtered by a relative path prefix.",
        "read_file": "Read a UTF-8 workspace file.",
        "write_file": "Write a UTF-8 workspace file; creates parent directories.",
        "query_table": "Read a table or one record from the in-memory database.",
        "update_record": "Merge fields into a record; create=true permits creating a missing row.",
        "calculate": "Evaluate numeric literals with +, -, *, / and parentheses.",
    }
    tools = []
    for name, (required, optional) in SCHEMAS.items():
        properties = {key: {"type": {"values": "object", "create": "boolean"}.get(key, "string")}
                      for key in sorted(required | optional)}
        tools.append({"type": "function", "function": {"name": name,
                      "description": descriptions[name], "parameters": {
                          "type": "object", "properties": properties,
                          "required": sorted(required), "additionalProperties": False}}})
    return tools

# 只保留明确允许公开的目标字段，并深拷贝以防策略修改原任务。
def public_goal(goal):
    return copy.deepcopy({key: value for key, value in goal.items() if key in GOAL_KEYS})

# 保留真实工具返回值与公开错误说明；错误 details 采用白名单过滤隐藏 seed/token 等信息。
def public_result(result):
    error = result["error"]
    if error:
        error = {"code": error["code"], "message": error["message"],
                 "retryable": error["retryable"],
                 "details": {k: copy.deepcopy(v) for k, v in error["details"].items()
                             if k in DETAIL_KEYS}}
    return {"ok": result["ok"], "value": copy.deepcopy(result["value"]), "error": error}

# 将调用转换为 assistant tool_calls 消息；arguments 使用规范化 JSON 字符串。
def action_message(call):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call["call_id"], "type": "function",
        "function": {"name": call["name"], "arguments": canonical(call["arguments"])}}]}

# 统一终止动作的监督与在线表示，避免训练目标和推理停止规则不一致。
def stop_message():
    return {"role": "assistant", "content": "Recovery complete."}

# 按时间顺序配对已执行调用与结果，tool_call_id 保证返回值与请求正确关联。
# 传入事件必须是当前决策之前的历史，不应包含未来执行结果。
def visible_history(events):
    history = []
    for event in events:
        history.append(action_message(event["call"]))
        history.append({"role": "tool", "name": event["call"]["name"],
                        "tool_call_id": event["call"]["call_id"],
                        "content": canonical(public_result(event["result"]))})
    return history

# 训练和在线推理共用的唯一 Prompt 入口：指令、公开目标、可见历史、剩余预算及工具定义。
def build_prompt(goal, events, remaining_steps):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": canonical(public_goal(goal))}]
    messages += visible_history(events)
    messages.append({"role": "user", "content": canonical({"remaining_recovery_steps": remaining_steps})})
    return {"messages": messages, "tools": tool_schemas()}

# 为 messages 与 tools 的完整公开内容生成指纹，用于重复、切分和偏好状态核对。
def prompt_id(prompt):
    return fingerprint(prompt)

# 解析单个工具调用或固定停止消息；strict=True 时额外执行工具参数校验。
# 偏好分支使用 strict=False，让错误参数候选在实际环境中产生错误观察。
def parse_action(message, strict=True):
    if message.get("role") != "assistant":
        raise ValueError("Expected assistant output")
    calls = message.get("tool_calls", [])
    if not calls:
        if message != stop_message():
            raise ValueError("Invalid stop output")
        return None
    if len(calls) != 1 or message.get("content") is not None:
        raise ValueError("Exactly one tool call required")
    call = calls[0]
    if call.get("type") != "function":
        raise ValueError("Invalid call type")
    args = json.loads(call["function"]["arguments"])
    from ..tools import validate
    result = ToolCall(call["function"]["name"], args, call["id"])
    if strict:
        validate(result)
    return result

class ConversationalPolicy:
    """Adapter for a future predictor; identical prompt builder to SFT, no model included."""
    # 接收可替换 predictor 回调；本适配器仅负责协议衔接，不加载或调用具体模型。
    def __init__(self, predictor):
        self.predictor = predictor
        self.reset()

    # 清空事件历史、待匹配动作及已消费结果计数，防止任务间上下文污染。
    def reset(self):
        self.events = []
        self.pending = None
        self.seen = 0

    # 将新结果与失败动作或上次预测动作配对，使用同一 Prompt 构造器请求下一动作。
    # 只支持从失败动作开始的 episode；无法从未知 prefix_actions 重建完整历史参数。
    def act(self, observation):
        from dataclasses import asdict
        fresh = observation.observations[self.seen:]
        if not self.events:
            if len(fresh) != 1:
                raise ValueError("Adapter expects an episode starting at its failed action")
            self.events.append({"call": asdict(observation.failed_action),
                                "result": fresh[0].to_dict()})
        elif fresh:
            if len(fresh) != 1 or self.pending is None:
                raise ValueError("Unmatched tool result")
            call = copy.deepcopy(self.pending)
            call["call_id"] = fresh[0].call_id
            self.events.append({"call": call, "result": fresh[0].to_dict()})
        self.seen = len(observation.observations)
        prompt = build_prompt(observation.goal, self.events, observation.remaining_steps)
        action = parse_action(self.predictor(copy.deepcopy(prompt)))
        self.pending = asdict(action) if action else None
        return action
