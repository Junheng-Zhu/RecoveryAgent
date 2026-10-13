import copy
from ..data_pipeline.serialization import build_prompt
from ..data_pipeline.common import canonical, fingerprint
from ..stateful.protocol import schemas as original_schemas

def schemas():
    # One narrowly scoped branch-switch operation; no arbitrary Git revision/expression execution.
    tools=original_schemas()
    tools.append({"type":"function","function":{"name":"git_switch","description":"Switch only to a permitted existing branch inside this task's isolated repository.",
       "parameters":{"type":"object","properties":{"branch":{"type":"string"}},"required":["branch"],"additionalProperties":False}}})
    return tools

STOP_PROTOCOLS=("legacy","termination-v0.3c-1","termination-v0.3c-2")
STOP_INSTRUCTION=(
    "Termination protocol v0.3-C.1: Once the user's goal is satisfied, stop calling tools "
    "and return a concise completion message. Do not repeat state queries merely to reconfirm "
    "a result already established by visible tool responses. If the goal is incomplete or "
    "necessary uncertainty remains, perform reasonable diagnostics before deciding whether to finish."
)

STOP_INSTRUCTION_C2=(
    "Termination protocol v0.3-C.2: Use the user's goal and observed tool results to decide "
    "whether a write operation is still necessary. Avoid repeating Git staging, file writes, "
    "database updates or commits when the intended resource state is already established. "
    "Complete every explicit requirement and necessary dependency, then immediately stop tools "
    "and return a concise completion message. One successful tool call does not establish that "
    "the entire task is complete. Allow necessary diagnostics while resolving real uncertainty, "
    "but do not add unrelated state queries, completion artifacts or inferred extra user requirements."
)

def prompt_for(task,events,remaining,feedback=None,stop_protocol="legacy"):
    # Exactly the same public prefix construction for collection, SFT and candidate comparison.
    if stop_protocol not in STOP_PROTOCOLS:
        raise ValueError("Unknown stop protocol")
    prompt=build_prompt(task["goal"],events,remaining)
    if stop_protocol!="legacy":
        prompt["messages"][0]["content"]+="\n"+(STOP_INSTRUCTION if stop_protocol=="termination-v0.3c-1" else STOP_INSTRUCTION_C2)
    prompt["tools"]=schemas()
    if feedback:
        prompt["messages"].append({"role":"user","content":canonical({"model_output_error":feedback})})
    return prompt

def schema_hash():
    return fingerprint(schemas())
