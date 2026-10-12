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

def prompt_for(task,events,remaining,feedback=None):
    # Exactly the same public prefix construction for collection, SFT and candidate comparison.
    prompt=build_prompt(task["goal"],events,remaining)
    prompt["tools"]=schemas()
    if feedback:
        prompt["messages"].append({"role":"user","content":canonical({"model_output_error":feedback})})
    return prompt

def schema_hash():
    return fingerprint(schemas())
