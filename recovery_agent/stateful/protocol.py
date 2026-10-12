import copy
import json
from ..models import ToolCall
from ..data_pipeline.common import canonical, fingerprint
from ..data_pipeline.serialization import build_prompt, tool_schemas, action_message, stop_message
from .environment import EXTRA_SCHEMAS

# Disclose extended tools while keeping the legacy protocol and schema source.
def schemas():
    # Extend the frozen v0.2 schema without altering the six existing function definitions.
    result = tool_schemas()
    descriptions = {
        "git_status": "Inspect isolated Git HEAD, index and working tree.",
        "git_diff": "Inspect changes against HEAD, optionally for one relative path.",
        "git_add": "Stage exactly one permitted relative path.",
        "git_commit": "Commit staged changes; optional expected_head implements optimistic concurrency.",
        "git_checkout": "Restore one permitted path from head/index; expected_content must match its current file content.",
        "execute_select": "Read only allowlisted SQLite accounts/entries rows; no raw SQL.",
        "insert_record": "Insert SQLite row. accounts: email(string),balance(integer >=0); entries: account_id(string FK),amount(integer >0).",
        "delete_record": "Delete one SQLite row by ID in accounts/entries.",
    }
    for name, (required, optional) in EXTRA_SCHEMAS.items():
        result.append({"type": "function", "function": {"name": name, "description": descriptions[name],
            "parameters": {"type": "object", "properties": {
                k: {"type": "object" if k == "values" else "string"} for k in sorted(required | optional)},
                "required": sorted(required), "additionalProperties": False}}})
    for tool in result:
        if tool["function"]["name"] == "update_record":
            tool["function"]["description"] += (
                " SQLite accounts/entries also use this signature; existing rows only, no create/upsert.")
    return result

def prompt_for(task, events, remaining, feedback=None):
    # Shared v0.2 causal history builder; only tool disclosure changes for the new environment.
    prompt = build_prompt(task["goal"], events, remaining)
    prompt["tools"] = schemas()
    if feedback:
        prompt["messages"].append({"role": "user", "content": canonical({"model_output_error": feedback})})
    return prompt

def parse_response(response):
    # Never rewrite malformed model calls into reference actions; report each parsing defect.
    if not isinstance(response, dict):
        return {"kind": "invalid", "error": "invalid_response_shape", "calls": []}
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict) or not isinstance(choices[0].get("message"), dict):
        return {"kind": "invalid", "error": "empty_response", "calls": []}
    choice, message = choices[0], choices[0]["message"]
    if choice.get("finish_reason") == "length":
        return {"kind": "invalid", "error": "truncated_output", "calls": []}
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        return {"kind": "invalid", "error": "invalid_tool_calls_shape", "calls": []}
    if calls:
        decoded, ids = [], set()
        for item in calls:
            try:
                fn = item["function"]
                if item["type"] != "function" or not isinstance(item["id"], str) or not item["id"] or item["id"] in ids:
                    raise ValueError("invalid_or_duplicate_id")
                ids.add(item["id"])
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                if not isinstance(fn["name"], str) or not isinstance(args, dict):
                    raise ValueError("arguments_must_be_object")
                decoded.append({"call": {"name": fn["name"], "arguments": args, "call_id": item["id"]}, "error": None})
            except (KeyError, TypeError, ValueError):
                decoded.append({"call": None, "error": "malformed_tool_call"})
        return {"kind": "tools", "calls": decoded, "error": None}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return {"kind": "stop", "calls": [], "error": None, "text": content}
    return {"kind": "invalid", "calls": [], "error": "empty_assistant_message"}
