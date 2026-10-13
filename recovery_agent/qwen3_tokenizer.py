"""Official Qwen3 tokenization, causal final-decision supervision and ID binding checks."""
import copy
import hashlib
import json
import os
from pathlib import Path

class TokenizerUnavailable(RuntimeError):
    """A preparation blocker, never an invitation to invent a substitute template."""

class TemplateIncompatible(ValueError):
    """The supplied template cannot preserve verified causal tool conversations."""

def validate_messages(messages, tools):
    """Validate FIFO tool responses; preserve IDs externally because Qwen3 omits them on wire."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Sample must end with the supervised assistant decision")
    names = {t["function"]["name"] for t in tools}
    pending, seen, bindings = [], set(), []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported message role")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("Only text content is supported")
        if content and any(tag in content for tag in
                           ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
                            "<tool_response>", "</tool_response>", "<think>", "</think>")):
            raise ValueError("Unescaped template control token in message content")
        if role == "tool":
            if not pending:
                raise ValueError("Orphan tool result")
            binding = pending.pop(0)
            if (message.get("tool_call_id") != binding["call_id"] or
                message.get("name") != binding["name"]):
                raise ValueError("Tool result ID/name/order mismatch")
            binding["response_message_index"] = index
            continue
        if pending:
            raise ValueError("Missing tool results before next decision")
        calls = message.get("tool_calls") or []
        if calls and role != "assistant":
            raise ValueError("Tool calls must belong to assistant")
        for group_index, call in enumerate(calls):
            fn = call.get("function", {})
            call_id = call.get("id")
            if call.get("type") != "function" or not call_id or call_id in seen:
                raise ValueError("Invalid or reused call ID")
            if fn.get("name") not in names:
                raise ValueError("Unknown tool in supervised conversation")
            args = fn.get("arguments")
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be a JSON object")
            seen.add(call_id)
            binding = {"call_id": call_id, "name": fn["name"], "arguments": copy.deepcopy(args),
                       "assistant_message_index": index, "group_index": group_index,
                       "group_size": len(calls), "response_message_index": None}
            bindings.append(binding)
            pending.append(binding)
        if role == "assistant" and not calls and not (content and content.strip()):
            raise ValueError("Empty assistant decision")
    # Pending calls are legal only for the final target: their future results are excluded.
    if any(b["assistant_message_index"] != len(messages)-1 for b in pending):
        raise ValueError("Unresolved historical tool call")
    return bindings

def load_tokenizer(identifier, local_files_only=True, revision=None):
    """Use AutoTokenizer only; no AutoModel, tensors, credentials, or model weights."""
    os.environ.setdefault("USE_TORCH", "0")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(identifier), revision=revision, local_files_only=local_files_only,
            trust_remote_code=False, use_fast=True, token=False)
    except Exception as exc:
        raise TokenizerUnavailable(
            f"Official tokenizer unavailable ({type(exc).__name__}); audit/export remain usable") from exc
    if not tokenizer.chat_template:
        raise TemplateIncompatible("Tokenizer has no repository chat template")
    return tokenizer

class Qwen3Adapter:
    """Supervise the last verified decision, with exact token-prefix alignment and EOS checks."""
    def __init__(self, tokenizer, max_seq_length=8192, enable_thinking=False):
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if enable_thinking:
            raise ValueError("This readiness protocol uses enable_thinking=False")
        self.tokenizer, self.max_seq_length = tokenizer, max_seq_length
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise TemplateIncompatible("A single explicit repository chat template is required")
        self.template_hash = hashlib.sha256(template.encode()).hexdigest()
        self._probe()

    def _render(self, messages, tools, generation=False, tokenize=True):
        try:
            return self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=tokenize, add_generation_prompt=generation,
                enable_thinking=False, truncation=False)
        except Exception as exc:
            raise TemplateIncompatible(f"Repository template failed: {type(exc).__name__}") from exc

    def _probe(self):
        """Exercise tool grouping, responses, all roles and final answer on the actual template."""
        tools = [{"type": "function", "function": {"name": "probe_tool",
                 "description": "Template capability probe.", "parameters": {
                     "type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}}]
        calls = [{"id": f"probe-{i}", "type": "function",
                  "function": {"name": "probe_tool", "arguments": json.dumps({"x": i})}}
                 for i in (1, 2)]
        messages = [{"role": "system", "content": "PROBE_SYSTEM"},
                    {"role": "user", "content": "PROBE_USER"},
                    {"role": "assistant", "content": None, "tool_calls": calls},
                    *[{"role": "tool", "name": "probe_tool", "tool_call_id": f"probe-{i}",
                       "content": f"PROBE_RESPONSE_{i}"} for i in (1, 2)],
                    {"role": "assistant", "content": "PROBE_FINAL"}]
        validate_messages(messages, tools)
        text = self._render(messages, tools, tokenize=False)
        body = text.split("<|im_end|>", 1)[-1]
        if (body.count("<tool_call>") != 2 or body.count("<tool_response>") != 2 or
            not all(s in text for s in ("PROBE_SYSTEM", "PROBE_USER", "PROBE_FINAL",
                                        "PROBE_RESPONSE_1", "PROBE_RESPONSE_2", "probe_tool"))):
            raise TemplateIncompatible("Repository template lacks required tool-call/response support")
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        if (self.eos_id is None or self.start_id is None or self.eos_id == self.start_id or
            self.tokenizer.eos_token_id != self.eos_id):
            raise TemplateIncompatible("Expected Qwen3 role and EOS token IDs are unavailable")
        self.encode({"messages": messages, "tools": tools})
        self.encode({"messages": messages[:2]+[messages[2]], "tools": tools})
        self.capabilities = {"tool_call_groups": True, "tool_responses": True,
                             "final_answer": True, "enable_thinking": False,
                             "native_generation_mask": "{% generation" in self.tokenizer.chat_template,
                             "call_ids_rendered": all(c["id"] in text for c in calls),
                             "mask_method": "exact_generation_prefix_token_alignment",
                             "mask_scope": "final_verified_assistant_only",
                             "eos_token_id": self.eos_id, "template_sha256": self.template_hash}

    def encode(self, sample, enforce_limit=False):
        """Retain full long samples for statistics; explicitly fail when enforcing a context limit."""
        messages, tools = sample["messages"], sample["tools"]
        bindings = validate_messages(messages, tools)
        ids = list(self._render(messages, tools))
        prefix = list(self._render(messages[:-1], tools, generation=True))
        rendered = self._render(messages, tools, tokenize=False)
        independently_encoded = self.tokenizer.encode(rendered, add_special_tokens=False)
        if ids != independently_encoded or ids[:len(prefix)] != prefix:
            raise TemplateIncompatible("Generation prefix is not an exact token-aligned causal prefix")
        ends = [i for i in range(len(prefix), len(ids)) if ids[i] == self.eos_id]
        if len(ends) != 1 or self.start_id in ids[len(prefix):]:
            raise TemplateIncompatible("Target assistant role/EOS boundary is ambiguous")
        end = ends[0]+1
        if end <= len(prefix)+1:
            raise TemplateIncompatible("Target contains no supervised content tokens")
        labels = [-100]*len(ids)
        labels[len(prefix):end] = ids[len(prefix):end]
        long = len(ids) > self.max_seq_length
        if long and enforce_limit:
            raise ValueError(f"Sequence length {len(ids)} exceeds {self.max_seq_length}; truncation refused")
        return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels,
                "target_token_span": [len(prefix), end], "tool_bindings": bindings,
                "token_length": len(ids), "assistant_supervised_tokens": end-len(prefix),
                "over_max_seq_length": long, "truncated": False}

    def alignment_example(self, sample, encoded):
        """Expose each real token/label and rendered role boundary for manual review."""
        ids, labels = encoded["input_ids"], encoded["labels"]
        role, segments, start = "unframed", [], None
        rows = []
        target_start, target_end = encoded["target_token_span"]
        target_header = max(i for i in range(target_end) if ids[i] == self.start_id)
        for i, token in enumerate(ids):
            if token == self.start_id:
                if start is not None:
                    segments.append({"start": start, "end": i, "role": role})
                start = i
                following = self.tokenizer.decode(ids[i+1:i+8], skip_special_tokens=False)
                role = following.split("\n", 1)[0]
                if i != target_header and role == "assistant":
                    role = "assistant_history_masked"
                elif role == "assistant":
                    role = "assistant_target"
            rows.append({"index": i, "token_id": token,
                         "token": self.tokenizer.decode([token], skip_special_tokens=False),
                         "label": labels[i], "loss": labels[i] != -100, "rendered_role": role})
        if start is not None:
            segments.append({"start": start, "end": len(ids), "role": role})
        return {"sample_id": sample["sample_id"], "source": sample["source"],
                "task_id": sample["task_id"], "domain": sample["domain"],
                "messages": sample["messages"], "tool_bindings": encoded["tool_bindings"],
                "target_token_span": [target_start, target_end],
                "rendered_role_segments": segments,
                "note": "Qwen3 wraps tool results as user/tool_response; historical assistants are masked.",
                "rendered": self._render(sample["messages"], sample["tools"], tokenize=False),
                "tokens": rows}

def tokenizer_manifest(tokenizer, identifier):
    """Record template/file hashes and installed tokenizer versions for reproducible CPU encoding."""
    import importlib.metadata
    path = Path(identifier)
    files = {}
    if path.is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"):
            p = path/name
            if p.exists():
                files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    download = path/"download_manifest.json"
    model_config = path/"config.json"
    context = json.loads(model_config.read_text(encoding="utf-8")).get("max_position_embeddings") if model_config.exists() else None
    return {"identifier": str(identifier), "download": json.loads(download.read_text(encoding="utf-8"))
            if download.exists() else None, "file_sha256": files,
            "versions": {n: importlib.metadata.version(n) for n in ("transformers", "tokenizers", "huggingface_hub")},
            "class": type(tokenizer).__name__, "model_max_length": tokenizer.model_max_length,
            "model_config_max_position_embeddings": context}
