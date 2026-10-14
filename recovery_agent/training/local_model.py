"""Local Qwen3/Hermes inference adapter; tools execute only through the existing Runtime."""
import copy
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from ..paths import project_path
from ..gpu_workspace import gpu_environment
from ..stateful.predictor import BudgetExceeded
from ..data_pipeline.common import fingerprint
from .data import json_file, tokenizer_identity

@dataclass
class LocalConfig:
    model: str = "Qwen/Qwen3-1.7B"
    model_mode: str = "base"
    adapter: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    max_requests: int = 16
    max_total_tokens: int = 32768
    seed: int = 42
    dtype: str = "bf16"

    def __post_init__(self):
        if (self.model_mode not in ("base", "adapter") or
            self.model_mode == "base" and self.adapter is not None or
            self.model_mode == "adapter" and not self.adapter):
            raise ValueError("Base/Adapter mode and adapter path must agree")
        if (min(self.max_tokens, self.max_requests, self.max_total_tokens) <= 0 or
            not 0 <= self.temperature < 2 or self.dtype not in ("bf16", "float32")):
            raise ValueError("Invalid local model limits/dtype")

    def public(self):
        return asdict(self)

def hermes_message(text, request_index):
    """Strictly parse emitted tags/JSON; never repair arguments or replace actions with Oracle."""
    opens, closes = text.count("<tool_call>"), text.count("</tool_call>")
    if opens or closes:
        matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S))
        if not opens or opens != closes or len(matches) != opens:
            raise ValueError("Unbalanced tool_call tags")
        calls = []
        for i, match in enumerate(matches):
            obj = json.loads(match.group(1))
            if not isinstance(obj, dict) or set(obj)-{"name", "arguments"} or not {"name", "arguments"} <= set(obj):
                raise ValueError("Tool JSON must contain only name/arguments")
            if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], (dict, str)):
                raise ValueError("Invalid tool name/argument shape")
            # Preserve a model-supplied JSON string verbatim; the Runtime parser checks validity.
            args = obj["arguments"] if isinstance(obj["arguments"], str) else json.dumps(obj["arguments"], ensure_ascii=False)
            calls.append({"id": f"local-{request_index}-{i}", "type": "function",
                          "function": {"name": obj["name"], "arguments": args}})
        content = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", text, flags=re.S).strip()
        if "<think>" in content or "</think>" in content:
            raise ValueError("Unexpected reasoning tags in non-thinking mode")
        return {"role": "assistant", "content": content or None, "tool_calls": calls}
    if "<think>" in text or "</think>" in text:
        raise ValueError("Unexpected reasoning tags in non-thinking mode")
    return {"role": "assistant", "content": text}

class LocalEngine:
    """Load one base/adapter on one GPU; no hosted endpoint, hub fallback or simulator."""
    def __init__(self, workspace, config, device=0):
        os.environ["USE_TORCH"] = "1"
        os.environ["USE_TF"] = "0"
        os.environ["USE_FLAX"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        preflight = gpu_environment(workspace, config.dtype, device=device)
        if not preflight["gpu_verified"]:
            raise RuntimeError("Local GPU evaluation blocked: "+"; ".join(preflight["blockers"]))
        stats = json_file(workspace.sft/"token_stats.json")
        self.identity = tokenizer_identity(workspace.model, stats)
        self.preflight = preflight
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = torch.device("cuda", device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(workspace.model), local_files_only=True,
                                                       trust_remote_code=False, token=False)
        if fingerprint(self.tokenizer.chat_template) != fingerprint(json_file(workspace.model/"tokenizer_config.json")["chat_template"]):
            raise ValueError("Loaded template differs from official prepared file")
        base = AutoModelForCausalLM.from_pretrained(str(workspace.model), local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.bfloat16 if config.dtype == "bf16" else torch.float32,
                attn_implementation="sdpa")
        if config.model_mode == "adapter":
            adapter = project_path(config.adapter, workspace.project)
            contract = json_file(adapter/"run_contract.json") if (adapter/"run_contract.json").exists() else \
                json_file(adapter/"checkpoint_metadata.json")["contract"]
            if contract["model_manifest"] != preflight["model_inventory"] or contract["tokenizer_identity"] != self.identity:
                raise ValueError("Adapter belongs to different base weights/tokenizer")
            if contract["mode"] != "train":
                raise ValueError("One-step smoke adapters are not SFT checkpoints for benchmark comparison")
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, str(adapter), local_files_only=True, is_trainable=False)
        self.model = base.to(self.device).eval()
        self.context_limit = self.identity["native_context"]
        self.model_id = "Qwen/Qwen3-1.7B:"+preflight["model_inventory"]["digest"][:16]
        if config.model_mode == "adapter":
            from .data import sha256
            self.model_id += ":adapter:"+sha256(project_path(config.adapter)/"adapter_model.safetensors")[:16]

    def set_seed(self, seed):
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def prompt_ids(self, prompt):
        """Use precisely the official inference generation prompt with thinking disabled."""
        return self.tokenizer.apply_chat_template(prompt["messages"], tools=prompt["tools"],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, truncation=False)

    def generate(self, ids, config):
        torch = self.torch
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(inputs)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(input_ids=inputs, attention_mask=attention,
                max_new_tokens=config.max_tokens, do_sample=config.temperature > 0,
                temperature=config.temperature if config.temperature > 0 else None,
                top_p=1.0 if config.temperature > 0 else None, top_k=0 if config.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True)
        torch.cuda.synchronize(self.device)
        tokens = generated[0, len(ids):].tolist()
        ended = bool(tokens and tokens[-1] == self.tokenizer.eos_token_id)
        content_tokens = tokens[:-1] if ended else tokens
        return {"text": self.tokenizer.decode(content_tokens, skip_special_tokens=False),
                "raw_text": self.tokenizer.decode(tokens, skip_special_tokens=False),
                "generated_token_ids": tokens,
                "finish_reason": "stop" if ended else "length",
                "latency_ms": (time.perf_counter()-start)*1000}

class LocalPredictor:
    """Existing Predictor interface; a shared engine never sees task answers or private state."""
    def __init__(self, engine, config, seed=None):
        self.engine, self.config = engine, copy.deepcopy(config)
        self.source = "local_lora" if config.model_mode == "adapter" else "local_base"
        self.config.model = engine.model_id
        self.requests = self.used_tokens = 0
        self.budget_stop_reason = None
        if seed is not None:
            self.config.seed = seed
        engine.set_seed(self.config.seed)

    def predict(self, prompt):
        if self.requests >= self.config.max_requests:
            self.budget_stop_reason = "request_budget"
            raise BudgetExceeded(self.budget_stop_reason)
        ids = self.engine.prompt_ids(prompt)
        if len(ids)+self.config.max_tokens > self.engine.context_limit:
            self.budget_stop_reason = "context_limit"
            raise BudgetExceeded(self.budget_stop_reason)
        if self.used_tokens+len(ids)+self.config.max_tokens > self.config.max_total_tokens:
            self.budget_stop_reason = "episode_token_budget"
            raise BudgetExceeded(self.budget_stop_reason)
        self.requests += 1
        started = time.perf_counter()
        try:
            generated = self.engine.generate(ids, self.config)
        except Exception as error:
            # Unknown generated-token consumption reserves the full request; no hidden automatic retry.
            self.used_tokens += len(ids)+self.config.max_tokens
            return {"raw_response": None, "error": {"type": "local_inference_error", "exception_type": type(error).__name__},
                    "usage": {}, "latency_ms": (time.perf_counter()-started)*1000, "model_id": self.engine.model_id}
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(generated["generated_token_ids"])}
        usage["total_tokens"] = sum(usage.values())
        self.used_tokens += usage["total_tokens"]
        format_error = None
        try:
            message = hermes_message(generated["text"], self.requests)
        except (ValueError, TypeError) as error:
            format_error = str(error)
            # A malformed real output remains an invalid tool decision in the original parser.
            message = {"role": "assistant", "content": None, "tool_calls": [{}]}
        finish = generated["finish_reason"]
        if message.get("tool_calls") and finish != "length":
            finish = "tool_calls"
        raw = {"model": self.engine.model_id, "choices": [{"message": message, "finish_reason": finish}],
               "usage": usage, "raw_generation": generated}
        return {"raw_response": raw, "error": None, "usage": usage,
                "latency_ms": generated["latency_ms"], "model_id": self.engine.model_id,
                "format_error": format_error}
