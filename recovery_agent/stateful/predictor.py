import copy
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from ..data_pipeline.common import canonical, write_json
from ..data_pipeline.serialization import action_message, stop_message
from .tasks import StatefulOracle

def load_dotenv(path):
    # Read credential variables locally without printing them; no third-party dotenv dependency.
    for line in path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []:
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in ("MODEL_NAME", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))

def redact(value, secret):
    # Redact credentials recursively, including a provider error accidentally echoing a key.
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [redact(x, secret) for x in value]
    if isinstance(value, dict):
        return {k: redact(v, secret) for k, v in value.items()
                if k.lower() not in ("authorization", "api_key", "openai_api_key")}
    return value

@dataclass
class LLMConfig:
    model: str = "mock"
    base_url: str = ""
    temperature: float | None = None
    max_tokens: int = 4096
    timeout: float = 90
    max_requests: int = 16
    max_total_tokens: int = 160000
    max_cost_cny: float = 10
    input_cny_per_million: float = 30
    output_cny_per_million: float = 90
    max_prompt_bytes: int = 40000

    def __post_init__(self):
        if any(v <= 0 for v in (self.max_tokens, self.timeout, self.max_requests,
                                self.max_total_tokens, self.max_cost_cny,
                                self.input_cny_per_million, self.output_cny_per_million, self.max_prompt_bytes)):
            raise ValueError("All limits and conservative prices must be positive")
        if self.temperature is not None and not 0 <= self.temperature < 2:
            raise ValueError("temperature outside API range")

    def public(self):
        value = asdict(self)
        value["base_url"] = urlparse(self.base_url).hostname or ""
        return value

class BudgetExceeded(Exception):
    pass

class Budget:
    # Write-ahead reservations survive uncertain API failures and process restarts.
    def __init__(self, config, path=None):
        self.config, self.path = config, path
        self.state = {"requests": 0, "reserved_or_used_tokens": 0, "reserved_or_used_cost_cny": 0.0}
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["config"] != config.public():
                raise ValueError("Budget configuration differs on resume")
            self.state = data["state"]

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(canonical({"config": self.config.public(), "state": self.state})+"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)

    def reserve(self, payload):
        # UTF-8 bytes + generous framing reserve bound input for this compatible API.
        size = len(canonical(payload).encode("utf-8"))
        if size > self.config.max_prompt_bytes:
            raise BudgetExceeded("prompt_limit")
        inp, out = size + 4096, self.config.max_tokens
        tokens = inp + out
        cost = (inp*self.config.input_cny_per_million + out*self.config.output_cny_per_million)/1e6
        if (self.state["requests"] >= self.config.max_requests or
            self.state["reserved_or_used_tokens"] + tokens > self.config.max_total_tokens or
            self.state["reserved_or_used_cost_cny"] + cost > self.config.max_cost_cny):
            raise BudgetExceeded("request_token_or_cost_limit")
        self.state["requests"] += 1
        self.state["reserved_or_used_tokens"] += tokens
        self.state["reserved_or_used_cost_cny"] += cost
        self.save()
        return inp, out, tokens, cost

    def settle(self, reservation, usage):
        if not usage or not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
            return  # Unknown billing consumes the full conservative reservation.
        inp, out = usage["prompt_tokens"], usage["completion_tokens"]
        cost = (inp*self.config.input_cny_per_million + out*self.config.output_cny_per_million)/1e6
        self.state["reserved_or_used_tokens"] += inp + out - reservation[2]
        self.state["reserved_or_used_cost_cny"] += cost - reservation[3]
        self.save()

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect_refused", headers, fp)

class OnlinePredictor:
    # Explicit opt-in only. Transport is injectable so tests never contact the network.
    source = "llm"
    def __init__(self, config, api_key, online=False, budget_path=None, transport=None):
        if not online:
            raise ValueError("Online mode must be explicitly enabled")
        url = urlparse(config.base_url)
        if url.scheme != "https" or url.username or url.password or url.query or url.fragment:
            raise ValueError("Use a credential-free HTTPS API endpoint")
        if not api_key or not config.model:
            raise ValueError("Missing model or credential")
        self.config, self.secret = config, api_key
        self.budget = Budget(config, budget_path)
        self.transport = transport or self._http

    def _http(self, payload):
        endpoint = self.config.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        request = urllib.request.Request(endpoint, data=canonical(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.secret})
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(request, timeout=self.config.timeout) as response:
            body = response.read(8*1024*1024+1)
            if len(body) > 8*1024*1024:
                raise ValueError("response_too_large")
            return json.loads(body)

    def predict(self, prompt):
        payload = {"model": self.config.model, **copy.deepcopy(prompt),
                   "max_tokens": self.config.max_tokens, "stream": False}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        reservation = self.budget.reserve(payload)
        start = time.perf_counter()
        try:
            response = self.transport(payload)
            if not isinstance(response, dict):
                raise ValueError("invalid_response")
            self.budget.settle(reservation, response.get("usage"))
            return {"raw_response": redact(response, self.secret), "error": None,
                    "latency_ms": (time.perf_counter()-start)*1000,
                    "usage": response.get("usage", {}), "model_id": response.get("model", self.config.model)}
        except urllib.error.HTTPError as e:
            error = {"type": "rate_limit" if e.code == 429 else "http_error", "status": e.code}
            try:
                body = e.read(8192).decode("utf-8", errors="replace")
                error["provider_response"] = redact(body, self.secret)
            except (AttributeError, OSError):
                pass
        except (TimeoutError, urllib.error.URLError):
            error = {"type": "timeout_or_connection_error"}
        except (ValueError, OSError):
            error = {"type": "invalid_api_response"}
        except Exception as exception:
            error = {"type": "api_exception", "exception_type": type(exception).__name__}
        return {"raw_response": None, "error": error, "usage": {}, "model_id": self.config.model,
                "latency_ms": (time.perf_counter()-start)*1000}

class MockPredictor:
    # Labeled Oracle-scripted mock, never presented as a real model baseline.
    source = "oracle_mock"
    def __init__(self, task, responses=None):
        self.oracle = StatefulOracle(task)
        self.responses = iter(responses) if responses is not None else None
        self.config = LLMConfig()
        self.events = []

    def predict(self, prompt):
        if self.responses is not None:
            response = next(self.responses)
        else:
            action = self.oracle.next_action(self.events)
            if action:
                action["call_id"] = f"mock-{self.oracle.index}"
                message = action_message(action)
            else:
                message = stop_message()
            response = {"model": "oracle-scripted-mock", "choices": [
                {"message": message, "finish_reason": "tool_calls" if action else "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        return {"raw_response": response, "error": None, "usage": response.get("usage", {}),
                "latency_ms": 0, "model_id": response.get("model", "mock")}
