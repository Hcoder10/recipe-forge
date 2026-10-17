"""OpenAI-compatible chat client (Crusoe Managed Inference, a local server, or a mock).

Plain urllib, no SDK: the laptop and the rig have different SDK versions and we need little of it.
Every call is metered (tokens, cached tokens, $) into the mission's usage log.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from forge.agents.config import PRICES, ModelSpec, Profile


def _connect_v4_first(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *a, **kw):
    """Like socket.create_connection but IPv4 first: on the rig the API's IPv6 addresses are unreachable and
    each costs a full connect timeout before urllib falls back."""
    host, port = address
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    infos.sort(key=lambda i: i[0] != socket.AF_INET)
    last: OSError | None = None
    for fam, typ, proto, _canon, sa in infos:
        sock = socket.socket(fam, typ, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            sock.connect(sa)
            return sock
        except OSError as e:
            last = e
            sock.close()
    raise last or OSError(f"cannot resolve {host}")


class _V4HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._create_connection = _connect_v4_first


class _V4HTTPConnection(http.client.HTTPConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._create_connection = _connect_v4_first


class _V4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_V4HTTPSConnection, req, context=self._context)


class _V4HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_V4HTTPConnection, req)


_OPENER = urllib.request.build_opener(_V4HTTPSHandler(), _V4HTTPHandler())


class LLMError(Exception):
    """A call failed after retries (infra: escalate, do not blame the agent)."""


class ContextTooLong(LLMError):
    """The server rejected the request as too long; the caller should compact and retry."""


class BudgetExceeded(LLMError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    raw_args: str = ""
    parse_error: str | None = None


@dataclass
class Usage:
    prompt: int = 0
    cached: int = 0
    completion: int = 0

    def cost(self, tier: str | tuple) -> float:
        p_in, p_cached, p_out = tier if isinstance(tier, tuple) else PRICES.get(tier, PRICES["ultra"])
        fresh = max(0, self.prompt - self.cached)
        return (fresh * p_in + self.cached * p_cached + self.completion * p_out) / 1e6


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    usage: Usage
    cost_usd: float
    finish_reason: str = ""
    latency_s: float = 0.0
    reasoning: str = ""

    def assistant_message(self) -> dict:
        """The message to append to history (reasoning is dropped to keep the context small)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.name, "arguments": tc.raw_args or json.dumps(tc.args)}}
                                 for tc in self.tool_calls]
        return msg


def parse_args(raw: Any) -> tuple[dict, str | None]:
    """Tool-call arguments into a dict. Tolerates a dict, a JSON string, a JSON string inside a string,
    and trailing junk after the object."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    s = str(raw).strip()
    for _ in range(2):
        try:
            v = json.loads(s)
        except json.JSONDecodeError:
            # trailing text after a complete object
            try:
                v, _end = json.JSONDecoder().raw_decode(s)
            except json.JSONDecodeError as e:
                return {}, f"arguments are not valid JSON ({e.msg} at char {e.pos})"
        if isinstance(v, dict):
            return v, None
        if isinstance(v, str):
            s = v.strip()
            continue
        return {}, "arguments must be a JSON object"
    return {}, "arguments must be a JSON object"


class Meter:
    """Thread-safe running totals for one mission, appended to usage.jsonl."""

    def __init__(self, path: Path | None, budget_usd: float | None = None):
        self.path = path
        self.budget_usd = budget_usd
        self.lock = threading.Lock()
        self.cost_usd = 0.0
        self.calls = 0
        self.prompt = self.cached = self.completion = 0
        self.by_role: dict[str, float] = {}

    def record(self, row: dict) -> None:
        with self.lock:
            self.calls += 1
            self.cost_usd += row["cost_usd"]
            self.prompt += row["prompt_tokens"]
            self.cached += row["cached_tokens"]
            self.completion += row["completion_tokens"]
            self.by_role[row.get("role", "?")] = self.by_role.get(row.get("role", "?"), 0.0) + row["cost_usd"]
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

    def over_budget(self) -> bool:
        return self.budget_usd is not None and self.cost_usd >= self.budget_usd

    def summary(self) -> dict:
        with self.lock:
            return {"calls": self.calls, "cost_usd": round(self.cost_usd, 4), "prompt_tokens": self.prompt,
                    "cached_tokens": self.cached, "completion_tokens": self.completion,
                    "cache_hit": round(self.cached / self.prompt, 3) if self.prompt else 0.0,
                    "by_role": {k: round(v, 4) for k, v in self.by_role.items()}}


# A mock backend is any callable(model, messages, tools, role) -> OpenAI-style "message" dict
# (content, tool_calls) optionally with a "usage" dict.
MockFn = Callable[[ModelSpec, list, list, str], dict]


class LLMClient:
    def __init__(self, profile: Profile, meter: Meter | None = None, mock: MockFn | None = None,
                 timeout_s: float = 300.0, max_retries: int = 6):
        self.profile = profile
        self.meter = meter or Meter(None)
        self.mock = mock
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.api_key = os.environ.get(profile.api_key_env, "") if profile.api_key_env else ""
        if profile.mock and mock is None:
            raise ValueError("mock profile needs a mock backend")

    # -- transport -------------------------------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.profile.base_url.rstrip("/") + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"}, method="POST")
        with _OPENER.open(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode())

    def list_models(self) -> list[str]:
        req = urllib.request.Request(self.profile.base_url.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {self.api_key}"})
        with _OPENER.open(req, timeout=30) as r:
            data = json.loads(r.read().decode())
        return [m.get("id") for m in data.get("data", [])]

    def _call_with_retries(self, body: dict) -> dict:
        last = ""
        for attempt in range(self.max_retries):
            try:
                return self._post("/chat/completions", body)
            except urllib.error.HTTPError as e:
                text = ""
                try:
                    text = e.read().decode()[:500]
                except Exception:
                    pass
                low = text.lower()
                if e.code == 400 and ("context" in low or "too long" in low or "maximum" in low
                                      and "token" in low):
                    raise ContextTooLong(f"HTTP 400: {text[:200]}")
                if e.code in (400, 401, 403, 404, 422):
                    raise LLMError(f"HTTP {e.code}: {text[:200]}")
                last = f"HTTP {e.code}: {text[:200]}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
            except json.JSONDecodeError as e:
                last = f"bad JSON from server: {e}"
            time.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
        raise LLMError(f"gave up after {self.max_retries} tries: {last}")

    # -- public ----------------------------------------------------------------------------------
    def chat(self, model: ModelSpec, messages: list[dict], tools: list[dict] | None = None, *,
             role: str = "", mission_id: str = "", max_out: int | None = None,
             tool_choice: str | None = None) -> LLMResponse:
        if self.meter.over_budget():
            raise BudgetExceeded(f"LLM budget ${self.meter.budget_usd:.2f} used up")
        t0 = time.time()
        if self.mock is not None:
            msg = self.mock(model, messages, tools or [], role)
            data = {"choices": [{"message": msg, "finish_reason": msg.get("finish_reason", "stop")}],
                    "usage": msg.get("usage") or _mock_usage(messages, msg)}
        else:
            body: dict[str, Any] = {"model": model.model_id, "messages": messages,
                                    "temperature": model.temperature, "max_tokens": max_out or model.max_out}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = tool_choice or "auto"
            data = self._call_with_retries(body)
        latency = time.time() - t0
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"malformed response: {str(data)[:200]}")
        u = data.get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        usage = Usage(int(u.get("prompt_tokens") or 0), int(details.get("cached_tokens") or 0),
                      int(u.get("completion_tokens") or 0))
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            args, err = parse_args(raw)
            calls.append(ToolCall(tc.get("id") or f"call_{int(t0 * 1000)}_{i}", fn.get("name") or "",
                                  args, raw if isinstance(raw, str) else json.dumps(raw or {}), err))
        cost = usage.cost(model.prices)
        self.meter.record({"t": round(t0, 3), "mission_id": mission_id, "role": role, "model": model.model_id,
                           "tier": model.tier, "prompt_tokens": usage.prompt, "cached_tokens": usage.cached,
                           "completion_tokens": usage.completion, "cost_usd": round(cost, 6),
                           "latency_s": round(latency, 2), "n_tool_calls": len(calls)})
        return LLMResponse(msg.get("content") or "", calls, usage, cost, choice.get("finish_reason") or "",
                           latency, msg.get("reasoning_content") or msg.get("reasoning") or "")


def _mock_usage(messages: list, msg: dict) -> dict:
    n_in = sum(len(json.dumps(m)) for m in messages) // 4
    return {"prompt_tokens": n_in, "completion_tokens": len(json.dumps(msg)) // 4,
            "prompt_tokens_details": {"cached_tokens": int(n_in * 0.8)}}
