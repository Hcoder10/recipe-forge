"""Tool registry and dispatch.

Every result an agent sees is a short JSON object. Failures are one line and carry a class:
  agent_error  - the call was wrong (bad args, missing file); the agent can fix it
  env_blocked  - the machine or a service is in the way (disk, lease, executor down); escalate at once
  tool_bug     - our code raised; escalate at once, never retry silently
No tracebacks reach an agent; they go to the mission's event log.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from forge.tools.schema import ArgError, coerce

RESULT_CHARS = 6000
# Common argument-name slips, mapped only when the alias is a real parameter of the tool being called.
ALIASES = {"limit": "lines", "num_lines": "lines", "n_lines": "lines", "offset": "start", "line": "start",
           "start_line": "start", "file": "path", "filename": "path", "command": "cmd", "delta": "config_delta",
           "config": "config_delta", "bundle_id": "bundle", "query": "pattern", "regex": "pattern",
           "message": "text", "note": "text"}


class ToolError(Exception):
    """Agent-fixable failure. The message is shown to the agent as is (keep it one line)."""


class EnvBlocked(Exception):
    """Infrastructure is in the way. Shown to the agent, and escalated."""


@dataclass
class Tool:
    name: str
    summary: str                    # one line: what it does (capability only)
    params: dict                    # JSON schema (type: object)
    fn: Callable[..., Any]
    roles: tuple[str, ...] = ("worker",)
    returns: str = ""               # one line: what comes back
    order: int = 50                 # manual / tool-list order; high-level tools first
    light: bool = True              # light tools never touch a GPU

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.summary,
                                                 "parameters": self.params}}


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> Tool:
        self.tools[tool.name] = tool
        return tool

    def for_role(self, role: str) -> list[Tool]:
        return sorted((t for t in self.tools.values() if role in t.roles), key=lambda t: (t.order, t.name))

    def schemas(self, role: str) -> list[dict]:
        return [t.openai_schema() for t in self.for_role(role)]

    def manual(self, role: str) -> str:
        """Compact generated MANUAL: capabilities only, in tool order."""
        lines = ["TOOLS (call them with native tool calls; results are JSON):"]
        for t in self.for_role(role):
            props = t.params.get("properties", {})
            req = set(t.params.get("required", []))
            args = ", ".join(f"{k}{'' if k in req else '?'}" for k in props)
            line = f"- {t.name}({args}): {t.summary}"
            if t.returns:
                line += f" Returns {t.returns}"
            lines.append(line)
        lines.append("Errors come back as {ok:false, error, class}. class agent_error: fix the call. "
                     "class env_blocked or tool_bug: it is reported automatically; do something else.")
        return "\n".join(lines)

    def dispatch(self, ctx: Any, role: str, name: str, args: dict, parse_error: str | None = None) -> dict:
        """Run one tool call. Never raises. Returns {ok, ..., class?, error?, elapsed_s}."""
        t0 = time.time()
        tool = self.tools.get(name)
        if tool is None or role not in tool.roles:
            allowed = ", ".join(t.name for t in self.for_role(role))
            return {"ok": False, "class": "agent_error",
                    "error": f"no tool {name!r} for {role}; available: {allowed}"}
        if parse_error:
            return {"ok": False, "class": "agent_error", "error": f"{name}: {parse_error}"}
        props = tool.params.get("properties") or {}
        args = {ALIASES[k] if k not in props and ALIASES.get(k) in props else k: v for k, v in (args or {}).items()}
        unknown = [k for k in args if k not in props]
        if unknown:
            return {"ok": False, "class": "agent_error",
                    "error": f"{name}: unknown argument {unknown[0]!r}; allowed: {', '.join(props) or 'none'}"}
        try:
            clean = coerce(args, tool.params)
        except ArgError as e:
            return {"ok": False, "class": "agent_error", "error": f"{name}: {e}"}
        try:
            out = tool.fn(ctx, **clean)
            res = out if isinstance(out, dict) else {"result": out}
            res = {"ok": True, **res} if "ok" not in res else res
        except ToolError as e:
            res = {"ok": False, "class": "agent_error", "error": _one_line(e)}
        except EnvBlocked as e:
            res = {"ok": False, "class": "env_blocked", "error": _one_line(e)}
        except Exception as e:  # noqa: BLE001 - this is the tool_bug boundary
            res = {"ok": False, "class": "tool_bug",
                   "error": f"internal error in {name} ({type(e).__name__}); reported to the operators"}
            log = getattr(ctx, "log_event", None)
            if log:
                log("tool_bug", tool=name, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-4000:])
        res["elapsed_s"] = round(time.time() - t0, 2)
        return res


def _one_line(e: BaseException) -> str:
    s = str(e).strip().splitlines()
    return (s[0] if s else type(e).__name__)[:400]


def render_result(res: dict, limit: int = RESULT_CHARS) -> str:
    s = json.dumps(res, default=str)
    if len(s) <= limit:
        return s
    # Shrink the largest string fields first so the JSON stays valid.
    res = dict(res)
    for k in sorted(res, key=lambda k: -len(str(res[k]))):
        v = res[k]
        if isinstance(v, str) and len(v) > 400:
            keep = max(200, limit - (len(s) - len(v)) - 80)
            res[k] = v[: keep // 2] + f"\n...[{len(v) - keep} chars cut]...\n" + v[-keep // 2:]
            s = json.dumps(res, default=str)
            if len(s) <= limit:
                return s
    return s[:limit] + '..."}'
