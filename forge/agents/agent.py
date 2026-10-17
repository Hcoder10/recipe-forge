"""One team member: holds its history, builds cache-stable requests, enforces its context cap, compacts in
chunks (newest turns kept + a learned summary), and runs its tool calls."""
from __future__ import annotations

import json
import time

from forge.agents import prompts
from forge.agents.llm import BudgetExceeded, ContextTooLong, LLMError
from forge.tools.context import ToolContext
from forge.tools.registry import render_result

CHARS_PER_TOKEN = 3.2   # conservative for code-heavy text; over-estimating is the safe side


def est_tokens(obj) -> int:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return int(len(s) / CHARS_PER_TOKEN) + 4


class Agent:
    def __init__(self, name: str, role: str, runner, model, cap: int, role_text: str):
        self.name = name
        self.role = role
        self.r = runner                   # MissionRunner
        self.model = model
        self.cap = cap
        self.tools = runner.registry.schemas(role)
        self.prefix = prompts.fixed_prefix(role_text, runner.registry.manual(role), runner.brief,
                                           runner.protocol_text)
        self.history: list[dict] = []
        self.learned = ""
        self.context_msg = ""
        self.turns = 0
        self.steps = 0
        self.infra_retry_at = 0.0
        self.last_ctx_refresh = -10**9
        self.ctx = ToolContext(runner.mission, runner.executor, role, name, runner.log, runner.services)
        self.fixed_tokens = est_tokens(self.prefix) + est_tokens(self.tools)

    # -- prompt -----------------------------------------------------------------------------------
    def refresh_context(self, force: bool = False) -> None:
        every = int(self.r.playbook["knobs"].get("facts_refresh_turns", 8))
        if force or self.turns - self.last_ctx_refresh >= every or not self.context_msg:
            self.context_msg = prompts.context_block(self.r.services.brain.prompt_view(self.r.mission.recipe_id),
                                                     self.learned)
            self.last_ctx_refresh = self.turns

    def messages(self, status: str) -> list[dict]:
        return ([{"role": "system", "content": self.prefix}, {"role": "user", "content": self.context_msg}]
                + self.history + [{"role": "user", "content": status}])

    def request_tokens(self, status: str) -> int:
        return self.fixed_tokens + est_tokens(self.context_msg) + sum(est_tokens(m) for m in self.history) \
            + est_tokens(status) + 64

    def add_event_message(self, text: str) -> None:
        """A durable event for this agent (new task, answer from a human, gate result)."""
        self._append_user(text)

    def _append_user(self, text: str) -> None:
        if self.history and self.history[-1]["role"] == "user":
            self.history[-1] = {"role": "user", "content": self.history[-1]["content"] + "\n" + text}
        else:
            self.history.append({"role": "user", "content": text})

    # -- compaction -------------------------------------------------------------------------------
    def _turn_starts(self) -> list[int]:
        """Indexes where a turn may start (never between an assistant tool call and its tool results)."""
        return [i for i, m in enumerate(self.history) if m["role"] in ("user",) or
                (m["role"] == "assistant" and (i == 0 or self.history[i - 1]["role"] != "assistant"))]

    def compact(self, status: str) -> None:
        keep_turns = int(self.r.playbook["knobs"].get("keep_turns", 12))
        budget = int(self.cap * 0.85)
        starts = [i for i in self._turn_starts() if i > 0]
        # Compact a chunk: cut at the latest turn start that still keeps keep_turns assistant turns, unless
        # even that tail does not fit under ~60% of the cap, then cut later (the cap is a hard limit).
        assistant_idx = [i for i, m in enumerate(self.history) if m["role"] == "assistant"]
        keep_from = assistant_idx[-keep_turns] if len(assistant_idx) > keep_turns else 0
        cands = [i for i in starts if i <= keep_from] or starts[:1]
        cut = cands[-1] if cands else len(self.history)
        for i in [s for s in starts if s >= cut]:
            tail = sum(est_tokens(m) for m in self.history[i:])
            if self.fixed_tokens + est_tokens(self.context_msg) + tail + est_tokens(status) < budget * 0.7:
                cut = i
                break
        else:
            cut = starts[-1] if starts else len(self.history)
        old, self.history = self.history[:cut], self.history[cut:]
        while self.history and self.history[0]["role"] == "tool":
            old.append(self.history.pop(0))
        summary = self._summarize(old)
        self.learned = (self.learned + "\n" + summary).strip()[-6000:]
        self.refresh_context(force=True)
        self.r.log("compaction", agent=self.name, dropped_msgs=len(old), kept_msgs=len(self.history),
                   learned_chars=len(self.learned))
        # last resort: shrink big tool results in what is left
        for m in self.history:
            if self.request_tokens(status) <= budget:
                break
            if m["role"] == "tool" and len(m["content"]) > 1500:
                m["content"] = m["content"][:700] + " ...[trimmed by compaction]... " + m["content"][-500:]

    def _summarize(self, msgs: list[dict]) -> str:
        if not msgs:
            return ""
        text = "\n".join(f"{m['role']}: {(m.get('content') or '')[:1500]}"
                         + (f" CALLS {json.dumps(m.get('tool_calls'))[:800]}" if m.get("tool_calls") else "")
                         for m in msgs)[-60000:]
        try:
            resp = self.r.client.chat(self.model, [{"role": "system", "content": prompts.SUMMARIZE},
                                                   {"role": "user", "content": text}],
                                      role=f"{self.name}:compact", mission_id=self.r.mission.id, max_out=600)
            return resp.content.strip()
        except LLMError:
            return "(summary unavailable) " + text[-1200:]

    # -- one step ---------------------------------------------------------------------------------
    def step(self, status: str) -> dict:
        """One LLM call and its tool calls. Returns {called: [names], text, stop?}."""
        self.refresh_context()
        if self.request_tokens(status) > int(self.cap * 0.85):
            self.compact(status)
        msgs = self.messages(status)
        try:
            resp = self.r.client.chat(self.model, msgs, self.tools, role=self.name, mission_id=self.r.mission.id)
        except ContextTooLong:
            self.compact(status)
            resp = self.r.client.chat(self.model, self.messages(status), self.tools, role=self.name,
                                      mission_id=self.r.mission.id)
        self.turns += 1
        self.history.append(resp.assistant_message())
        self.r.log("llm_call", agent=self.name, role=self.role, prompt_tokens=resp.usage.prompt,
                   cached_tokens=resp.usage.cached, completion_tokens=resp.usage.completion,
                   cost_usd=round(resp.cost_usd, 5), n_tool_calls=len(resp.tool_calls),
                   text=(resp.content or "")[:300])
        called = []
        infra_blocked = False
        for tc in resp.tool_calls:
            if infra_blocked:
                res = {"ok": False, "class": "agent_error", "error": "Call skipped after an infrastructure "
                       "or tool failure in this batch; inspect the failure before continuing."}
            else:
                res = self.r.registry.dispatch(self.ctx, self.role, tc.name, tc.args, tc.parse_error)
            if res.get("class") in ("env_blocked", "tool_bug"):
                infra_blocked = True
                self.infra_retry_at = time.time() + 60
            self.steps += 1
            called.append(tc.name)
            self.r.log("tool_call", agent=self.name, tool=tc.name, args=_short(tc.args), ok=res.get("ok"),
                       cls=res.get("class"), error=res.get("error"), elapsed_s=res.get("elapsed_s"),
                       result=_short(res, 600))
            self.r.detector.observe(self.name, self.role, tc.name, tc.args, res)
            self.history.append({"role": "tool", "tool_call_id": tc.id, "content": render_result(res)})
        return {"called": called, "text": resp.content, "infra_blocked": infra_blocked}


def _short(obj, n: int = 300) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "..."
