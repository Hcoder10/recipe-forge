"""Drift and loop detector. Watches each agent's tool calls.

Signals: repeat_call, repeat_failure, read_streak, no_progress, prediction_miss, tool_bug, env_blocked.
Agent faults go in two stages: a nudge to the agent, then a Lead trigger if it happens again.
Infra faults (tool_bug, env_blocked) escalate to the Lead and the human inbox on first occurrence.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque

READ_TOOLS = {"read", "search", "diff", "facts", "profile", "gate_status", "team_status", "gate_requests"}
PROGRESS_TOOLS = {"bench", "make_bundle", "request_gate", "kernel_test", "review_gate", "report"}


class Detector:
    def __init__(self, decisions, inbox, log, *, repeat_n=3, read_streak_n=14, no_progress_steps=30):
        self.decisions = decisions
        self.inbox = inbox
        self.log = log
        self.repeat_n = repeat_n
        self.read_streak_n = read_streak_n
        self.no_progress_steps = no_progress_steps
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=12))
        self.read_streak: dict[str, int] = defaultdict(int)
        self.since_progress: dict[str, int] = defaultdict(int)
        self.fail_streak: dict[str, list] = defaultdict(list)
        self.strikes: dict[tuple, int] = defaultdict(int)
        self.nudges: dict[str, list[str]] = defaultdict(list)
        self.infra_seen: set[str] = set()
        self.counts: dict[str, int] = defaultdict(int)

    def take_nudges(self, agent: str) -> list[str]:
        out, self.nudges[agent] = self.nudges[agent], []
        return out

    def _signal(self, agent: str, signal: str, detail: str, infra: bool = False) -> None:
        self.counts[signal] += 1
        self.log("detector", agent=agent, signal=signal, detail=detail[:400], infra=infra)
        if infra:
            key = f"{signal}:{detail[:80]}"
            if key in self.infra_seen:
                return
            self.infra_seen.add(key)
            self.decisions.trigger(signal, f"{agent}: {detail}", source=agent, infra=True)
            self.inbox.ask("detector", f"[infra:{signal}] {agent}: {detail}")
            return
        k = (agent, signal)
        self.strikes[k] += 1
        if self.strikes[k] == 1:
            self.nudges[agent].append(f"[detector:{signal}] {detail}")
        else:
            self.decisions.trigger(signal, f"{agent}: {detail}", source=agent)
            self.strikes[k] = 0

    def observe(self, agent: str, role: str, name: str, args: dict, res: dict) -> None:
        sig = name + ":" + json.dumps(args, sort_keys=True, default=str)[:400]
        h = self.hist[agent]
        h.append(sig)
        cls = res.get("class")
        if cls in ("tool_bug", "env_blocked"):
            self._signal(agent, cls, f"{name}: {res.get('error', '')}", infra=True)
        if list(h).count(sig) >= self.repeat_n and name not in ("team_status", "gate_status", "gate_requests"):
            self._signal(agent, "repeat_call", f"{name} called with the same arguments {list(h).count(sig)} "
                                               "times recently")
            h.clear()
        if cls == "agent_error":
            fs = self.fail_streak[agent]
            fs.append(name)
            if len(fs) >= self.repeat_n and len(set(fs[-self.repeat_n:])) == 1:
                self._signal(agent, "repeat_failure", f"{name} failed {len(fs)} times in a row: "
                                                      f"{res.get('error', '')}")
                fs.clear()
        else:
            self.fail_streak[agent] = []
        if role == "worker":
            if name in READ_TOOLS:
                self.read_streak[agent] += 1
                if self.read_streak[agent] >= self.read_streak_n:
                    self._signal(agent, "read_streak", f"{self.read_streak[agent]} reading calls in a row "
                                                       "without a measurement or a candidate")
                    self.read_streak[agent] = 0
            else:
                self.read_streak[agent] = 0
            if name in PROGRESS_TOOLS and res.get("ok"):
                self.since_progress[agent] = 0
            else:
                self.since_progress[agent] += 1
                if self.since_progress[agent] >= self.no_progress_steps:
                    self._signal(agent, "no_progress", f"{self.since_progress[agent]} calls without a bench, "
                                                       "bundle or gate request")
                    self.since_progress[agent] = 0

    def observe_prediction(self, agent: str, predicted: float | None, actual: float | None, bench_id: str) -> None:
        if predicted is None or actual is None:
            return
        miss = abs(actual - predicted) > max(2.0, abs(predicted))
        if miss:
            self._signal(agent, "prediction_miss", f"{bench_id}: predicted {predicted:+.1f}%, measured "
                                                   f"{actual:+.1f}%")
