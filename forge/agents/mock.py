"""A scripted mock brain for end-to-end runs without an LLM (tests and dry runs only; never agent-visible).

It walks the experiment protocol mechanically so the runner, tools, gate path and auditor get exercised.
"""
from __future__ import annotations

import itertools
import json
import re

_ids = itertools.count(1)


def _call(name: str, **args) -> dict:
    return {"id": f"mock_{next(_ids)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _msg(*calls, content: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def _since_task(messages: list[dict]) -> tuple[list[dict], str]:
    """Tool results after the most recent TASK message, and that task's text."""
    idx, task = None, ""
    for i, m in enumerate(messages):
        if m["role"] == "user" and "TASK from the lead" in (m.get("content") or ""):
            idx, task = i, m["content"]
    if idx is None:
        return [], ""
    out = []
    for m in messages[idx + 1:]:
        if m["role"] == "tool":
            try:
                out.append(json.loads(m["content"]))
            except json.JSONDecodeError:
                out.append({})
    return out, task


class MockBrain:
    def __init__(self, deltas: dict[str, dict] | None = None, gate_bar: float = 3.0, stop_on_gate: bool = True):
        # worker name -> config_delta it will try
        self.deltas = deltas or {}
        self.bar = gate_bar
        self.stop_on_gate = stop_on_gate

    def __call__(self, model, messages, tools, role):
        if role.endswith(":compact"):
            return {"role": "assistant", "content": "earlier: measured reference; tried one change."}
        status = messages[-1]["content"] if messages else ""
        if role == "lead":
            return self.lead(messages, status)
        if role == "verifier":
            return self.verifier(messages, status)
        return self.worker(role, messages, status)

    def lead(self, messages, status):
        pend = re.findall(r"(T\d+) \[([a-z_]+)", status)
        if pend:
            kinds = {k for _, k in pend}
            ids = [t for t, _ in pend]
            if self.stop_on_gate and "gate_result" in kinds:
                return _msg(_call("decide", decision="stop", trigger_ids=ids, evidence="gate verdict recorded"))
            return _msg(_call("decide", decision="continue", trigger_ids=ids, evidence="triggers reviewed"))
        if not any(m["role"] == "assistant" for m in messages):
            workers = re.findall(r"(worker-\d+)", messages[0]["content"])
            workers = sorted(set(workers))
            return _msg(*[_call("assign", worker=w, lane="config", task=f"measure one config change ({w})")
                          for w in workers])
        return {"role": "assistant", "content": "waiting for results"}

    def verifier(self, messages, status):
        m = re.search(r"pending review: ([R\d, ]+)\.", status)
        ids = [x.strip() for x in m.group(1).split(",")] if m else []
        if not ids:
            return {"role": "assistant", "content": "nothing to review"}
        return _msg(*[_call("review_gate", request_id=i, approve=True, reason="bench of this bundle clears bar")
                      for i in ids])

    def worker(self, name, messages, status):
        if "You have no task" in status:
            return {"role": "assistant", "content": "idle"}
        results, _task = _since_task(messages)
        n = len(results)
        delta = self.deltas.get(name, {"seed": int(name.rsplit("-", 1)[-1])})
        if n == 0:
            return _msg(_call("bench", claim="reference"))
        if n == 1:
            return _msg(_call("bench", config_delta=json.dumps(delta), claim=f"set {delta}", predict_pct="1"))
        cand = results[1]
        d = cand.get("delta_pct") or 0.0
        if n == 2:
            return _msg(_call("fact_add", claim=f"{delta} gives {d:+.2f}%", lane="config",
                              status="measured", evidence=cand.get("bench_id", "?")))
        if d < self.bar:
            if n == 3:
                return _msg(_call("report", summary=f"{delta}: {d:+.2f}% (below bar)", status="done"))
            return {"role": "assistant", "content": "done"}
        if n == 3:
            return _msg(_call("make_bundle", config_delta=delta, note="mock"))
        bundle = results[3].get("bundle")
        if n == 4:
            return _msg(_call("bench", bundle=bundle, claim="bundle check"))
        if n == 5:
            return _msg(_call("request_gate", bundle=bundle, claim=f"{delta}", bench_id=results[4].get("bench_id")))
        if n == 6:
            return _msg(_call("report", summary=f"gate requested for {bundle}", status="done"))
        return {"role": "assistant", "content": "done"}
