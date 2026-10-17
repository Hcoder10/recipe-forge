"""python -m forge.agents run --spec mission.json [--profile P] | probe [--profile P] | selftest | answer ...

run       run one mission and print its row
probe     list the profile's /v1/models and check the configured ids, then one tiny tool-call round trip
selftest  a full mission in-process with the mock brain, mock bench and mock gate (no GPU, no network)
answer    append a human answer for a mission's inbox question
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def cmd_run(a) -> int:
    from forge.agents.mission import run_mission
    spec = json.loads(Path(a.spec).read_text())
    if a.profile:
        spec["profile"] = a.profile
    if a.profile == "mock" or spec.get("profile") == "mock":
        from forge.agents.mock import MockBrain
        row = run_mission(spec, mock_llm=MockBrain())
    else:
        row = run_mission(spec)
    print(json.dumps(row, indent=1, default=str))
    return 0


def cmd_probe(a) -> int:
    from forge.agents.config import load_secrets, profile
    from forge.agents.llm import LLMClient, Meter
    load_secrets()
    p = profile(a.profile)
    c = LLMClient(p, Meter(None))
    if not c.api_key and p.name == "crusoe":
        print("CRUSOE_API_KEY not set (looked in env and ~/forge_secrets.env)")
        return 2
    try:
        ids = c.list_models()
    except Exception as e:  # noqa: BLE001
        print(f"/models failed: {type(e).__name__}: {e}")
        return 1
    print("served models:")
    for i in ids:
        print("  ", i)
    ok = True
    for tier, m in p.models.items():
        hit = m.model_id in ids
        ok &= hit or tier == "nano"
        print(f"{tier:6s} {m.model_id}: {'OK' if hit else 'MISSING'}")
    tools = [{"type": "function", "function": {"name": "note", "description": "Post a note.",
                                               "parameters": {"type": "object", "properties": {"text": {
                                                   "type": "string"}}, "required": ["text"]}}}]
    for tier in ("super", "ultra"):
        m = p.models[tier]
        if m.model_id not in ids:
            continue
        msgs = [{"role": "system", "content": "You are a test harness. " * 200},
                {"role": "user", "content": "Call the note tool with text 'ping'."}]
        for i in range(2):  # second call shows whether the prefix is cached
            try:
                r = c.chat(m, msgs, tools, role=f"probe-{tier}")
                print(f"{tier} call {i + 1}: tool_calls={[(t.name, t.args) for t in r.tool_calls]} "
                      f"prompt={r.usage.prompt} cached={r.usage.cached} out={r.usage.completion} "
                      f"${r.cost_usd:.5f} {r.latency_s:.1f}s")
            except Exception as e:  # noqa: BLE001
                print(f"{tier} call failed: {e}")
                ok = False
    return 0 if ok else 1


def cmd_selftest(a) -> int:
    from forge.agents.mission import MissionRunner
    from forge.agents.mock import MockBrain
    from forge.tools.bench import MockBenchRunner
    from forge.tools.bundle import canon
    from forge.tools.executor import LocalExecutor
    from forge.tools.gateclient import MockGate
    tmp = Path(tempfile.mkdtemp(prefix="forge-selftest-"))
    sp = tmp / "site" / "vllm"
    sp.mkdir(parents=True)
    (sp / "x.py").write_text("A = 1\n")
    (tmp / "prompts.jsonl").write_text('{"prompt": "hi"}\n')
    spec = {"mission_id": "selftest", "recipe": {"id": "toy", "model": "toy", "serve": {"args": []},
                                                 "workload": {"metric": "decode_tok_s"}},
            "machine": {"name": "local"}, "python": sys.executable, "site_packages": str(tmp / "site"),
            "bench_prompts": str(tmp / "prompts.jsonl"), "time_box_min": 1, "budget_usd": 1,
            "team": {"workers": 2, "lead_every_s": 0.2}, "profile": "mock", "require_confirm": False,
            "gate_poll_s": 0.1, "tick_s": 0.1}
    key = canon({"d": {"--seed": 1}, "e": None, "b": None})
    row = MissionRunner(spec, executor=LocalExecutor(tmp / "target"),
                        bench=MockBenchRunner(noise=0, fixed_effects={key: 0.08}),
                        gate=MockGate(tmp / "gate.jsonl", verdict_fn=lambda b: ("win", 5.0)),
                        mock_llm=MockBrain(), root=tmp / "state").run()
    ok = row["done"]["status"] == "win" and not row["errors"]
    print(json.dumps({"ok": ok, "status": row["done"]["status"], "delta_pct": row["delta_pct"],
                      "errors": row["errors"], "state": str(tmp)}, indent=1))
    return 0 if ok else 1


def cmd_answer(a) -> int:
    from forge.agents import events
    from forge.agents.config import state_root
    d = state_root() / "missions" / a.mission
    events.append(d / "answers.jsonl", {"question_id": a.question, "answer": a.text, **({"to": a.to} if a.to
                                                                                      else {})})
    print("ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m forge.agents")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--profile", choices=["crusoe", "local", "mock"])
    p = sub.add_parser("probe")
    p.add_argument("--profile", default="crusoe", choices=["crusoe", "local"])
    sub.add_parser("selftest")
    an = sub.add_parser("answer")
    an.add_argument("mission")
    an.add_argument("question")
    an.add_argument("text")
    an.add_argument("--to")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "probe": cmd_probe, "selftest": cmd_selftest, "answer": cmd_answer}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
