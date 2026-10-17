import json
from pathlib import Path

import pytest

from forge.agents import done, mechanic, playbook
from forge.agents.decisions import Decisions, Inbox
from forge.agents.detector import Detector
from forge.agents.llm import LLMClient, Meter, Usage, parse_args
from forge.agents.config import profile
from forge.agents.mission import normalize_recipe
from forge.tools.bundle import hash_dir, hash_files, merged_args, serve_command
from forge.tools.registry import Registry, Tool, ToolError, EnvBlocked
from forge.tools.schema import ArgError, coerce
from forge.tools.builtin import build_registry


def test_parse_args_variants():
    assert parse_args('{"a": 1}') == ({"a": 1}, None)
    assert parse_args(json.dumps('{"a": 1}')) == ({"a": 1}, None)
    assert parse_args('{"a": 1} trailing')[0] == {"a": 1}
    assert parse_args({"a": 2}) == ({"a": 2}, None)
    assert parse_args("")[0] == {}
    assert parse_args("not json")[1]


def test_coerce():
    sch = {"type": "object", "properties": {"d": {"type": "object"}, "n": {"type": "integer"},
                                            "x": {"type": "number"}, "b": {"type": "boolean"},
                                            "l": {"type": "array", "items": {"type": "string"}},
                                            "e": {"type": "string", "enum": ["stop", "continue"]}},
           "required": ["n"]}
    out = coerce({"d": '{"k": 1}', "n": "3", "x": "2.5%", "b": "true", "l": "T1", "e": "STOP"}, sch)
    assert out == {"d": {"k": 1}, "n": 3, "x": 2.5, "b": True, "l": ["T1"], "e": "stop"}
    with pytest.raises(ArgError):
        coerce({}, sch)
    with pytest.raises(ArgError):
        coerce({"n": "3.5"}, sch)


def test_registry_error_classes():
    reg = Registry()
    events = []

    class Ctx:
        def log_event(self, kind, **f):
            events.append(kind)

    def boom(ctx):
        raise KeyError("x")

    def agent_err(ctx):
        raise ToolError("bad path\nsecond line")

    def env(ctx):
        raise EnvBlocked("disk full")

    for n, f in (("boom", boom), ("ae", agent_err), ("env", env)):
        reg.add(Tool(n, "t", {"type": "object", "properties": {}}, f))
    c = Ctx()
    r = reg.dispatch(c, "worker", "boom", {})
    assert r["class"] == "tool_bug" and "Traceback" not in r["error"] and events == ["tool_bug"]
    assert reg.dispatch(c, "worker", "ae", {}) | {"elapsed_s": 0} == {"ok": False, "class": "agent_error",
                                                                     "error": "bad path", "elapsed_s": 0}
    assert reg.dispatch(c, "worker", "env", {})["class"] == "env_blocked"
    assert reg.dispatch(c, "lead", "boom", {})["class"] == "agent_error"
    assert reg.dispatch(c, "worker", "nope", {})["class"] == "agent_error"


def test_manual_is_capability_only():
    reg = build_registry()
    for role in ("lead", "worker", "verifier"):
        man = reg.manual(role)
        assert "--" not in man.replace("---", "")
    names = [t.name for t in reg.for_role("worker")]
    assert names[:2] == ["profile", "bench"] and names[-2:] == ["sh", "py"]
    for pb_text in (playbook.DEFAULT[k] for k in mechanic.EDITABLE):
        assert not mechanic._FLAG.search(pb_text)


def test_usage_cost():
    u = Usage(prompt=1_000_000, cached=500_000, completion=100_000)
    assert abs(u.cost("ultra") - (0.5 + 0.125 + 0.32)) < 1e-9
    assert abs(Usage(1_000_000, 0, 0).cost("super") - 0.30) < 1e-9


def test_done_audit():
    c = done.compile_criteria("m1", {"id": "r", "workload": {"metric": "decode_tok_s"}}, "rig",
                              "at least 2% faster", require_confirm=True)
    assert c["min_gain_pct"] == 2.0
    base = {"mission_id": "m1", "recipe_id": "r", "status": "done"}
    rows = [{**base, "gate_id": "g1", "verdict": "win", "delta_pct": 5, "bundle_hash": "h1", "kind": "candidate"}]
    assert done.audit(c, rows)["status"] == "no_win"          # no confirm yet
    rows.append({**base, "gate_id": "g2", "verdict": "win", "delta_pct": 4, "bundle_hash": "h1", "kind": "confirm"})
    a = done.audit(c, rows)
    assert a["status"] == "win" and a["best"]["gate_id"] == "g1"
    rows2 = [{**base, "gate_id": "g3", "verdict": "lossy", "delta_pct": 9, "bundle_hash": "h2"},
             {**base, "gate_id": "g4", "status": "running", "verdict": None}]
    assert done.audit(c, rows2)["status"] == "pending"


def test_done_lock_tamper(tmp_path):
    c = done.compile_criteria("m1", {"id": "r"}, "rig", "")
    p = tmp_path / "lock.json"
    done.lock(c, p)
    assert done.load_locked(p) == c
    d = json.loads(p.read_text())
    d["criteria"]["min_gain_pct"] = -5
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        done.load_locked(p)


def test_decisions(tmp_path):
    log = lambda *a, **k: None
    d = Decisions(tmp_path / "d.jsonl", log)
    t1 = d.trigger("worker_report", "w1 done")
    t2 = d.trigger("time", "50%")
    assert t1 in d.pending_text() and t2 in d.pending_text()
    r = d.decide("lead", "Proceed", [t1, "T99"], "B1")
    assert r["recorded"] == "continue" and r["closed"] == [t1] and r["ignored_ids"] == ["T99"]
    r = d.decide("lead", "halt", t2, "done")
    assert d.stop_requested and not d.pending()
    with pytest.raises(ToolError):
        d.decide("lead", "dance", [], "")


def test_detector(tmp_path):
    log = lambda *a, **k: None
    dec = Decisions(tmp_path / "d.jsonl", log)
    inbox = Inbox(tmp_path / "i.jsonl", tmp_path / "a.jsonl", log)
    det = Detector(dec, inbox, log, read_streak_n=3)
    for _ in range(3):
        det.observe("w1", "worker", "read", {"path": "a"}, {"ok": True})
    assert det.take_nudges("w1")                 # first strike: a nudge
    for i in range(3):
        det.observe("w1", "worker", "read", {"path": f"b{i}"}, {"ok": True})
    assert any(t["kind"] == "read_streak" for t in dec.pending())   # second strike: Lead trigger
    det.observe("w2", "worker", "bench", {}, {"ok": False, "class": "env_blocked", "error": "disk"})
    assert any(t["kind"] == "env_blocked" and t["infra"] for t in dec.pending())
    assert (tmp_path / "i.jsonl").exists()      # infra goes to the human inbox at once


def test_bundle_hash_and_args(tmp_path):
    files = {"manifest.json": b"{}", "overlays/vllm/x.py": b"A=2\n"}
    for rel, b in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b)
    assert hash_dir(tmp_path) == hash_files(files)
    args = merged_args(["--a", "1", "--flag", "--b", "x"], {"a": 2, "flag": None, "c": True})
    assert args == ["--a", "2", "--b", "x", "--c"]
    (tmp_path / "manifest.json").write_text(json.dumps({"config_delta": {"--a": 3}, "env": {"E": "1"},
                                                        "overlays": {"vllm/x.py": "overlays/vllm/x.py"}}))
    argv, env = serve_command("py", "m", ["--a", "1"], tmp_path, 1234, hook_dir=tmp_path / "hook")
    assert argv[:5] == ["py", "-m", "vllm.entrypoints.cli.main", "serve", "m"]
    assert "3" in argv and env["E"] == "1" and "vllm.x" in env["FORGE_OVERLAY_MAP"]


def test_normalize_catalog_recipe():
    r = normalize_recipe({"id": "bolt", "model": "m", "vllm": {"env": {"A": "1"}}, "serve": {"args": []},
                          "chat": {"chat_template_kwargs": {"enable_thinking": False}},
                          "variants": [{"name": "c1", "concurrency": 1, "max_tokens": 512,
                                        "metric": "decode_tok_s"}]}, "c1", "/p.jsonl")
    assert r["id"] == "bolt:c1" and r["serve"]["env"] == {"A": "1"}
    assert r["workload"]["bench_prompts"] == "/p.jsonl"
    assert r["workload"]["chat"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_mechanic_compare_and_propose(tmp_path):
    base = mechanic.current(tmp_path)

    def row(pb, rid, status, d=0.0):
        return {"playbook": pb, "recipe_id": rid, "done": {"status": status}, "delta_pct": d,
                "gate_verdicts": [{"verdict": "win"}] if status == "win" else [], "n_benches": 3}
    rows = [row(base, "a", "no_win"), row("new", "a", "win", 4), row(base, "b", "no_win"), row("new", "b", "no_win")]
    v = mechanic.compare(base, "new", rows, tmp_path)
    assert v["verdict"] == "promote" and v["pairs"] == 2
    assert mechanic.current(tmp_path) == "new"
    rows.append(row(base, "c", "win", 3))
    rows.append(row("new", "c", "no_win"))
    assert mechanic.compare(base, "new", rows, tmp_path)["verdict"] == "reject"

    def mock(model, messages, tools, role):
        return {"content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "propose_playbook", "arguments": json.dumps({
                "worker": playbook.DEFAULT["worker"] + "\n- Report blockers early.",
                "knobs": {"verifier_bar_pct": 99}, "rationale": "read streaks"})}}]}
    client = LLMClient(profile("mock"), Meter(None), mock=mock)
    h = mechanic.propose([], playbook.content_hash(playbook.DEFAULT), tmp_path, client=client)
    pb = playbook.load(h, tmp_path / "playbooks")
    assert pb["knobs"]["verifier_bar_pct"] == 10.0 and pb["parent"] == playbook.content_hash(playbook.DEFAULT)

    def leaky(model, messages, tools, role):
        return {"content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "propose_playbook", "arguments": json.dumps({"worker": "Always set --max-num-seqs 4.",
                                                                 "rationale": "x"})}}]}
    client = LLMClient(profile("mock"), Meter(None), mock=leaky)
    assert mechanic.propose([], "base", tmp_path, client=client) is None
    assert any(r["event"] == "rejected_proposal" for r in
               map(json.loads, (tmp_path / "n1/log.jsonl").read_text().splitlines()))


def test_unknown_args_and_aliases():
    reg = Registry()
    reg.add(Tool("read", "t", {"type": "object", "properties": {"path": {"type": "string"},
                                                                 "lines": {"type": "integer"}}},
                 lambda ctx, path, lines=10: {"lines": lines}))
    assert reg.dispatch(None, "worker", "read", {"path": "a", "limit": "40"})["lines"] == 40
    r = reg.dispatch(None, "worker", "read", {"path": "a", "bogus": 1})
    assert r["class"] == "agent_error" and "bogus" in r["error"]


def test_remote_bench_classifies_failures(tmp_path):
    from forge.tools.bench import RemoteBenchRunner
    from forge.tools.executor import ExecResult

    class Ex:
        def __init__(self, out):
            self.out = out

        def workspace(self, mid):
            return str(tmp_path)

        def write_text(self, p, t):
            pass

        def run(self, *a, **k):
            return ExecResult(0, "BENCH_RESULT " + json.dumps(self.out), "", 1.0)

    class M:
        id = "m"
        python = "py"
        bench_prompts = "p.jsonl"
        recipe = {"model": "x", "serve": {"args": []}, "workload": {}}

    with pytest.raises(EnvBlocked):
        RemoteBenchRunner(Ex({"ok": False, "infra": True, "error": "boom"}), "/r", "py").run(M())
    with pytest.raises(ToolError):
        RemoteBenchRunner(Ex({"ok": False, "error": "bad flag"}), "/r", "py").run(M(), config_delta={"--x": 1})
    ok = RemoteBenchRunner(Ex({"ok": True, "reference": {"value": 1}}), "/r", "py").run(M())
    assert ok["reference"]["value"] == 1
