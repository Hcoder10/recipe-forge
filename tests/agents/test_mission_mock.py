"""End to end: a mission with the mock brain, a local executor, the mock bench and the mock gate."""
import json
from pathlib import Path

from forge.agents.mission import MissionRunner
from forge.agents.mock import MockBrain
from forge.tools.bench import MockBenchRunner
from forge.tools.bundle import canon
from forge.tools.executor import LocalExecutor
from forge.tools.gateclient import MockGate

RECIPE = {"id": "toy-c1", "model": "nvidia/toy", "engine": "vllm",
          "serve": {"args": ["--max-model-len", "8192"], "source": "https://example.invalid/recipe"},
          "workload": {"concurrency": 1, "max_tokens": 64, "metric": "decode_tok_s"}}


def make_spec(tmp: Path, **kw):
    sp = tmp / "site"
    (sp / "vllm").mkdir(parents=True)
    (sp / "vllm" / "x.py").write_text("A = 1\n")
    prompts = tmp / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n')
    spec = {"mission_id": "m-test", "recipe": RECIPE, "machine": {"name": "local", "gpu_class": "toy"},
            "python": "python", "site_packages": str(sp), "bench_prompts": str(prompts), "time_box_min": 2,
            "budget_usd": 5, "team": {"workers": 2, "lead_every_s": 0.2}, "profile": "mock", "goal": "",
            "gate_poll_s": 0.1, "tick_s": 0.1, "join_timeout_s": 5}
    spec.update(kw)
    return spec


def key(delta):
    return canon({"d": delta, "e": None, "b": None})


def test_mock_mission_win(tmp_path):
    spec = make_spec(tmp_path, require_confirm=False)
    bench = MockBenchRunner(noise=0.0, fixed_effects={key({"--seed": 1}): 0.08, key({"--seed": 2}): -0.02})
    gate = MockGate(tmp_path / "gate.jsonl", verdict_fn=lambda b: ("win", 6.5))
    r = MissionRunner(spec, executor=LocalExecutor(tmp_path / "target"), bench=bench, gate=gate,
                      mock_llm=MockBrain(), root=tmp_path / "state")
    row = r.run()
    assert row["done"]["status"] == "win", row
    assert row["delta_pct"] == 6.5
    assert row["change_summary"]["config_delta"] == {"--seed": 1}
    assert row["baseline"] == 100.0
    ev = [json.loads(l) for l in (tmp_path / "state/missions/m-test/events.jsonl").read_text().splitlines()]
    kinds = {e["kind"] for e in ev}
    assert {"mission_start", "done_locked", "gate_request", "gate_review", "gate_result", "decision",
            "mission_end"} <= kinds
    assert row["ended"].startswith("lead stop")
    assert row["cost_usd"] > 0
    assert not row["errors"], row["errors"]
    # shared brain got facts from both workers
    facts = (tmp_path / "state/brain/facts.jsonl").read_text().splitlines()
    assert len(facts) == 2


def test_mock_mission_confirm_required(tmp_path):
    """A candidate win without a confirm row does not meet the locked criteria."""
    spec = make_spec(tmp_path, require_confirm=True, time_box_min=0.15)
    bench = MockBenchRunner(noise=0.0, fixed_effects={key({"--seed": 1}): 0.08})
    gate = MockGate(tmp_path / "gate.jsonl", verdict_fn=lambda b: ("win", 6.5))
    row = MissionRunner(spec, executor=LocalExecutor(tmp_path / "target"), bench=bench, gate=gate,
                        mock_llm=MockBrain(), root=tmp_path / "state").run()
    assert row["done"]["status"] == "no_win"
    assert row["delta_pct"] == 0.0
    assert any(v["verdict"] == "win" for v in row["gate_verdicts"])
