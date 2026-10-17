"""Reader tests: fixtures render every panel, missing/half-written files never raise, tails are incremental."""
import json
import os
import time

from forge.dash import fixtures
from forge.dash.reader import JsonlTail, Reader, build_leaderboard


def _append(path, *recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_empty_root_gives_empty_panels(tmp_path):
    s = Reader(str(tmp_path / "nothing-here")).snapshot()
    assert s["fixture"] is False
    assert s["leaderboard"] == [] and s["missions"] == [] and s["gates"] == []
    assert s["brain"]["total"] == 0 and s["n1"]["versions"] == []
    assert s["spend"]["crusoe"]["total"] == 0 and s["spend"]["vast"]["total_usd"] == 0
    json.dumps(s)


def test_fixture_fills_every_panel(tmp_path):
    root = str(tmp_path / "fx")
    fixtures.write(root)
    s = Reader(root).snapshot()
    json.dumps(s)
    assert s["fixture"] is True
    statuses = {r["status"] for r in s["leaderboard"]}
    assert {"verified", "running", "no-gain", "queued"} <= statuses
    top = s["leaderboard"][0]
    assert top["status"] == "verified" and top["delta_pct"] > 0 and top["verdict_id"] and top["lossless"] is True
    live = [m for m in s["missions"] if m["status"] == "running"]
    assert len(live) == 5
    assert all(m["team"] and m["recipe"] and m["machine"] and m["deadline"] for m in live)
    assert any(a.get("tool") for m in live for a in m["team"])
    assert any(a.get("lane") for m in live for a in m["team"])
    ended = [m for m in s["missions"] if m["status"] != "running"]
    assert {m["status"] for m in ended} == {"win", "no_win"}
    assert any(g["running"] for g in s["gates"]) and any(g["queued"] for g in s["gates"])
    assert sum(g["n_verdicts"] for g in s["gates"]) >= 5
    assert s["decisions"] and s["alerts"]
    assert {"agent", "infra"} <= {a["fault"] for a in s["alerts"]}
    assert s["brain"]["total"] == 30 and "measured" in s["brain"]["by_status"]
    assert [v["status"] for v in s["n1"]["versions"]] == ["promoted", "promoted", "rejected", "candidate"]
    assert s["spend"]["crusoe"]["total"] > 0 and set(s["spend"]["crusoe"]["by_role"]) >= {"lead", "worker"}
    assert s["spend"]["vast"]["burn_usd_per_hr"] > 0 and len(s["spend"]["vast"]["machines"]) == 5


def test_jsonl_tail_partial_lines_and_truncation(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_bytes(b'{"a": 1}\n{"a": 2')
    t = JsonlTail(str(p))
    recs, reset = t.poll()
    assert [r["a"] for r in recs] == [1] and not reset
    with open(p, "ab") as f:
        f.write(b'}\nnot json\n{"a": 3}\n')
    recs, _ = t.poll()
    assert [r["a"] for r in recs] == [2, 3] and t.bad_lines == 1
    assert t.poll() == ([], False)
    p.write_bytes(b'{"a": 9}\n')  # rewritten shorter: start over
    recs, reset = t.poll()
    assert reset and [r["a"] for r in recs] == [9]


def test_events_are_folded_incrementally(tmp_path):
    root = tmp_path
    ev = root / "missions" / "mx" / "events.jsonl"
    os.makedirs(ev.parent)
    (ev.parent / "spec.json").write_text(json.dumps(
        {"mission_id": "mx", "recipe": {"id": "r1"}, "machine": {"name": "rig", "gpu_class": "G"},
         "time_box_min": 75, "budget_usd": 15}))
    t0 = time.time() - 60
    _append(str(ev), {"t": t0, "kind": "mission_start", "team": [{"name": "lead", "role": "lead"}]})
    r = Reader(str(root))
    m = r.snapshot()["missions"][0]
    assert (m["recipe"], m["machine"], m["gpu_class"], m["status"]) == ("r1", "rig", "G", "running")
    assert abs(m["deadline"] - (t0 + 75 * 60)) < 1e-6
    _append(str(ev), {"t": t0 + 1, "kind": "tool_call", "agent": "w1", "role": "worker", "tool": "bench"},
            {"t": t0 + 2, "kind": "llm_call", "agent": "w1", "role": "worker", "cost_usd": 0.25},
            {"t": t0 + 3, "kind": "decision", "by": "lead", "decision": "replan", "plan": "p"},
            {"t": t0 + 4, "kind": "detector", "agent": "w1", "signal": "repeat_call", "infra": False},
            {"t": t0 + 5, "kind": "tool_bug", "agent": "w1", "detail": "x"},
            {"t": t0 + 6, "kind": "gate_submitted", "gate_id": "g1"})
    s = r.snapshot()
    m = s["missions"][0]
    w1 = next(a for a in m["team"] if a["name"] == "w1")
    assert w1["tool"] == "bench" and w1["step"] == 1 and m["usd"] == 0.25
    assert s["spend"]["crusoe"]["by_role"] == {"worker": 0.25}
    assert s["decisions"][0]["decision"] == "replan" and s["decisions"][0]["agent"] == "lead"
    assert [a["fault"] for a in s["alerts"]] == ["infra", "agent"]
    assert s["gates"][0]["machine"] == "rig" and s["gates"][0]["queued"][0]["request_id"] == "g1"
    _append(str(ev), {"t": t0 + 7, "kind": "gate_result", "gate_id": "g1", "verdict": "win", "delta_pct": 7.5})
    s = r.snapshot()
    g = s["gates"][0]
    assert g["queued"] == [] and g["n_wins"] == 1 and g["verdicts"][0]["delta_pct"] == 7.5
    row = {"mission_id": "mx", "recipe_id": "r1", "gpu_class": "G", "baseline": 100.0, "best": 107.5,
           "delta_pct": 7.5, "gate_verdicts": [{"gate_id": "g1", "verdict": "win", "delta_pct": 7.5}],
           "change_summary": "c", "done": {"status": "win"}}
    _append(str(ev), {"t": t0 + 8, "kind": "mission_end", "row": row})
    s = r.snapshot()
    assert s["missions"][0]["status"] == "win"
    lb = s["leaderboard"][0]
    assert (lb["status"], lb["best_tok_s"], lb["delta_pct"], lb["verdict_id"], lb["lossless"]) == \
        ("verified", 107.5, 7.5, "g1", True)


def test_half_written_json_does_not_raise(tmp_path):
    (tmp_path / "missions" / "m1").mkdir(parents=True)
    (tmp_path / "missions" / "m1" / "spec.json").write_text('{"mission_id": "m1", "rec')
    (tmp_path / "results" / "gate" / "rig").mkdir(parents=True)
    (tmp_path / "results" / "gate" / "rig" / "queue.json").write_text("{")
    (tmp_path / "fleet").mkdir()
    (tmp_path / "fleet" / "machines.json").write_text("[{")
    s = Reader(str(tmp_path)).snapshot()
    json.dumps(s)
    assert s["sources"]["gate_queue"]["err"]


def test_state_subdir_is_read(tmp_path):
    _append(str(tmp_path / "state" / "brain" / "facts.jsonl"),
            {"fact_id": "f1", "claim": "c", "status": "measured", "t": 1})
    assert Reader(str(tmp_path)).snapshot()["brain"]["by_status"] == {"measured": 1}


def test_leaderboard_keeps_best_verified_and_ignores_zero_best():
    rows = [
        {"mission_id": "a", "recipe_id": "r", "gpu_class": "G", "baseline": 100, "best": 110,
         "done": {"status": "win"}, "ended": 1},
        {"mission_id": "b", "recipe_id": "r", "gpu_class": "G", "baseline": 100, "best": 0,
         "done": {"status": "no_win"}, "ended": 2},
        {"mission_id": "c", "recipe_id": "r", "gpu_class": "G", "baseline": 100, "best": 105,
         "done": {"status": "win"}, "ended": 3},
    ]
    missions = [{"id": "d", "status": "running", "recipe": "r", "gpu_class": "G", "machine": "m"}]
    (c,) = build_leaderboard([], rows, [], missions)
    assert (c["status"], c["best_tok_s"], c["delta_pct"], c["mission"], c["running"]) == ("verified", 110, 10.0, "a", 1)
    (q,) = build_leaderboard([{"id": "r2", "hardware": ["G"]}], [], [], [])
    assert q["status"] == "queued" and q["best_tok_s"] is None
