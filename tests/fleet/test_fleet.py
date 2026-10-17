"""Fleet unit tests. Run on Linux (the rig): python -m pytest tests/fleet -q"""
from __future__ import annotations

import json
import os
import time

import pytest

from forge.recipes import apply_flags, bases, get, load_all, load_prompts


def test_catalog_loads_and_cites_sources():
    bs = bases()
    assert len(bs) >= 10
    for b in bs:
        assert b["serve"]["source"].startswith("https://")
        assert b["serve"]["quoted"].strip()
        assert b["variants"]


def test_variants_respect_published_max_num_seqs():
    for r in load_all():
        a = r["serve"]["published_args"]
        if "--max-num-seqs" in a:
            assert r["workload"]["concurrency"] <= int(a[a.index("--max-num-seqs") + 1]), r["id"]


def test_ids_and_lookup():
    r = get("nano-9b-v2:c8")
    assert r["id"] == "nano-9b-v2:c8" and r["workload"]["concurrency"] == 8
    assert r["workload"]["metric"] == "output_tok_s"
    assert get("nano-9b-v2.c8")["id"] == "nano-9b-v2:c8"
    assert get("bolt-74b-lilicorr")["variant"] == "c1"


def test_adapt_only_for_its_gpu_class():
    plain = get("supervl-120b-nvfp4-lilicorr:c1")
    pro = get("supervl-120b-nvfp4-lilicorr:c1", "rtxpro6000_95g")
    a = pro["serve"]["args"]
    assert a[a.index("--gpu-memory-utilization") + 1] == "0.92"
    assert plain["serve"]["args"] == plain["serve"]["published_args"]
    assert pro["serve"]["published_args"] == plain["serve"]["published_args"]
    assert pro["serve"]["adapted"]["why"]


def test_apply_flags():
    assert apply_flags(["--a", "1", "--b"], {"--a": "2", "--c": "3"}) == ["--a", "2", "--b", "--c", "3"]


def test_public_prompts_deterministic():
    ps = load_prompts()
    assert 12 <= len(ps) <= 32
    assert len({p["id"] for p in ps}) == len(ps)
    assert all(p["messages"][0]["role"] == "user" for p in ps)


@pytest.mark.skipif(os.name == "nt", reason="executor runs bash on Linux targets")
def test_local_executor_run_jobs_and_transfer(tmp_path, monkeypatch):
    from forge.fleet.executor import FleetExecutor
    ex = FleetExecutor({"name": "t", "forge_root": str(tmp_path / "root")})
    r = ex.run(["echo", "hi there"])
    assert r.ok and r.stdout.strip() == "hi there"
    r = ex.run("exit 3")
    assert r.rc == 3 and r.infra_error is None
    r = ex.run("sleep 5", timeout_s=1)
    assert r.rc == 124
    ex.write_text(str(tmp_path / "a" / "b.txt"), "x\n")
    assert ex.exists(str(tmp_path / "a" / "b.txt")) and ex.read_text(str(tmp_path / "a" / "b.txt")) == "x\n"
    assert ex.list_files(str(tmp_path / "a")) == ["b.txt"]
    ex.start_job("j1", "echo start; sleep 1; echo end; exit 7")
    for _ in range(50):
        if ex.job_status("j1")["state"] == "exited":
            break
        time.sleep(0.2)
    st = ex.job_status("j1")
    assert st["state"] == "exited" and st["rc"] == 7
    text, off = ex.read_log("j1")
    assert "start" in text and "end" in text and off == len(text.encode())
    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    (src / "d" / "f.py").write_text("print(1)\r\n")
    ex.put_tree(src, str(tmp_path / "dst"))
    assert (tmp_path / "dst" / "d" / "f.py").read_bytes() == b"print(1)\n"
    ex.get_tree(str(tmp_path / "dst" / "d"), tmp_path / "back")
    assert (tmp_path / "back" / "d" / "f.py").exists()
    assert set(ex.stat_tree(str(tmp_path / "dst"))) == {"d/f.py"}


@pytest.mark.skipif(os.name == "nt", reason="needs Linux")
def test_gpu_job_without_scheduler_is_infra_error(tmp_path, monkeypatch):
    from forge.fleet.executor import FleetExecutor
    monkeypatch.setenv("FORGE_SCHED_PORT", "1")       # nothing listens there
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ex = FleetExecutor({"name": "t", "forge_root": repo, "python": "python3"})
    r = ex.run("echo should-not-run", kind="bench", vram_gb=1, timeout_s=30)
    assert r.infra_error and "should-not-run" not in r.stdout


def test_leaderboard_only_counts_gate_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_STATE", str(tmp_path / "state"))
    from forge.fleet import controller as C

    class _NoRig:
        def write_text(self, *a, **k):
            pass

    c = C.Controller.__new__(C.Controller)
    c.dir = tmp_path / "state" / "fleet"
    c.dir.mkdir(parents=True)
    c.rig = _NoRig()
    rows = tmp_path / "state" / "results" / "rows"
    rows.mkdir(parents=True)
    (rows / "a.json").write_text(json.dumps({"mission_id": "a", "recipe_id": "x:c1", "gpu_class": "g", "baseline": 100,
                                             "best": 0, "delta_pct": 9.0, "done": {"status": "no_win"}}))
    (rows / "b.json").write_text(json.dumps({"mission_id": "b", "recipe_id": "x:c1", "gpu_class": "g", "baseline": 100,
                                             "best": 104, "delta_pct": 4.0, "done": {"status": "win"},
                                             "evidence_ids": ["g1"]}))
    lb = c.leaderboard()
    assert len(lb["rows"]) == 1
    e = lb["rows"][0]
    assert e["delta_pct"] == 4.0 and e["verified"] and e["missions"] == 2 and e["evidence_ids"] == ["g1"]
