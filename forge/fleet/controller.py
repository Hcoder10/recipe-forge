"""Fleet controller: a queue of missions (recipe x machine) -> free machines -> monitor -> rows -> leaderboard,
with Version N+1 between waves.

Runs on the laptop (`forge run-fleet`). Mission runners run on the rig as detached jobs
(`python -m forge.agents run --spec ...`); each mission's executor targets its machine (the rig itself or a box).
State (mirrored to the rig's ~/forge/state for the dashboard):
  fleet/queue.json      [{mission_id, recipe_id, machine, wave, playbook, status, ...}]
  fleet/machines.json   snapshot of machines + box setup status
  fleet/alerts.jsonl    infra problems
  results/rows/<mission_id>.json, results/leaderboard.json, results/leaderboard.md

Waves: wave 1 runs the "train" recipes with the current playbook. After a wave, the Mechanic proposes a
revision from its traces; the next wave's held-out (not yet run) recipes run under BOTH the old and the new
playbook where machines allow, and compare() promotes or rejects it.
"""
from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

from forge.fleet.executor import ExecutorError, executor_for
from forge.fleet.machines import all_machines, get_machine, python_for, state_dir
from forge.recipes import fits, get as get_recipe

RIG_STATE = "/home/cube/forge/state"
RIG_ROOT = "/home/cube/forge"
SECRETS = "/home/cube/forge_secrets.env"
VRAM_ASK = {"rtxpro6000_95g": 90, "rtx5090_32g": 29}


def now() -> float:
    return time.time()


def _read(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


class Controller:
    def __init__(self, defaults: dict | None = None, log=print):
        self.dir = state_dir()
        self.qpath = self.dir / "queue.json"
        self.log = log
        self.defaults = {"time_box_min": 75, "budget_usd": 15, "profile": "crusoe", "slots_per_machine": 1,
                         "team": {"workers": 2}, **(defaults or {})}
        self.rig = executor_for(get_machine("rig"))

    # ------------------------------------------------------------------ queue
    def queue(self) -> list[dict]:
        return _read(self.qpath, [])

    def save_queue(self, q: list[dict]) -> None:
        _write(self.qpath, q)

    def add(self, recipe_id: str, machine: str, wave: int = 1, playbook: str | None = None, **over) -> dict:
        m = get_machine(machine)
        r = get_recipe(recipe_id, m.get("gpu_class"))
        if not fits(r, m["gpu_class"]):
            raise ValueError(f"{recipe_id} does not fit {machine} ({m['gpu_class']})")
        if r["base_id"] not in (m.get("recipes") or []):
            raise ValueError(f"{machine} is not set up for {r['base_id']} (has {m.get('recipes')})")
        q = self.queue()
        mid = f"{time.strftime('%m%d-%H%M%S')}-{r['id'].replace(':', '-').replace('.', '-')}-{machine}"[:80]
        item = {"mission_id": mid, "recipe_id": r["id"], "machine": machine, "wave": wave, "playbook": playbook,
                "status": "queued", "queued_at": now(), **over}
        q.append(item)
        self.save_queue(q)
        return item

    def spec_for(self, item: dict) -> dict:
        m = get_machine(item["machine"])
        r = get_recipe(item["recipe_id"], m.get("gpu_class"))
        root = m.get("forge_root", "/root/forge")
        bp = r["workload"]["bench_prompts"]
        spec = {
            "mission_id": item["mission_id"], "recipe": r, "machine": m, "python": python_for(m, r),
            "bench_prompts": bp if bp.startswith("/") else f"{root}/{bp}",
            "time_box_min": item.get("time_box_min", self.defaults["time_box_min"]),
            "budget_usd": item.get("budget_usd", self.defaults["budget_usd"]),
            "team": item.get("team", self.defaults["team"]), "playbook": item.get("playbook"),
            "profile": item.get("profile", self.defaults["profile"]),
            "goal": item.get("goal") or f"Make a verified, lossless speedup of {r['metric'] if 'metric' in r else r['workload']['metric']} "
                                         f"for NVIDIA's published recipe {r['id']} on {m['gpu_class']}.",
            "recipe_files_dir": f"{root}/state/recipe_files/{r['base_id']}",
            "wave": item.get("wave"),
            # the bench job's VRAM ask must fit the card (it is exclusive on a timing GPU anyway)
            "bench_vram_gb": item.get("bench_vram_gb", VRAM_ASK.get(m.get("gpu_class"), 90)),
        }
        return spec

    # ------------------------------------------------------------------ dispatch / monitor
    def machine_ready(self, m: dict) -> bool:
        if m.get("kind") != "vast":
            return True
        min_abi = int(os.environ.get("FORGE_MIN_LANDLOCK_ABI", "6"))
        if m.get("landlock_abi") is not None and m["landlock_abi"] < min_abi:
            return False      # core's sandbox cannot start there
        return bool(m.get("ready"))

    def dispatch(self) -> list[str]:
        q = self.queue()
        ms = all_machines()
        busy: dict[str, int] = {}
        for it in q:
            if it["status"] == "running":
                busy[it["machine"]] = busy.get(it["machine"], 0) + 1
        started = []
        for it in q:
            if it["status"] != "queued":
                continue
            m = ms.get(it["machine"])
            if not m or not self.machine_ready(m):
                continue
            slots = m.get("slots", self.defaults["slots_per_machine"])
            if busy.get(it["machine"], 0) >= slots:
                continue
            try:
                self.start(it)
                busy[it["machine"]] = busy.get(it["machine"], 0) + 1
                started.append(it["mission_id"])
            except (ExecutorError, OSError) as e:
                it["status"] = "queued"
                it["last_error"] = str(e)[:300]
                self.alert("dispatch_failed", it["mission_id"], str(e))
        self.save_queue(q)
        return started

    def start(self, it: dict) -> None:
        spec = self.spec_for(it)
        spath = f"{RIG_STATE}/fleet/specs/{it['mission_id']}.json"
        self.rig.write_text(spath, json.dumps(spec, indent=1, default=str))
        py = get_machine("rig")["python"]
        cmd = (f"set -a; [ -f {SECRETS} ] && . {SECRETS}; set +a; "
               f"exec {py} -m forge.agents run --spec {shlex.quote(spath)}")
        self.rig.start_job(f"mission-{it['mission_id']}", cmd, cwd=RIG_ROOT,
                           env={"PYTHONPATH": RIG_ROOT, "FORGE_ROOT": RIG_ROOT, "FORGE_STATE": RIG_STATE})
        it.update(status="running", started_at=now())
        self.log(f"started {it['mission_id']} on {it['machine']}")

    def monitor(self) -> list[str]:
        q = self.queue()
        finished = []
        for it in q:
            if it["status"] != "running":
                continue
            try:
                st = self.rig.job_status(f"mission-{it['mission_id']}")
            except ExecutorError as e:
                self.alert("rig_unreachable", it["mission_id"], str(e))
                continue
            if st["state"] == "running":
                continue
            row = None
            try:
                row = json.loads(self.rig.read_text(f"{RIG_STATE}/missions/{it['mission_id']}/row.json"))
            except (FileNotFoundError, ValueError):
                pass
            it["ended_at"] = now()
            it["rc"] = st.get("rc")
            if row:
                row.setdefault("machine", it["machine"])
                row["wave"] = it.get("wave")
                row["playbook"] = row.get("playbook") or it.get("playbook")
                self.rig.write_text(f"{RIG_STATE}/results/rows/{it['mission_id']}.json", json.dumps(row, indent=1, default=str))
                _write(self.dir.parent / "results" / "rows" / f"{it['mission_id']}.json", row)
                it["status"] = "done"
                it["row"] = {k: row.get(k) for k in ("baseline", "best", "delta_pct", "metric") if k in row}
                it["row"]["done"] = (row.get("done") or {}).get("status")
            else:
                it["status"] = "failed"
                it["error_tail"] = self.rig.tail(f"mission-{it['mission_id']}", 30)[-2000:]
                self.alert("mission_failed", it["mission_id"], it["error_tail"][-300:])
            finished.append(it["mission_id"])
        self.save_queue(q)
        if finished:
            self.leaderboard()
        return finished

    def stop(self, mission_id: str | None = None) -> list[str]:
        """Stop running missions this controller started (all, or one)."""
        q = self.queue()
        out = []
        for it in q:
            if it["status"] == "running" and (mission_id is None or it["mission_id"] == mission_id):
                self.rig.stop_job(f"mission-{it['mission_id']}")
                it["status"] = "stopped"
                out.append(it["mission_id"])
            elif it["status"] == "queued" and mission_id is None:
                it["status"] = "cancelled"
        self.save_queue(q)
        return out

    def alert(self, kind: str, subject: str, detail: str) -> None:
        p = self.dir / "alerts.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": now(), "source": "fleet", "kind": kind, "subject": subject,
                                "detail": detail[:1000]}) + "\n")

    # ------------------------------------------------------------------ leaderboard
    def rows(self) -> list[dict]:
        d = self.dir.parent / "results" / "rows"
        return [_read(p, {}) for p in sorted(d.glob("*.json"))] if d.is_dir() else []

    def leaderboard(self) -> dict:
        best: dict[tuple, dict] = {}
        for r in self.rows():
            key = (r.get("recipe_id"), r.get("gpu_class") or r.get("machine"))
            cur = best.get(key)
            won = (r.get("done") or {}).get("status") == "win"
            score = (r.get("delta_pct") or 0) if won else -1e9
            entry = {"recipe_id": key[0], "gpu_class": key[1], "machine": r.get("machine"),
                     "metric": r.get("metric"), "baseline": r.get("baseline"),
                     "best": r.get("best") if won else r.get("baseline"),
                     "delta_pct": r.get("delta_pct") if won else 0.0, "verified": won,
                     "change": r.get("change_summary") if won else None,
                     "evidence_ids": r.get("evidence_ids") if won else [],
                     "gate_verdicts": r.get("gate_verdicts"), "mission_id": r.get("mission_id"),
                     "missions": 1 + (cur["missions"] if cur else 0), "_score": score}
            if cur is None or score > cur["_score"]:
                best[key] = entry
            else:
                cur["missions"] = entry["missions"]
        rows = sorted(best.values(), key=lambda e: (-(e["delta_pct"] or 0), e["recipe_id"] or ""))
        for e in rows:
            e.pop("_score", None)
        lb = {"updated": now(), "rows": rows}
        res = self.dir.parent / "results"
        _write(res / "leaderboard.json", lb)
        md = ["| recipe | GPU | metric | NVIDIA baseline | best verified | delta | change | proof |",
              "|---|---|---|---|---|---|---|---|"]
        for e in rows:
            md.append(f"| {e['recipe_id']} | {e['gpu_class']} | {e['metric'] or ''} | {_n(e['baseline'])} | "
                      f"{_n(e['best'])} | {(e['delta_pct'] or 0):+.1f}% | {(e['change'] or '-')[:80]} | "
                      f"{', '.join(e['evidence_ids'] or []) or '-'} |")
        (res / "leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        try:
            self.rig.write_text(f"{RIG_STATE}/results/leaderboard.json", json.dumps(lb, indent=1, default=str))
            self.rig.write_text(f"{RIG_STATE}/results/leaderboard.md", "\n".join(md) + "\n")
        except (ExecutorError, OSError) as e:
            self.alert("rig_unreachable", "leaderboard", str(e))
        return lb

    # ------------------------------------------------------------------ Version N+1
    def mechanic(self, fn: str, *args) -> object:
        py = get_machine("rig")["python"]
        code = (f"import json,sys; from forge.agents import mechanic as M; "
                f"print('MECH '+json.dumps(M.{fn}(*json.loads(sys.argv[1])), default=str))")
        r = self.rig.run(f"set -a; [ -f {SECRETS} ] && . {SECRETS}; set +a; {py} -c {shlex.quote(code)} "
                         f"{shlex.quote(json.dumps(list(args), default=str))}",
                         cwd=RIG_ROOT, env={"PYTHONPATH": RIG_ROOT, "FORGE_ROOT": RIG_ROOT, "FORGE_STATE": RIG_STATE},
                         timeout_s=1800)
        line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("MECH ")), None)
        if not line:
            raise RuntimeError(f"mechanic.{fn} failed: {(r.stderr or r.stdout)[-500:]}")
        return json.loads(line[5:])

    def wave_done(self, wave: int) -> bool:
        items = [i for i in self.queue() if i.get("wave") == wave]
        return bool(items) and all(i["status"] in ("done", "failed", "stopped", "cancelled") for i in items)

    def plan_path(self) -> Path:
        return self.dir / "n1_plan.json"

    def set_n1_plan(self, after_wave: int, held_out: list[tuple[str, str]], time_box_min: int) -> dict:
        plan = {"after_wave": after_wave, "held_out": held_out, "time_box_min": time_box_min, "started": False}
        _write(self.plan_path(), plan)
        return plan

    def n1_tick(self) -> dict | None:
        """When the planned wave is finished, run Version N+1 once: propose, then queue the paired wave."""
        plan = _read(self.plan_path(), None)
        if not plan or plan.get("started") or not self.wave_done(plan["after_wave"]):
            return None
        plan["started"] = now()
        _write(self.plan_path(), plan)
        log = self.next_wave(plan["after_wave"], [tuple(x) for x in plan["held_out"]],
                             time_box_min=plan.get("time_box_min"))
        plan["log"] = log
        _write(self.plan_path(), plan)
        self.log(f"N+1: wave {plan['after_wave'] + 1} queued, proposed={log.get('proposed')}")
        return log

    def next_wave(self, wave: int, held_out: list[tuple[str, str]], time_box_min: int | None = None) -> dict:
        """After `wave` finishes: propose N+1 from its traces, then queue held-out (recipe, machine) pairs under
        the current and the proposed playbook (paired)."""
        mids = [i["mission_id"] for i in self.queue() if i.get("wave") == wave and i["status"] == "done"]
        base = self.mechanic("current")
        new = self.mechanic("propose", mids, base) if mids else None
        log = {"t": now(), "wave": wave, "base": base, "proposed": new, "missions": mids}
        extra = {"time_box_min": time_box_min} if time_box_min else {}
        for rid, mname in held_out:
            self.add(rid, mname, wave=wave + 1, playbook=base, arm="old", **extra)
            if new:
                self.add(rid, mname, wave=wave + 1, playbook=new, arm="new", **extra)
        with open(self.dir / "waves.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        return log

    def judge_wave(self, wave: int) -> dict | None:
        items = [i for i in self.queue() if i.get("wave") == wave and i["status"] == "done"]
        old = {i.get("playbook") for i in items if i.get("arm") == "old"}
        new = {i.get("playbook") for i in items if i.get("arm") == "new"}
        if len(old) != 1 or len(new) != 1:
            return None
        rows = [_read(self.dir.parent / "results" / "rows" / f"{i['mission_id']}.json", {}) for i in items]
        return self.mechanic("compare", old.pop(), new.pop(), rows)

    # ------------------------------------------------------------------ snapshot for the dash
    def snapshot(self, rentals: dict | None = None) -> dict:
        ms = all_machines()
        snap = {"t": now(), "machines": [], "rentals": rentals}
        q = self.queue()
        for name, m in ms.items():
            snap["machines"].append({
                "name": name, "kind": m.get("kind", "static"), "gpu_class": m.get("gpu_class"),
                "building": m.get("building"), "recipes": m.get("recipes"), "ready": self.machine_ready(m),
                "status": m.get("status"), "usd_per_hour": m.get("usd_per_hour"),
                "running": [i["mission_id"] for i in q if i["machine"] == name and i["status"] == "running"],
                "queued": sum(1 for i in q if i["machine"] == name and i["status"] == "queued")})
        _write(self.dir / "machines.json", snap)
        try:
            self.rig.write_text(f"{RIG_STATE}/fleet/machines.json", json.dumps(snap, indent=1, default=str))
            self.rig.write_text(f"{RIG_STATE}/fleet/queue.json", json.dumps(q, indent=1, default=str))
        except (ExecutorError, OSError):
            pass
        return snap


def _n(x) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else "-"
