"""Mission runner: one team (Lead, Workers, Verifier) per mission = recipe x machine x GPU x time box.

    row = run_mission(spec)          # blocking; spec documented in docs/interfaces/agents.md

State per mission (runner side): <state>/missions/<id>/{events,usage,decisions,inbox,answers,notes,benches,
gate_requests}.jsonl, done.lock.json, spec.json, row.json. The workspace lives on the target machine.
"""
from __future__ import annotations

import itertools
import json
import posixpath
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from forge.agents import done as done_mod
from forge.agents import events, playbook as pb_mod
from forge.agents.agent import Agent
from forge.agents.brain import Brain
from forge.agents.config import TeamConfig, load_secrets, profile as get_profile, state_root
from forge.agents.decisions import Decisions, Inbox, Notes
from forge.agents.detector import Detector
from forge.agents.events import EventLog
from forge.agents.llm import BudgetExceeded, LLMClient, LLMError, Meter
from forge.agents.prompts import mission_brief, status_block
from forge.tools.builtin import build_registry
from forge.tools.registry import ToolError


# ------------------------------------------------------------------------------------------ state
@dataclass
class MissionState:
    id: str
    recipe: dict
    machine: dict
    python: str
    bench_prompts: str
    site_packages: str
    ws: str
    gpu_class: str
    time_box_min: float
    budget_usd: float
    n_workers: int
    criteria: dict
    done_hash: str
    executor: object = None
    meter: Meter | None = None
    readable_roots: list = field(default_factory=list)
    t_start: float = field(default_factory=time.time)
    benches: dict = field(default_factory=dict)
    bundles: dict = field(default_factory=dict)
    bench_log: Path | None = None
    _pkgs: set | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _bench_ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    @property
    def recipe_id(self) -> str:
        return self.recipe["id"]

    @property
    def metric(self) -> str:
        return self.criteria["metric"]

    @property
    def worker_names(self) -> list[str]:
        return [f"worker-{i + 1}" for i in range(self.n_workers)]

    def minutes_left(self) -> float:
        return max(0.0, self.time_box_min - (time.time() - self.t_start) / 60)

    def frac_used(self) -> float:
        return min(1.0, (time.time() - self.t_start) / 60 / self.time_box_min) if self.time_box_min else 1.0

    def record_bench(self, agent: str, row: dict) -> str:
        with self._lock:
            bid = f"B{next(self._bench_ids)}"
            self.benches[bid] = {"bench_id": bid, "by": agent, **row}
        if self.bench_log:
            events.append(self.bench_log, {"bench_id": bid, "by": agent, **row})
        return bid

    def reference(self) -> dict | None:
        with self._lock:
            for b in reversed(list(self.benches.values())):
                ref = (b.get("result") or {}).get("reference")
                if ref:
                    return {"bench_id": b["bench_id"], **ref}
        return None

    def bundle_dir(self, bid: str) -> str:
        return posixpath.join(self.ws, "bundles", bid)

    def register_bundle(self, bid: str, info: dict) -> None:
        with self._lock:
            self.bundles[bid] = info

    def overlay_root(self, agent: str) -> str:
        return posixpath.join(self.ws, "agents", agent, "overlays")

    def overlay_files(self, agent: str) -> list[str]:
        return self.executor.list_files(self.overlay_root(agent))

    def packages(self) -> set:
        if self._pkgs is None:
            self._pkgs = {x for x in self.executor.listdir(self.site_packages)
                          if not x.endswith((".dist-info", ".pth", ".egg-info"))}
        return self._pkgs


# ------------------------------------------------------------------------------------------ services
class GateQueue:
    """Worker gate requests wait here for the Verifier; approved ones go to the gate and are watched."""

    def __init__(self, runner):
        self.r = runner
        self.reqs: dict[str, dict] = {}
        self.ids = itertools.count(1)
        self.lock = threading.Lock()
        self.path = runner.dir / "gate_requests.jsonl"

    @property
    def bar(self) -> float:
        return float(self.r.playbook["knobs"].get("verifier_bar_pct", 3.0))

    def bar_text(self) -> str:
        return f"quick-bench delta_pct >= {self.bar}% on a bench of this exact bundle"

    def submit(self, ctx, bundle: str, claim: str, bench_id: str) -> dict:
        m = self.r.mission
        info = m.bundles.get(bundle)
        if not info:
            raise ToolError(f"unknown bundle {bundle!r}; use the id make_bundle returned")
        b = m.benches.get(bench_id)
        if not b:
            raise ToolError(f"unknown bench_id {bench_id!r}")
        if b.get("bundle") != bundle:
            raise ToolError(f"{bench_id} is not a bench of bundle {bundle}; run bench(bundle={bundle!r}) first")
        delta = (b.get("result") or {}).get("delta_pct")
        with self.lock:
            for r in self.reqs.values():
                if r["hash"] == info["hash"]:
                    return {"request_id": r["id"], "status": r["status"], "duplicate": True}
            for row in self.r.services.gate.ledger(recipe_id=m.recipe_id):
                if row.get("bundle_hash") == info["hash"]:
                    return {"status": "already gated", "gate_id": row["gate_id"], "verdict": row.get("verdict")}
            rid = f"R{next(self.ids)}"
            req = {"id": rid, "bundle": bundle, "hash": info["hash"], "claim": claim[:500], "bench_id": bench_id,
                   "delta_pct": delta, "by": ctx.agent, "status": "pending_review", "manifest": info["manifest"]}
            self.reqs[rid] = req
        events.append(self.path, {"event": "submit", **req})
        self.r.log("gate_request", request_id=rid, bundle=bundle, delta_pct=delta, by=ctx.agent, claim=claim[:300])
        self.r.wake("verifier")
        return {"request_id": rid, "status": "pending_review", "bar": self.bar_text()}

    def listing(self) -> list[dict]:
        with self.lock:
            out = []
            for r in self.reqs.values():
                b = self.r.mission.benches.get(r["bench_id"], {})
                out.append({k: r[k] for k in ("id", "bundle", "claim", "bench_id", "delta_pct", "by", "status")}
                           | {"manifest": r["manifest"], "bench": b.get("result"), "gate_id": r.get("gate_id")})
            return out

    def pending(self) -> list[str]:
        with self.lock:
            return [r["id"] for r in self.reqs.values() if r["status"] == "pending_review"]

    def review(self, ctx, request_id: str, approve: bool, reason: str) -> dict:
        with self.lock:
            req = self.reqs.get(request_id)
            if not req:
                raise ToolError(f"unknown request {request_id!r}")
            if req["status"] != "pending_review":
                return {"request_id": request_id, "status": req["status"], "note": "already reviewed"}
            req["status"] = "approved" if approve else "rejected"
            req["review"] = reason[:500]
        events.append(self.path, {"event": "review", "id": request_id, "approve": approve, "reason": reason[:500]})
        self.r.log("gate_review", request_id=request_id, approve=approve, reason=reason[:300], by=ctx.agent)
        worker = self.r.agents.get(req["by"])
        if not approve:
            if worker:
                worker.add_event_message(f"GATE REQUEST {request_id} rejected by the verifier: {reason}")
            return {"request_id": request_id, "status": "rejected"}
        m = self.r.mission
        try:
            g = self.r.services.gate.request(recipe_id=m.recipe_id, bundle_dir=m.bundle_dir(req["bundle"]),
                                             bundle_hash=req["hash"], mission_id=m.id, requester=req["by"],
                                             claim=req["claim"])
        except Exception as e:  # the gate service failing is infra
            with self.lock:
                req["status"] = "gate_error"
            self.r.decisions.trigger("env_blocked", f"gate request failed: {type(e).__name__}: {str(e)[:200]}",
                                     infra=True)
            return {"request_id": request_id, "status": "gate_error", "error": str(e)[:200]}
        with self.lock:
            req["status"] = "gated"
            req["gate_id"] = g["gate_id"]
        self.r.log("gate_submitted", request_id=request_id, gate_id=g["gate_id"], bundle=req["bundle"])
        if worker:
            worker.add_event_message(f"GATE REQUEST {request_id} approved; gate {g['gate_id']} queued.")
        threading.Thread(target=self._watch, args=(req,), daemon=True, name=f"gatewatch-{g['gate_id']}").start()
        return {"request_id": request_id, "status": "gated", "gate_id": g["gate_id"]}

    def _watch(self, req: dict) -> None:
        gid = req["gate_id"]
        while not self.r.finished.is_set() or self.r.wait_gates:
            try:
                st = self.r.services.gate.status(gid)
            except Exception:
                st = {"status": "unknown"}
            if st.get("verdict") in done_mod_FINAL or st.get("status") in ("done", "error"):
                with self.lock:
                    req["status"] = f"verdict:{st.get('verdict')}"
                self.r.log("gate_result", gate_id=gid, request_id=req["id"], verdict=st.get("verdict"),
                           delta_pct=st.get("delta_pct"), bundle=req["bundle"])
                self.r.decisions.trigger("gate_result", f"gate {gid} ({req['bundle']}): {st.get('verdict')} "
                                                        f"delta {st.get('delta_pct')}")
                w = self.r.agents.get(req["by"])
                if w:
                    w.add_event_message(f"GATE {gid} for {req['bundle']}: verdict {st.get('verdict')}, "
                                        f"delta_pct {st.get('delta_pct')}. {st.get('detail') or ''}")
                self.r.wake("lead")
                return
            self.r.finished.wait(self.r.gate_poll_s)


done_mod_FINAL = {"win", "loss", "lossy", "invalid", "error"}


class Team:
    def __init__(self, runner):
        self.r = runner
        self.tasks: dict[str, dict | None] = {w: None for w in runner.mission.worker_names}
        self.lock = threading.Lock()
        self.reports: list[dict] = []

    def assign(self, by: str, worker: str, task: str, lane: str) -> dict:
        if worker not in self.tasks:
            raise ToolError(f"no worker {worker!r}; workers: {', '.join(self.tasks)}")
        with self.lock:
            self.tasks[worker] = {"task": task[:1500], "lane": lane, "since": time.time()}
        self.r.log("worker_task", worker=worker, lane=lane, task=task[:500], by=by)
        self.r.agents[worker].add_event_message(f"TASK from the lead (lane {lane}): {task}")
        self.r.wake(worker)
        return {"assigned": worker, "lane": lane}

    def report(self, agent: str, summary: str, status: str) -> dict:
        with self.lock:
            t = self.tasks.get(agent)
            self.tasks[agent] = None
        self.reports.append({"by": agent, "summary": summary, "status": status, "task": t})
        self.r.log("worker_report", worker=agent, status=status, summary=summary[:600])
        self.r.decisions.trigger("worker_report", f"{agent} {status}: {summary[:300]}", source=agent)
        self.r.wake("lead")
        return {"reported": True, "note": "wait for the lead's next task"}

    def task_of(self, worker: str) -> dict | None:
        with self.lock:
            return self.tasks.get(worker)

    def status(self) -> dict:
        m = self.r.mission
        out = {}
        for w in m.worker_names:
            mine = [b for b in m.benches.values() if b["by"] == w][-4:]
            t = self.task_of(w)
            out[w] = {"lane": t and t["lane"], "task": t and t["task"][:200], "idle": t is None,
                      "steps": self.r.agents[w].steps,
                      "recent_benches": [{"bench_id": b["bench_id"], "delta_pct": (b.get("result") or {}).get(
                          "delta_pct"), "change": b.get("config_delta") or b.get("env") or b.get("bundle")}
                          for b in mine]}
        ref = m.reference()
        return {"workers": out, "reference": ref, "gate_requests": [
            {k: r[k] for k in ("id", "bundle", "delta_pct", "status", "gate_id")} for r in
            self.r.services.gate_queue.listing()], "recent_notes": self.r.services.notes.recent(6)}


@dataclass
class Services:
    brain: Brain
    decisions: Decisions
    inbox: Inbox
    notes: Notes
    gate: object
    bench: object
    gate_queue: GateQueue | None = None
    team: Team | None = None
    detector: Detector | None = None


# ------------------------------------------------------------------------------------------ runner
class MissionRunner:
    def __init__(self, spec: dict, *, executor=None, client: LLMClient | None = None, gate=None, bench=None,
                 mock_llm=None, root: Path | None = None):
        self.spec = spec
        self.root = Path(root or state_root())
        mid = spec["mission_id"]
        self.dir = self.root / "missions" / mid
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "spec.json").write_text(json.dumps(spec, indent=1, default=str))
        self.log = EventLog(self.dir / "events.jsonl", mid)
        self.finished = threading.Event()
        self.wait_gates = False
        self.gate_poll_s = float(spec.get("gate_poll_s", 15))
        self.wakers: dict[str, threading.Event] = {}
        self.team_cfg = TeamConfig.from_dict(spec.get("team"))
        self.playbook = pb_mod.load(spec.get("playbook"), self.root / "playbooks")
        recipe = normalize_recipe(spec["recipe"], spec.get("variant"), spec.get("bench_prompts"))
        if spec.get("recipe_id"):
            recipe["id"] = spec["recipe_id"]
        machine = spec["machine"] if isinstance(spec["machine"], dict) else {"name": spec["machine"]}
        self.executor = executor or self._default_executor(machine)

        criteria = done_mod.compile_criteria(mid, recipe, machine.get("name", "?"), spec.get("goal", ""),
                                             require_confirm=spec.get("require_confirm", True),
                                             min_gain_pct=spec.get("min_gain_pct"))
        lock_path = self.dir / "done.lock.json"
        if lock_path.exists():
            criteria = done_mod.load_locked(lock_path)
            done_hash = json.loads(lock_path.read_text())["sha256"]
        else:
            done_hash = done_mod.lock(criteria, lock_path)
        python = (spec.get("python") or recipe.get("python") or machine.get("python")
                  or "python3")
        ws = self.executor.workspace(mid)
        self.mission = MissionState(
            id=mid, recipe=recipe, machine=machine, python=python,
            bench_prompts=spec.get("bench_prompts") or recipe.get("workload", {}).get("bench_prompts", ""),
            site_packages=spec.get("site_packages") or _site_packages(self.executor, python),
            ws=ws, gpu_class=spec.get("gpu_class") or machine.get("gpu_class", "?"),
            time_box_min=float(spec.get("time_box_min", 75)), budget_usd=float(spec.get("budget_usd", 15)),
            n_workers=int(self.team_cfg.workers), criteria=criteria, done_hash=done_hash,
            executor=self.executor, readable_roots=list(spec.get("readable_roots") or []),
            bench_log=self.dir / "benches.jsonl")
        self.meter = Meter(self.dir / "usage.jsonl", self.mission.budget_usd)
        self.mission.meter = self.meter
        prof = get_profile(spec.get("profile", "crusoe"))
        if prof.name == "crusoe":
            load_secrets()
        self.client = client or LLMClient(prof, self.meter, mock=mock_llm)
        self.client.meter = self.meter
        self.profile = prof
        self.decisions = Decisions(self.dir / "decisions.jsonl", self.log)
        self.inbox = Inbox(self.dir / "inbox.jsonl", self.dir / "answers.jsonl", self.log)
        brain = Brain(Path(spec.get("brain_path") or self.root / "brain" / "facts.jsonl"))
        self.services = Services(brain, self.decisions, self.inbox, Notes(self.dir / "notes.jsonl"),
                                 gate or self._default_gate(machine), bench or self._default_bench(machine))
        self.detector = Detector(self.decisions, self.inbox, self.log)
        self.services.detector = self.detector
        self.registry = build_registry()
        self.brief = mission_brief(self.mission)
        self.protocol_text = self.playbook["protocol"].replace("{bar}", str(self.playbook["knobs"]
                                                                            ["verifier_bar_pct"]))
        self.agents: dict[str, Agent] = {}
        self.services.gate_queue = GateQueue(self)
        tc = self.team_cfg
        self.agents["lead"] = Agent("lead", "lead", self, prof.model_for(tc.lead.tier), tc.lead.context_cap,
                                    self.playbook["lead"])
        for w in self.mission.worker_names:
            self.agents[w] = Agent(w, "worker", self, prof.model_for(tc.worker.tier), tc.worker.context_cap,
                                   self.playbook["worker"])
        self.agents["verifier"] = Agent("verifier", "verifier", self, prof.model_for(tc.verifier.tier),
                                        tc.verifier.context_cap, self.playbook["verifier"])
        self.services.team = Team(self)
        for name in self.agents:
            self.wakers[name] = threading.Event()
        self.end_reason = ""
        self.errors: list[str] = []

    # -- defaults ---------------------------------------------------------------------------------
    def _default_executor(self, machine):
        try:
            from forge.fleet.executor import executor_for  # noqa: PLC0415
            return executor_for(machine)
        except ImportError:
            from forge.tools.executor import LocalExecutor  # noqa: PLC0415
            return LocalExecutor(machine.get("forge_root", str(self.root.parent)) + "/state")

    def _default_gate(self, machine):
        from forge.tools.gateclient import CoreGate  # noqa: PLC0415
        return CoreGate(machine)

    def _default_bench(self, machine):
        from forge.tools.bench import RemoteBenchRunner  # noqa: PLC0415
        return RemoteBenchRunner(self.executor, machine.get("forge_root", str(self.root.parent)),
                                 self.spec.get("python") or self.spec["recipe"].get("python")
                                 or machine.get("python", "python3"),
                                 vram_gb=float(self.spec.get("bench_vram_gb", 90)))

    # -- coordination -----------------------------------------------------------------------------
    def wake(self, name: str) -> None:
        ev = self.wakers.get(name)
        if ev:
            ev.set()

    def _status(self, agent: Agent, extra: list[str] | None = None) -> str:
        pend = self.decisions.pending_text() if agent.role == "lead" else None
        ex = list(extra or [])
        if agent.role == "worker":
            t = self.services.team.task_of(agent.name)
            ex.append(f"YOUR TASK (lane {t['lane']}): {t['task']}" if t else "You have no task; wait for one.")
        if agent.role == "verifier":
            ex.append(f"Requests pending review: {', '.join(self.services.gate_queue.pending()) or 'none'}. "
                      f"Bar: {self.services.gate_queue.bar_text()}.")
        return status_block(self.mission, agent.role, pend, self.detector.take_nudges(agent.name), ex)

    def _guard(self, fn, name):
        def run():
            try:
                fn()
            except BudgetExceeded as e:
                self.end(f"budget: {e}")
            except LLMError as e:
                self.errors.append(f"{name}: {e}")
                self.log("error", agent=name, error=str(e)[:400], infra=True)
                self.decisions.trigger("env_blocked", f"{name} LLM failure: {str(e)[:200]}", infra=True)
                self.inbox.ask("runner", f"[infra:llm] {name}: {str(e)[:300]}")
                if name == "lead":
                    self.end(f"lead LLM failure: {e}")
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{name}: {type(e).__name__}: {e}")
                self.log("error", agent=name, error=f"{type(e).__name__}: {e}",
                         traceback=traceback.format_exc()[-3000:])
                self.decisions.trigger("tool_bug", f"{name} crashed: {type(e).__name__}", infra=True)
        return run

    def end(self, reason: str) -> None:
        if not self.finished.is_set():
            self.end_reason = reason
            self.log("mission_ending", reason=reason)
            self.finished.set()
            for ev in self.wakers.values():
                ev.set()

    def _lead_loop(self) -> None:
        lead = self.agents["lead"]
        lead.add_event_message("The mission starts now. Plan the first round and assign every worker a task.")
        every = float(self.team_cfg.lead_every_s)
        while not self.finished.is_set():
            for _ in range(4):        # a Lead "turn" is a few calls; it must close pending triggers
                if self.finished.is_set():
                    return
                before = {t["id"] for t in self.decisions.pending()}
                out = lead.step(self._status(lead))
                if self.decisions.stop_requested:
                    self.end(f"lead stop: {self.decisions.stop_requested.get('evidence', '')[:200]}")
                    return
                still = {t["id"] for t in self.decisions.pending()}
                if not out["called"] and not still:
                    break
                if not still and "decide" in out["called"]:
                    break
                if before and still == before and not out["called"]:
                    lead.add_event_message("Pending triggers are still open; call decide() on them.")
            self.wakers["lead"].wait(every)
            self.wakers["lead"].clear()

    def _worker_loop(self, name: str) -> None:
        w = self.agents[name]
        idle_replies = 0
        gap = float(self.team_cfg.worker_step_gap_s)
        while not self.finished.is_set():
            if self.services.team.task_of(name) is None:
                self.wakers[name].wait(5)
                self.wakers[name].clear()
                continue
            out = w.step(self._status(w))
            if out["called"]:
                idle_replies = 0
            else:
                idle_replies += 1
                if idle_replies == 2:
                    w.add_event_message("Use a tool to make progress on your task, or call report().")
                if idle_replies >= 4:
                    self.services.team.report(name, "stopped responding with tool calls", "blocked")
                    idle_replies = 0
            if gap:
                self.finished.wait(gap)

    def _verifier_loop(self) -> None:
        v = self.agents["verifier"]
        while not self.finished.is_set():
            pend = self.services.gate_queue.pending()
            if not pend:
                self.wakers["verifier"].wait(5)
                self.wakers["verifier"].clear()
                continue
            for _ in range(6):
                if self.finished.is_set() or not self.services.gate_queue.pending():
                    break
                v.step(self._status(v))

    # -- run --------------------------------------------------------------------------------------
    def run(self) -> dict:
        m = self.mission
        self.log("mission_start", recipe_id=m.recipe_id, machine=m.machine.get("name"), gpu_class=m.gpu_class,
                 time_box_min=m.time_box_min, budget_usd=m.budget_usd, team=list(self.agents),
                 models={n: a.model.model_id for n, a in self.agents.items()}, playbook=self.playbook.get("hash"),
                 profile=self.profile.name)
        self.log("done_locked", sha256=m.done_hash, criteria=m.criteria)
        threads = [threading.Thread(target=self._guard(self._lead_loop, "lead"), name="lead", daemon=True),
                   threading.Thread(target=self._guard(self._verifier_loop, "verifier"), name="verifier",
                                    daemon=True)]
        threads += [threading.Thread(target=self._guard(lambda n=n: self._worker_loop(n), n), name=n, daemon=True)
                    for n in m.worker_names]
        for t in threads:
            t.start()
        warned = set()
        try:
            while not self.finished.is_set():
                if m.minutes_left() <= 0:
                    self.end("time box used up")
                    break
                if self.meter.over_budget():
                    self.end("LLM budget used up")
                    break
                if not threads[0].is_alive():
                    self.end(self.end_reason or "lead thread ended")
                    break
                for frac in (0.5, 0.8, 0.9):
                    if m.frac_used() >= frac and frac not in warned:
                        warned.add(frac)
                        self.decisions.trigger("time", f"{int(frac * 100)}% of the time box used; "
                                                       f"{m.minutes_left():.0f} min left")
                        self.wake("lead")
                    if self.meter.budget_usd and self.meter.cost_usd >= frac * self.meter.budget_usd \
                            and f"b{frac}" not in warned:
                        warned.add(f"b{frac}")
                        self.decisions.trigger("budget", f"{int(frac * 100)}% of the LLM budget used")
                        self.wake("lead")
                for ans in self.inbox.new_answers():
                    q = ans.get("question_id")
                    target = ans.get("to") or self._asker(q)
                    msg = f"HUMAN ANSWER to {q}: {ans.get('answer', '')}"
                    self.log("human_answer", question_id=q, to=target)
                    (self.agents.get(target) or self.agents["lead"]).add_event_message(msg)
                    self.wake(target or "lead")
                self.finished.wait(float(self.spec.get("tick_s", 2)))
        except KeyboardInterrupt:
            self.end("interrupted")
        for t in threads:
            t.join(timeout=float(self.spec.get("join_timeout_s", 120)))
        grace = float(self.spec.get("gate_grace_min", 0)) * 60
        if grace:
            self.wait_gates = True
            t0 = time.time()
            while time.time() - t0 < grace:
                a = done_mod.audit(m.criteria, self.services.gate.ledger(mission_id=m.id))
                if not a["open_gates"]:
                    break
                time.sleep(self.gate_poll_s)
            self.wait_gates = False
        row = self.row()
        self.log("mission_end", reason=self.end_reason, row=row)
        (self.dir / "row.json").write_text(json.dumps(row, indent=1, default=str))
        return row

    def _asker(self, qid: str | None) -> str | None:
        for r in events.read(self.inbox.path):
            if r.get("question_id") == qid:
                return r.get("by") if r.get("by") in self.agents else "lead"
        return None

    def row(self) -> dict:
        m = self.mission
        ledger = self.services.gate.ledger(mission_id=m.id)
        a = done_mod.audit(m.criteria, ledger)
        ref = m.reference()
        baseline = ref.get("value") if ref else None
        best = a["best"]
        change = None
        if best:
            for bid, info in m.bundles.items():
                if info["hash"] == best.get("bundle_hash"):
                    change = {"bundle": bid, **info["manifest"]}
        quick = [((b.get("result") or {}).get("delta_pct"), b["bench_id"]) for b in m.benches.values()
                 if (b.get("result") or {}).get("delta_pct") is not None]
        return {
            "mission_id": m.id, "recipe_id": m.recipe_id, "machine": m.machine.get("name"),
            "gpu_class": m.gpu_class, "metric": m.metric, "baseline": baseline,
            "best": round(baseline * (1 + float(best["delta_pct"]) / 100), 3) if best and baseline else baseline,
            "delta_pct": float(best["delta_pct"]) if best else 0.0,
            "gate_verdicts": a["verdicts"], "change_summary": change, "evidence_ids": a["evidence_ids"],
            "done": {"status": a["status"], "met": a["met"], "criteria_sha256": m.done_hash,
                     "open_gates": a["open_gates"]},
            "quick_bench_best": max(quick) if quick else None, "n_benches": len(m.benches),
            "n_gate_requests": len(self.services.gate_queue.reqs),
            "cost_usd": round(self.meter.cost_usd, 4), "llm": self.meter.summary(),
            "playbook": self.playbook.get("hash"), "ended": self.end_reason, "errors": self.errors[:10],
            "detector": dict(self.detector.counts), "minutes": round((time.time() - m.t_start) / 60, 1),
        }


def normalize_recipe(recipe: dict, variant: str | None = None, bench_prompts: str | None = None) -> dict:
    """Catalog recipe (forge/recipes/*.yaml) -> the shape the runner uses: serve.env merged from vllm.env,
    one workload picked from variants, chat settings folded into the workload, id suffixed with the variant."""
    r = dict(recipe)
    serve = dict(r.get("serve") or {})
    serve["env"] = {**((r.get("vllm") or {}).get("env") or {}), **(serve.get("env") or {})}
    r["serve"] = serve
    wl = dict(r.get("workload") or {})
    variants = r.get("variants") or []
    if not wl and variants:
        v = next((x for x in variants if x.get("name") == variant), None) if variant else variants[0]
        if v is None:
            raise ValueError(f"recipe {r.get('id')} has no variant {variant!r}")
        wl = dict(v)
        r["id"] = f"{r['id']}:{v['name']}"
    chat = r.get("chat")
    if chat and "chat" not in wl:
        wl["chat"] = {"system_prompt": chat.get("system_prompt"),
                      "chat_template_kwargs": chat.get("chat_template_kwargs") or {}}
    if bench_prompts:
        wl["bench_prompts"] = bench_prompts
    wl.setdefault("metric", "decode_tok_s" if int(wl.get("concurrency", 1)) <= 1 else "output_tok_s")
    r["workload"] = wl
    return r


def _site_packages(executor, python: str) -> str:
    r = executor.run([python, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], timeout_s=60)
    sp = r.stdout.strip().splitlines()[-1] if r.rc == 0 and r.stdout.strip() else ""
    return sp or "/nonexistent-site-packages"


def run_mission(spec: dict, **kw) -> dict:
    return MissionRunner(spec, **kw).run()


def audit_mission(mission_id: str, gate, root: Path | None = None) -> dict:
    """Re-audit a finished mission against the gate ledger (e.g. after its open gates finish)."""
    d = Path(root or state_root()) / "missions" / mission_id
    criteria = done_mod.load_locked(d / "done.lock.json")
    return done_mod.audit(criteria, gate.ledger(mission_id=mission_id))
