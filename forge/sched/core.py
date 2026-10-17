"""Per-machine scheduler: places grants on GPUs, queues what cannot run yet, holds the leases.

Kinds:
  gate   all timing GPUs, exclusive. While a gate waits, nothing new starts on timing GPUs.
  serve  one GPU (a long-lived server). Exclusive on a timing GPU; VRAM-accounted on a work GPU.
  bench  one GPU, like serve but meant to be short.
  work   a work GPU (or a timing GPU when the machine has no work GPU), VRAM-accounted.

A request never fails for lack of capacity: it waits in the queue with a `wait_reason`.
It only fails when it can never fit (e.g. more VRAM than the card has).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

from forge.sched.keeper import LeaseKeeper
from forge.sched.landlord import LandlordError, LandlordUnreachable
from forge.sched.machine import Machine

KINDS = ("gate", "serve", "bench", "work")
RETRY_REFUSED_S = 30
MAX_MINUTES = 12 * 60


class RequestError(ValueError):
    pass


def pid_alive(pid: int | None, start: str | None = None) -> bool:
    if not pid:
        return True       # no holder registered: lives until released or past deadline
    try:
        with open(f"/proc/{pid}/stat") as f:
            st = f.read().rsplit(")", 1)[1].split()
    except OSError:
        return False
    if st[0] == "Z":
        return False
    return start is None or st[19] == start


def pid_start(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class Scheduler:
    def __init__(self, machine: Machine, keeper: LeaseKeeper, state_path: str, clock=time.time,
                 alive=pid_alive):
        self.m = machine
        self.keeper = keeper
        self.state_path = state_path
        self.clock = clock
        self.alive = alive
        self.lock = threading.RLock()
        self.grants: dict[str, dict] = {}
        self.history: list[dict] = []
        self._load()

    # -- persistence -------------------------------------------------------------
    def _load(self):
        try:
            with open(self.state_path) as f:
                st = json.load(f)
            self.grants = st.get("grants", {})
            self.history = st.get("history", [])[-200:]
        except (OSError, ValueError):
            pass

    def save(self):
        with self.lock:
            tmp = self.state_path + ".tmp"
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump({"grants": self.grants, "history": self.history[-200:]}, f, indent=1)
            os.replace(tmp, self.state_path)

    # -- API ---------------------------------------------------------------------
    def request(self, kind: str, owner: str, purpose: str, minutes: int = 60, vram_gb: float | None = None,
                gpu: int | None = None, where: str = "any", pct: int | None = None,
                holder_pid: int | None = None) -> dict:
        if kind not in KINDS:
            raise RequestError(f"kind must be one of {KINDS}")
        if not purpose or len(purpose.strip()) < 8:
            raise RequestError("purpose is required (what runs and why it needs this much)")
        if where not in ("any", "timing", "work"):
            raise RequestError("where must be any, timing or work")
        minutes = int(max(1, min(int(minutes), MAX_MINUTES)))
        if gpu is not None and gpu not in self.m.all_gpus():
            raise RequestError(f"no GPU {gpu} on {self.m.name}; GPUs: {self.m.all_gpus()}")
        if kind == "gate" and not self.m.timing_gpus:
            raise RequestError(f"{self.m.name} has no timing GPUs")
        if kind == "work" and vram_gb is None:
            raise RequestError("work grants need vram_gb")
        if vram_gb is not None:
            biggest = max(self.m.vram_gb.values() or [0])
            if vram_gb > biggest - self.m.vram_reserve_gb:
                raise RequestError(f"vram_gb {vram_gb} can never fit (largest GPU {biggest} GB)")
        gid = f"{kind}-{uuid.uuid4().hex[:10]}"
        now = self.clock()
        g = {"id": gid, "kind": kind, "owner": owner, "purpose": purpose.strip(), "minutes": minutes,
             "vram_gb": vram_gb, "gpu_req": gpu, "where": where, "pct": pct, "holder_pid": holder_pid,
             "holder_start": pid_start(holder_pid) if holder_pid else None,
             "state": "queued", "wait_reason": "waiting for the scheduler", "gpus": [], "leases": {},
             "ports": None, "cpus": [], "created": now, "granted": None, "deadline": None,
             "retry_after": 0.0, "ended": None, "outcome": None}
        with self.lock:
            self.grants[gid] = g
            self.save()
        self.tick()
        return self.get(gid)

    def get(self, gid: str) -> dict:
        with self.lock:
            g = self.grants.get(gid) or next((h for h in self.history if h["id"] == gid), None)
            if g is None:
                raise KeyError(gid)
            return self._public(g)

    def renew(self, gid: str, minutes: int) -> dict:
        with self.lock:
            g = self.grants.get(gid)
            if not g or g["state"] not in ("active", "queued", "leasing"):
                raise KeyError(gid)
            g["minutes"] = int(min(MAX_MINUTES, minutes))
            if g["state"] == "active":
                g["deadline"] = self.clock() + g["minutes"] * 60
            self.save()
            return self._public(g)

    def release(self, gid: str, outcome: str = "done") -> dict:
        with self.lock:
            g = self.grants.get(gid)
            if not g:
                return self.get(gid)
            self._end(g, "released", outcome)
            return self._public(g)

    def status(self) -> dict:
        with self.lock:
            return {"machine": self.m.to_dict(),
                    "grants": [self._public(g) for g in self._ordered()],
                    "recent": [self._public(h) for h in self.history[-20:]],
                    "keeper": self.keeper.snapshot(), "time": self.clock()}

    # -- internals ---------------------------------------------------------------
    def _public(self, g: dict) -> dict:
        d = {k: v for k, v in g.items() if k not in ("retry_after", "holder_start")}
        d["env"] = self.env_for(g)
        d["gpu_uuids"] = [self.m.gpu_uuids.get(i) for i in g["gpus"]]
        return d

    def env_for(self, g: dict) -> dict:
        """Env for an unsandboxed process on this grant (physical indexes; sandboxed jobs remap to 0..n)."""
        if g["state"] != "active":
            return {}
        env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(map(str, g["gpus"]))}
        lids = [g["leases"][str(i)] for i in g["gpus"] if str(i) in g["leases"]]
        if lids:
            env["LANDLORD_LEASE_ID"] = lids[0]
            env["FORGE_LEASE_IDS"] = ",".join(lids)
        if g["ports"]:
            env["FORGE_PORT_LO"], env["FORGE_PORT_HI"] = map(str, g["ports"])
        env["FORGE_GRANT_ID"] = g["id"]
        return env

    def _ordered(self) -> list[dict]:
        return sorted(self.grants.values(), key=lambda g: (g["kind"] != "gate", g["created"]))

    def _end(self, g: dict, state: str, outcome: str):
        for lid in g["leases"].values():
            self.keeper.release(lid, f"{g['kind']} {g['id']} ({g['owner']}): {outcome}"[:400])
        g["state"], g["outcome"], g["ended"] = state, outcome, self.clock()
        g["wait_reason"] = None
        self.grants.pop(g["id"], None)
        self.history.append(g)
        self.save()

    def _used(self) -> tuple[set[int], dict[int, float]]:
        """GPUs held exclusively, and VRAM held per shared GPU, by grants that hold or are taking GPUs."""
        excl, vram = set(), {}
        for g in self.grants.values():
            if g["state"] not in ("active", "leasing"):
                continue
            for i in g["gpus"]:
                if self._exclusive(g, i):
                    excl.add(i)
                else:
                    vram[i] = vram.get(i, 0.0) + (g["vram_gb"] or 0.0)
        return excl, vram

    def _exclusive(self, g: dict, gpu: int) -> bool:
        return g["kind"] == "gate" or (g["kind"] in ("serve", "bench") and gpu in self.m.timing_gpus
                                       and gpu not in self.m.work_gpus)

    def _gpu_vram(self, i: int) -> float:
        return self.m.vram_gb.get(i, 0.0) - self.m.vram_reserve_gb

    def _place(self, g: dict, gate_waiting: bool) -> tuple[list[int] | None, str]:
        excl, vram = self._used()
        timing = list(self.m.timing_gpus)
        if g["kind"] == "gate":
            busy = [i for i in timing if i in excl or vram.get(i, 0) > 0]
            if busy:
                return None, f"gate waits for timing GPUs {busy} to be released"
            return timing, ""
        if g["kind"] == "work":
            pool = list(self.m.work_gpus) or timing
        elif g["where"] == "timing":
            pool = timing
        elif g["where"] == "work":
            pool = list(self.m.work_gpus)
        else:
            pool = timing + [i for i in self.m.work_gpus if i not in timing]
        if g["gpu_req"] is not None:
            pool = [i for i in pool if i == g["gpu_req"]] or [g["gpu_req"]]
        if not pool:
            return None, f"no {g['where']} GPU on this machine"
        reasons = []
        for i in pool:
            if i in timing and gate_waiting and i not in self.m.work_gpus:
                reasons.append(f"GPU {i}: a gate is waiting for timing GPUs")
                continue
            if i in excl:
                reasons.append(f"GPU {i}: held exclusively")
                continue
            if self._exclusive(g, i):
                if vram.get(i, 0) > 0:
                    reasons.append(f"GPU {i}: shared work is running")
                    continue
                return [i], ""
            need = g["vram_gb"] if g["vram_gb"] is not None else self._gpu_vram(i)
            if vram.get(i, 0) + need > self._gpu_vram(i) + 1e-6:
                reasons.append(f"GPU {i}: {self._gpu_vram(i) - vram.get(i, 0):.1f} GB free, need {need:.1f}")
                continue
            if g["vram_gb"] is None:
                g["vram_gb"] = need
            return [i], ""
        return None, "; ".join(reasons)

    def _ports(self) -> tuple[int, int] | None:
        used = {tuple(g["ports"]) for g in self.grants.values() if g["ports"]}
        for k in range(self.m.port_blocks):
            lo = self.m.port_base + k * self.m.port_block
            p = (lo, lo + self.m.port_block - 1)
            if p not in used:
                return p
        return None

    def tick(self) -> None:
        """One scheduling pass: reap dead holders and deadlines, place queued grants, keep leases."""
        now = self.clock()
        with self.lock:
            for g in list(self.grants.values()):
                if not self.alive(g["holder_pid"], g.get("holder_start")):
                    self._end(g, "released", "holder process exited")
                elif g["state"] == "active" and g["deadline"] and now > g["deadline"]:
                    self._end(g, "expired", "deadline passed without renew")
            gate_waiting = any(g["kind"] == "gate" and g["state"] == "queued" for g in self.grants.values())
            todo = [g for g in self._ordered() if g["state"] == "queued" and g["retry_after"] <= now]
        for g in todo:
            with self.lock:
                if g["id"] not in self.grants or g["state"] != "queued":
                    continue
                gpus, why = self._place(g, gate_waiting and g["kind"] != "gate")
                if gpus is None:
                    g["wait_reason"] = why
                    continue
                ports = self._ports()
                if ports is None:
                    g["wait_reason"] = "no free port block"
                    continue
                g["state"], g["gpus"], g["ports"] = "leasing", gpus, list(ports)
                g["wait_reason"] = "renting from landlord"
                self.save()
            self._lease(g)
        active = {g["id"] for g in self.grants.values() if g["state"] in ("active", "leasing")}
        self.keeper.tick(active, {g["id"]: f"{g['kind']} for {g['owner']} still running"
                                  for g in self.grants.values()})

    def _lease(self, g: dict) -> None:
        exclusive = g["kind"] in ("gate", "serve", "bench")
        failure = None
        for i in g["gpus"]:
            if str(i) in g["leases"]:
                continue
            excl = self._exclusive(g, i)
            pct = 100 if excl else int(g["pct"] or 50)
            vram = round(self.m.vram_gb.get(i, 0) if excl else (g["vram_gb"] or 1.0), 2)
            purpose = f"forge {g['kind']} for {g['owner']}: {g['purpose']}"[:500]
            try:
                r = self.keeper.rent(i, pct, vram, g["minutes"] if exclusive else g["minutes"], purpose, g["id"])
                g["leases"][str(i)] = r["lease_id"]
            except LandlordUnreachable as e:
                failure = f"landlord unreachable ({e}); retrying"
                break
            except LandlordError as e:
                failure = f"landlord refused GPU {i}: {str(e)[:300]}; retrying in {RETRY_REFUSED_S}s"
                break
        with self.lock:
            if g["id"] not in self.grants:   # released while we were renting
                for lid in g["leases"].values():
                    self.keeper.release(lid, "grant released before it started")
                return
            if failure:
                for lid in g["leases"].values():
                    self.keeper.release(lid, "grant not started (other GPUs unavailable); returned")
                g["leases"] = {}
                g["state"], g["gpus"], g["ports"] = "queued", [], None
                g["wait_reason"] = failure
                g["retry_after"] = self.clock() + (RETRY_REFUSED_S if "refused" in failure else 10)
            else:
                g["state"], g["granted"] = "active", self.clock()
                g["deadline"] = g["granted"] + g["minutes"] * 60
                g["cpus"] = list(self.m.gate_cpus if g["kind"] == "gate" else self.m.agent_cpus)
                g["wait_reason"] = None
            self.save()

    def shutdown(self, outcome: str = "scheduler stopped") -> bool:
        with self.lock:
            for g in list(self.grants.values()):
                self._end(g, "released", outcome)
        return self.keeper.close(outcome)
