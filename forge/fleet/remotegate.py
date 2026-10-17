"""Gate access for a mission whose target machine is not the one the mission runner is on.

forge.gate keeps its ledger on its own machine (one gate per machine), so a runner on the rig that drives a box
calls forge.gate ON the box through the executor. Same interface as forge.tools.gateclient.CoreGate:

    g = RemoteGate(machine)                   # or gate_for(machine): RemoteGate when remote, CoreGate when local
    g.request(recipe_id=, bundle_dir=, bundle_hash=, mission_id=, requester=, claim=) -> row
    g.status(gate_id) -> row;  g.ledger(mission_id=None, recipe_id=None) -> [rows]
Rows are normalised with forge.tools.gateclient.normalize_row. bundle_dir is a path on the target.
"""
from __future__ import annotations

import json
import shlex

from forge.fleet.executor import executor_for
from forge.tools.gateclient import CoreGate, normalize_row

_PROG = ("import json,sys; import forge.gate as g; a=json.loads(sys.argv[1]); fn=a.pop('fn'); "
         "print('GATE_RPC '+json.dumps(getattr(g, fn)(**a), default=str))")


class GateUnavailable(RuntimeError):
    pass


class RemoteGate:
    def __init__(self, machine: dict):
        self.machine = machine
        self.ex = executor_for(machine)

    def _call(self, fn: str, **kw):
        py = self.machine.get("sched_python") or self.machine.get("python") or "python3"
        root = self.machine.get("forge_root", "/root/forge")
        r = self.ex.run(f"{py} -c {shlex.quote(_PROG)} {shlex.quote(json.dumps({'fn': fn, **kw}))}", cwd=root,
                        env={"PYTHONPATH": root, "FORGE_ROOT": root}, timeout_s=120)
        if r.infra_error:
            raise GateUnavailable(f"gate on {self.machine['name']} unreachable: {r.infra_error}")
        line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("GATE_RPC ")), None)
        if line is None:
            raise GateUnavailable(f"gate.{fn} on {self.machine['name']} failed: {(r.stderr or r.stdout).strip()[-400:]}")
        return json.loads(line[len("GATE_RPC "):])

    def request(self, *, recipe_id: str, bundle_dir: str, bundle_hash: str, mission_id: str,
                requester: str, claim: str) -> dict:
        r = self._call("request", recipe_id=recipe_id, bundle_dir=bundle_dir, mission_id=mission_id,
                       machine=self.machine.get("name"), requester=requester, claim=claim)
        return normalize_row({**(r if isinstance(r, dict) else {"gate_id": r}), "mission_id": mission_id,
                              "recipe_id": recipe_id, "bundle_hash": bundle_hash})

    def status(self, gate_id: str) -> dict:
        return normalize_row(self._call("status", gate_id=gate_id))

    def ledger(self, mission_id: str | None = None, recipe_id: str | None = None) -> list[dict]:
        rows = [normalize_row(r) for r in self._call("ledger", machine=self.machine.get("name"))]
        return [r for r in rows if (mission_id is None or r["mission_id"] == mission_id)
                and (recipe_id is None or r["recipe_id"] == recipe_id)]


def gate_for(machine: dict):
    """CoreGate when the gate is on this machine, RemoteGate otherwise."""
    ex = executor_for(machine)
    return CoreGate(machine) if ex.ssh is None else RemoteGate(machine)
