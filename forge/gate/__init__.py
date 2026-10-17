"""forge.gate: the only judge of a win. See docs/interfaces/core.md.

    import forge.gate as gate
    row = gate.request(recipe_id="bolt-74b-lilicorr:c1", bundle_dir="/abs/bundle", mission_id="m7",
                       machine="squaredcube1-ubuntu", requester="m7/verifier", claim="...")
    gate.status(row["gate_id"]); gate.ledger(); gate.wait(row["gate_id"])

Requests are queued for the machine's gate service (python -m forge.gate.service ensure).
"""
from __future__ import annotations

import time

from forge.gate import records as _ledger

FINAL = ("done", "error")


def _public(r: dict) -> dict:
    return {k: r.get(k) for k in _ledger.PUBLIC if k in r} | {"t": r.get("t")}


def request(recipe_id: str, bundle_dir: str, mission_id: str | None = None, machine: str | None = None,
            requester: str | None = None, claim: str | None = None, bundle_hash: str | None = None,
            recipe: dict | None = None, python: str | None = None, kind: str = "candidate", reps: int | None = None) -> dict:
    """Queue a gate run of a bundle against the recipe's published reference. Identical pending or finished
    requests (same recipe, bundle hash and kind) return the existing row with dedup=True."""
    r = _ledger.submit(kind, recipe_id, bundle_dir=bundle_dir, bundle_hash=bundle_hash, mission_id=mission_id,
                       machine=machine, requester=requester, claim=claim, recipe=recipe, python=python, reps=reps)
    return _public(r) | ({"dedup": True} if r.get("dedup") else {})


submit = request


def reference(recipe_id: str, recipe: dict | None = None, python: str | None = None, reps: int | None = None,
              requester: str = "setup") -> dict:
    """Queue a reference control (reference vs reference) for a recipe."""
    return _public(_ledger.submit("reference", recipe_id, recipe=recipe, python=python, reps=reps,
                                  requester=requester, claim="reference control"))


def status(gate_id: str) -> dict:
    r = _ledger.latest().get(gate_id)
    if r is None:
        return {"gate_id": gate_id, "status": "error", "verdict": "error", "detail": "unknown gate id"}
    return _public(r)


def ledger(machine: str | None = None, recipe_id: str | None = None, mission_id: str | None = None) -> list[dict]:
    out = []
    for r in _ledger.latest().values():
        if machine and r.get("machine") not in (None, machine):
            continue
        if recipe_id and recipe_id not in (r.get("recipe_id"), r.get("recipe_key")):
            continue
        if mission_id and r.get("mission_id") != mission_id:
            continue
        out.append(_public(r))
    return sorted(out, key=lambda r: r.get("t") or 0)


def wait(gate_id: str, timeout: float | None = None, poll: float = 10.0) -> dict:
    end = None if timeout is None else time.time() + timeout
    while True:
        r = status(gate_id)
        if r.get("status") in FINAL:
            return r
        if end is not None and time.time() > end:
            return r
        time.sleep(poll)
