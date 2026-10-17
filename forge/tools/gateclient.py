"""Gate access for the mission runner. The gate itself is core's (forge.gate); this is the adapter.

Ledger rows the done auditor relies on (normalized here whatever core's shape is):
  {gate_id, mission_id, recipe_id, machine, bundle_hash, kind: candidate|confirm|reference,
   status: queued|running|done|error, verdict: win|loss|lossy|invalid|error|None, delta_pct, evidence_id, t}
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

FINAL = {"win", "loss", "lossy", "invalid", "error"}


def normalize_row(r: dict) -> dict:
    verdict = r.get("verdict") or r.get("result")
    if isinstance(verdict, str):
        verdict = verdict.lower()
        verdict = {"pass": "win", "champion": "win", "fail": "loss", "lossy_fail": "lossy"}.get(verdict, verdict)
    status = (r.get("status") or ("done" if verdict in FINAL else "queued")).lower()
    return {"gate_id": r.get("gate_id") or r.get("id"), "mission_id": r.get("mission_id"),
            "recipe_id": r.get("recipe_id") or r.get("recipe"), "machine": r.get("machine"),
            "bundle_hash": r.get("bundle_hash") or r.get("bundle"), "kind": r.get("kind", "candidate"),
            "status": status, "verdict": verdict, "delta_pct": r.get("delta_pct") or r.get("gain_pct"),
            "evidence_id": r.get("evidence_id") or r.get("gate_id") or r.get("id"),
            "confirmed": bool(r.get("confirmed", False)), "t": r.get("t") or r.get("ts"),
            "detail": r.get("detail") or r.get("reason") or ""}


class CoreGate:
    """Adapter over core's forge.gate. Imports lazily so the runner works before core lands."""

    def __init__(self, machine: dict):
        self.machine = machine

    def _mod(self):
        try:
            import forge.gate as g  # noqa: PLC0415
            return g
        except ImportError as e:
            raise RuntimeError(f"forge.gate unavailable: {e}")

    def request(self, *, recipe_id: str, bundle_dir: str, bundle_hash: str, mission_id: str,
                requester: str, claim: str) -> dict:
        g = self._mod()
        fn = getattr(g, "request", None) or getattr(g, "submit")
        r = fn(recipe_id=recipe_id, bundle_dir=bundle_dir, mission_id=mission_id, machine=self.machine.get("name"),
               requester=requester, claim=claim)
        return normalize_row({**(r if isinstance(r, dict) else {"gate_id": r}), "mission_id": mission_id,
                              "recipe_id": recipe_id, "bundle_hash": bundle_hash})

    def status(self, gate_id: str) -> dict:
        g = self._mod()
        return normalize_row(g.status(gate_id))

    def ledger(self, mission_id: str | None = None, recipe_id: str | None = None) -> list[dict]:
        g = self._mod()
        try:
            raw = g.ledger(mission_id=mission_id, recipe_id=recipe_id)
        except TypeError:
            raw = g.ledger()
        rows = [normalize_row(r) for r in raw]
        return [r for r in rows if (mission_id is None or r["mission_id"] == mission_id)
                and (recipe_id is None or r["recipe_id"] == recipe_id)]


class MockGate:
    """Ledger in a JSONL file. Verdict = win when the mock bench effect for the bundle is above
    `win_above_pct`, decided after `delay_s`."""

    def __init__(self, path: Path, verdict_fn=None, delay_s: float = 0.0):
        self.path = Path(path)
        self.verdict_fn = verdict_fn or (lambda bundle_dir: ("loss", 0.0))
        self.delay_s = delay_s
        self.lock = threading.Lock()

    def _append(self, row: dict) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")

    def request(self, *, recipe_id, bundle_dir, bundle_hash, mission_id, requester, claim):
        gid = "gate-" + uuid.uuid4().hex[:10]
        row = {"gate_id": gid, "mission_id": mission_id, "recipe_id": recipe_id, "bundle_hash": bundle_hash,
               "kind": "candidate", "status": "queued", "verdict": None, "t": time.time(), "bundle_dir": bundle_dir}
        self._append(row)

        def finish():
            time.sleep(self.delay_s)
            verdict, delta = self.verdict_fn(bundle_dir)
            self._append({**row, "status": "done", "verdict": verdict, "delta_pct": delta, "t": time.time(),
                          "evidence_id": f"ev-{gid}"})
        threading.Thread(target=finish, daemon=True).start()
        return normalize_row(row)

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        latest: dict[str, dict] = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["gate_id"]] = r
        return list(latest.values())

    def status(self, gate_id):
        for r in self._rows():
            if r["gate_id"] == gate_id:
                return normalize_row(r)
        return normalize_row({"gate_id": gate_id, "status": "error", "verdict": "error", "detail": "unknown id"})

    def ledger(self, mission_id=None, recipe_id=None):
        rows = [normalize_row(r) for r in self._rows()]
        return [r for r in rows if (mission_id is None or r["mission_id"] == mission_id)
                and (recipe_id is None or r["recipe_id"] == recipe_id)]
