"""Done Means Done: criteria compiled from the recipe + goal and locked at mission start; the auditor
checks them against the gate ledger only (never against what an agent says)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

VERSION = 1


def compile_criteria(mission_id: str, recipe: dict, machine: str, goal: str, *, require_confirm: bool = True,
                     min_gain_pct: float | None = None) -> dict:
    if min_gain_pct is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", goal or "")
        min_gain_pct = float(m.group(1)) if m else 0.0
    metric = (recipe.get("metric") or recipe.get("workload", {}).get("metric") or "decode_tok_s")
    if isinstance(metric, dict):
        metric = metric.get("name", "decode_tok_s")
    return {
        "version": VERSION,
        "mission_id": mission_id,
        "recipe_id": recipe.get("id"),
        "machine": machine,
        "metric": metric,
        "goal": goal,
        "min_gain_pct": min_gain_pct,
        "require_confirm": require_confirm,
        "lossless": "decided by the gate's lossless check; a lossy verdict never counts",
        "success": ("the gate ledger holds a candidate row for this mission and recipe with verdict win and "
                    f"delta_pct >= {min_gain_pct}" + (", and a confirm row with verdict win for the same bundle"
                                                      if require_confirm else "")),
        "termination": "time box or LLM budget used up, or a Lead stop decision, and no gate of this mission "
                       "still queued or running",
    }


def lock(criteria: dict, path: Path) -> str:
    body = json.dumps(criteria, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(body.encode()).hexdigest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"criteria": criteria, "sha256": h}, indent=1))
    return h


def load_locked(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    body = json.dumps(d["criteria"], sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(body.encode()).hexdigest() != d["sha256"]:
        raise ValueError("done criteria were modified after lock")
    return d["criteria"]


def audit(criteria: dict, ledger: list[dict]) -> dict:
    """-> {status: win|no_win|pending, met, best, evidence_ids, open_gates, verdicts}"""
    rows = [r for r in ledger if r.get("mission_id") == criteria["mission_id"]
            and (r.get("recipe_id") in (None, criteria["recipe_id"]))]
    open_gates = [r["gate_id"] for r in rows if r.get("status") in ("queued", "running")]
    cands = [r for r in rows if r.get("kind", "candidate") == "candidate" and r.get("verdict")]
    confirms = {r.get("bundle_hash") for r in rows if r.get("kind") == "confirm" and r.get("verdict") == "win"}
    wins = []
    for r in cands:
        if r["verdict"] != "win":
            continue
        d = r.get("delta_pct")
        if d is None or float(d) < criteria["min_gain_pct"]:
            continue
        if criteria["require_confirm"] and not (r.get("confirmed") or r.get("bundle_hash") in confirms):
            continue
        wins.append(r)
    best = max(wins, key=lambda r: float(r["delta_pct"])) if wins else None
    verdicts = [{"gate_id": r["gate_id"], "verdict": r["verdict"], "delta_pct": r.get("delta_pct"),
                 "kind": r.get("kind"), "bundle_hash": r.get("bundle_hash")} for r in rows if r.get("verdict")]
    status = "win" if best else ("pending" if open_gates else "no_win")
    return {"status": status, "met": bool(best), "best": best, "open_gates": open_gates, "verdicts": verdicts,
            "evidence_ids": [r.get("evidence_id") for r in ([best] if best else []) if r.get("evidence_id")]}
