"""Shared Brain: facts that Nemotron agents record, shared across missions (append-only JSONL on the rig).

A fact: {fact_id, claim, recipe_id, gpu_class, lane, status: measured|refuted|blocked|open, evidence,
mission_id, by, t}. Only agents write here; humans and Claude never add findings (NO-LEAK).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.agents import events


class Brain:
    def __init__(self, path: Path):
        self.path = Path(path)

    def add(self, fact: dict) -> str:
        fid = "f-" + hashlib.sha256(json.dumps(fact, sort_keys=True).encode()).hexdigest()[:10]
        events.append(self.path, {"fact_id": fid, **{k: (str(v)[:600] if isinstance(v, str) else v)
                                                     for k, v in fact.items()}})
        return fid

    def all(self) -> list[dict]:
        return events.read(self.path)

    def view(self, recipe_id: str | None, lane: str | None = None, query: str | None = None, limit: int = 30,
             model: str | None = None) -> list[dict]:
        """Facts for this recipe first, then same-model facts, then the rest; newest first within a group."""
        rows = self.all()
        if lane:
            rows = [r for r in rows if r.get("lane") == lane]
        if query:
            q = query.lower()
            rows = [r for r in rows if q in (r.get("claim", "") + " " + r.get("evidence", "")).lower()]

        def rank(r):
            same = r.get("recipe_id") == recipe_id
            return (0 if same else 1, -float(r.get("t") or 0))
        rows.sort(key=rank)
        keep = ("fact_id", "claim", "lane", "status", "evidence", "recipe_id", "gpu_class")
        return [{k: r.get(k) for k in keep} for r in rows[:limit]]

    def prompt_view(self, recipe_id: str, max_chars: int = 3000) -> str:
        """Compact text for the slowly refreshed facts block of a prompt."""
        rows = self.view(recipe_id, limit=60)
        if not rows:
            return "(no shared facts yet)"
        lines = []
        for r in rows:
            where = "this recipe" if r["recipe_id"] == recipe_id else r["recipe_id"]
            lines.append(f"[{r['fact_id']}] ({r['lane']}, {r['status']}, {where}, {r.get('gpu_class') or '?'}) "
                         f"{r['claim']} :: {str(r['evidence'])[:160]}")
        out, n = [], 0
        for l in lines:
            if n + len(l) > max_chars:
                out.append(f"... {len(lines) - len(out)} more (use the facts tool)")
                break
            out.append(l)
            n += len(l) + 1
        return "\n".join(out)
