"""Append-only JSONL logs (events, decisions, inbox, notes, facts). One writer lock per file per process."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def _lock(path: Path) -> threading.Lock:
    with _GUARD:
        return _LOCKS.setdefault(str(path), threading.Lock())


def append(path: Path, row: dict) -> dict:
    path = Path(path)
    row = {"t": round(time.time(), 3), **row}
    line = json.dumps(row, default=str)
    with _lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return row


def read(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a torn last line from a crash
    return out


class EventLog:
    """events.jsonl for one mission. Kinds used: mission_start, done_locked, llm_call, tool_call, tool_bug,
    detector, decision, trigger, gate_request, gate_review, gate_result, compaction, ask_human, worker_task,
    worker_report, budget, mission_end, error."""

    def __init__(self, path: Path, mission_id: str):
        self.path = Path(path)
        self.mission_id = mission_id
        self.listeners = []

    def __call__(self, kind: str, **fields) -> dict:
        row = append(self.path, {"mission_id": self.mission_id, "kind": kind, **fields})
        for fn in self.listeners:
            try:
                fn(row)
            except Exception:
                pass
        return row
