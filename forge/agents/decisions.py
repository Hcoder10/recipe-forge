"""Triggers, the forced decision log, the human inbox and the team note board."""
from __future__ import annotations

import itertools
import threading
from pathlib import Path

from forge.agents import events
from forge.tools.registry import ToolError

DECISIONS = ("continue", "replan", "stop", "ask_human")
SYNONYMS = {"proceed": "continue", "keep_going": "continue", "go": "continue", "ok": "continue",
            "re-plan": "replan", "re_plan": "replan", "pivot": "replan", "change_plan": "replan",
            "halt": "stop", "end": "stop", "finish": "stop", "abort": "stop", "done": "stop",
            "ask": "ask_human", "escalate": "ask_human", "human": "ask_human"}


class Decisions:
    """Open triggers need a Lead decision. Every Lead turn lists the pending ids; decide() closes them.
    Unknown or already-closed ids are a no-op, not an error."""

    def __init__(self, path: Path, log):
        self.path = Path(path)
        self.log = log
        self.lock = threading.Lock()
        self.triggers: dict[str, dict] = {}
        self.counter = itertools.count(1)
        self.stop_requested: dict | None = None
        self.rows: list[dict] = []

    def trigger(self, kind: str, summary: str, source: str = "", infra: bool = False) -> str:
        with self.lock:
            tid = f"T{next(self.counter)}"
            self.triggers[tid] = {"id": tid, "kind": kind, "summary": summary[:400], "source": source,
                                  "infra": infra, "open": True}
        self.log("trigger", trigger_id=tid, trigger_kind=kind, summary=summary[:400], source=source, infra=infra)
        return tid

    def close(self, tid: str, why: str = "auto") -> None:
        with self.lock:
            t = self.triggers.get(tid)
            if t and t["open"]:
                t["open"] = False
                t["closed_by"] = why

    def pending(self) -> list[dict]:
        with self.lock:
            return [dict(t) for t in self.triggers.values() if t["open"]]

    def pending_text(self) -> str:
        p = self.pending()
        if not p:
            return "PENDING TRIGGERS: none"
        return "PENDING TRIGGERS (each needs decide()): " + "; ".join(
            f"{t['id']} [{t['kind']}{', infra' if t['infra'] else ''}] {t['summary']}" for t in p)

    def decide(self, by: str, decision: str, trigger_ids, evidence: str, plan: str = "") -> dict:
        d = (decision or "").strip().lower().replace(" ", "_")
        d = SYNONYMS.get(d, d)
        if d not in DECISIONS:
            raise ToolError(f"decision must be one of {', '.join(DECISIONS)}")
        if isinstance(trigger_ids, str):
            trigger_ids = [trigger_ids]
        ids = [str(t).strip() for t in (trigger_ids or [])]
        closed, ignored = [], []
        with self.lock:
            for tid in ids:
                t = self.triggers.get(tid) or self.triggers.get(tid.upper()) or self.triggers.get("T" + tid)
                if t and t["open"]:
                    t["open"] = False
                    t["closed_by"] = by
                    closed.append(t["id"])
                else:
                    ignored.append(tid)
        row = events.append(self.path, {"by": by, "decision": d, "trigger_ids": closed, "ignored": ignored,
                                        "evidence": evidence[:1000], "plan": plan[:1500]})
        self.rows.append(row)
        self.log("decision", by=by, decision=d, trigger_ids=closed, evidence=evidence[:500], plan=plan[:500])
        if d == "stop":
            self.stop_requested = row
        return {"recorded": d, "closed": closed, "ignored_ids": ignored, "still_pending": [t["id"] for t in
                                                                                          self.pending()]}


class Inbox:
    """ask_human goes to a file a human answers through (forge CLI or dash). Never blocks the run."""

    def __init__(self, path: Path, answers_path: Path, log):
        self.path = Path(path)
        self.answers_path = Path(answers_path)
        self.log = log
        self.counter = itertools.count(1)
        self.seen: set[str] = set()
        self.prefix = Path(path).parent.name[:12]

    def ask(self, by: str, question: str) -> str:
        qid = f"Q{next(self.counter)}-{self.prefix}"
        events.append(self.path, {"question_id": qid, "by": by, "question": question[:1500]})
        self.log("ask_human", question_id=qid, by=by, question=question[:500])
        return qid

    def new_answers(self) -> list[dict]:
        out = []
        for r in events.read(self.answers_path):
            qid = r.get("question_id")
            if qid and qid not in self.seen:
                self.seen.add(qid)
                out.append(r)
        return out


class Notes:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.counter = itertools.count(1)
        self.rows: list[dict] = []
        self.lock = threading.Lock()

    def add(self, by: str, text: str) -> str:
        with self.lock:
            nid = f"N{next(self.counter)}"
            row = events.append(self.path, {"note_id": nid, "by": by, "text": text[:800]})
            self.rows.append(row)
        return nid

    def recent(self, n: int = 12) -> list[dict]:
        with self.lock:
            return self.rows[-n:]
