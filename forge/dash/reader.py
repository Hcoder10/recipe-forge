"""Read-only snapshot of everything under FORGE_ROOT that the dashboard shows.

Nothing here writes. Every source is optional: a missing or half-written file yields an empty panel and
a note in ``sources``, never an exception. JSONL files are tailed incrementally (only appended bytes are
parsed), so polling once a second stays cheap as event logs grow.

Field names are matched loosely (``_pick`` takes the first key present) because the writers' formats are
still settling; the canonical names are the first ones listed and are written up in docs/interfaces/dash.md.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Any, Iterable

MAX_FIRST_READ = 8 * 1024 * 1024  # a log bigger than this is read from its last 8 MB only
FEED_LEN = 60

DECISIONS = ("continue", "replan", "stop", "ask_human")
LIVE_STATUSES = {"running", "active", "started", "live"}
END_TYPES = {"mission_end", "end", "done", "finished", "mission_done", "row"}
START_TYPES = {"mission_start", "start", "mission_started"}
STEP_TYPES = {"step", "tool_call", "tool", "tool_start", "tool_result", "action"}
LLM_TYPES = {"llm_call", "llm", "llm_usage", "completion"}
DECISION_TYPES = {"decision", "decide"}
ALERT_TYPES = {"drift", "alert", "nudge", "escalate", "escalation", "infra", "infra_fault", "loop", "detector",
               "tool_bug", "trigger", "error"}
GATE_REQ_TYPES = {"gate_request", "request_gate", "gate_requested", "gate_submitted"}
GATE_VERDICT_TYPES = {"gate_verdict", "verdict", "gate_result"}
FACT_TYPES = {"fact", "fact_add", "fact_added"}
ASK_TYPES = {"ask_human"}
TASK_TYPES = {"worker_task"}
ENDING_TYPES = {"mission_ending"}
# where the writers keep state: FORGE_ROOT itself, or FORGE_ROOT/state
PREFIXES = ("", "state/")


# ---------------------------------------------------------------- small helpers

def _pick(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _ts(v: Any) -> float | None:
    """Unix seconds from a float/int (s or ms) or an ISO-8601 string."""
    if v is None:
        return None
    f = _num(v)
    if f is not None:
        return f / 1000.0 if f > 1e11 else f
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _etype(ev: dict) -> str:
    return str(_pick(ev, "type", "kind", "event", default="")).lower()


def _short(v: Any, n: int = 160) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v)
    elif not isinstance(v, str):
        v = json.dumps(v, default=str) if isinstance(v, dict) else str(v)
    v = " ".join(v.split())
    return v if len(v) <= n else v[: n - 1] + "…"


def _count(v: Any) -> Any:
    return len(v) if isinstance(v, (list, tuple, dict)) else v


def _name(v: Any) -> Any:
    return _pick(v, "name", "building", "id") if isinstance(v, dict) else v


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class JsonlTail:
    """Incremental reader of an append-only JSONL file. Tolerates partial last lines and truncation."""

    def __init__(self, path: str):
        self.path = path
        self.offset = 0
        self.inode = None
        self.bad_lines = 0
        self.truncated_head = False

    def poll(self) -> tuple[list[dict], bool]:
        """Return (new records, reset). ``reset`` means the file was replaced or shrank: rebuild state."""
        try:
            st = os.stat(self.path)
        except OSError:
            return [], False
        reset = False
        ident = (st.st_dev, st.st_ino)
        if self.inode is not None and (ident != self.inode or st.st_size < self.offset):
            self.offset, reset = 0, True
        self.inode = ident
        if st.st_size == self.offset:
            return [], reset
        out: list[dict] = []
        with open(self.path, "rb") as f:
            start = self.offset
            skip_first = False
            if start == 0 and st.st_size > MAX_FIRST_READ:
                start, skip_first, self.truncated_head = st.st_size - MAX_FIRST_READ, True, True
            f.seek(start)
            data = f.read(st.st_size - start)
        end = data.rfind(b"\n")
        if end < 0:
            return [], reset  # no complete line yet
        chunk, self.offset = data[: end + 1], start + end + 1
        lines = chunk.split(b"\n")
        if skip_first:
            lines = lines[1:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                self.bad_lines += 1
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out, reset


# ---------------------------------------------------------------- mission state

class MissionState:
    """Folds a mission's events.jsonl (plus spec.json / mission.json / row.json) into the Live Missions view."""

    def __init__(self, mid: str):
        self.id = mid
        self.meta: dict = {}
        self.agents: dict[str, dict] = {}
        self.usd = 0.0
        self.usd_by_role: Counter = Counter()
        self.tokens = Counter()
        self.decisions: deque = deque(maxlen=FEED_LEN)
        self.alerts: deque = deque(maxlen=FEED_LEN)
        self.gate_requests: dict[str, dict] = {}
        self.verdicts: list[dict] = []
        self.facts = 0
        self.steps = 0
        self.ended: dict | None = None
        self.ending: str | None = None
        self.row: dict | None = None
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        self.lane: str | None = None
        self.spend_points: list[tuple[float, float, str]] = []  # (ts, usd, role)

    def _agent(self, ev: dict, member: bool = False) -> dict | None:
        if member:
            name = _pick(ev, "name", "agent", "id")
        else:
            name = _pick(ev, "agent", "worker", "member", "who", "by")  # worker_task: by=lead, worker=w1
        role = _pick(ev, "role")
        if name is None and role is None:
            return None
        name = str(name or role)
        a = self.agents.setdefault(name, {"name": name, "role": None, "model": None, "step": None,
                                          "tool": None, "lane": None, "task": None, "last_ts": None,
                                          "calls": 0, "tools": 0, "usd": 0.0})
        if role:
            a["role"] = str(role)
        elif a["role"] is None:
            a["role"] = _role_from_name(name)
        if _pick(ev, "model"):
            a["model"] = str(ev["model"])
        return a

    def absorb_meta(self, d: dict) -> None:
        """mission.json / spec.json / mission_start: flatten recipe and machine dicts into plain fields."""
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if v is None or k in ("type", "kind", "event"):
                continue
            if k == "recipe" and isinstance(v, dict):
                self.meta["recipe"] = _pick(v, "id", "recipe_id")
                self.meta.setdefault("model", _pick(v, "model"))
            elif k == "machine" and isinstance(v, dict):
                self.meta["machine"] = _pick(v, "name", "building")
                if _pick(v, "gpu_class"):
                    self.meta.setdefault("gpu_class", v["gpu_class"])
            elif k == "team":
                self.meta["team"] = v
                members = v if isinstance(v, list) else _pick(v, "members", default=[]) if isinstance(v, dict) else []
                for m in members or []:
                    if isinstance(m, dict):
                        self._agent(m, member=True)
            elif k in ("t", "ts") and "started_at" not in self.meta:
                self.meta["started_at"] = v
            else:
                self.meta[k] = v

    def feed(self, ev: dict) -> None:
        t = _etype(ev)
        ts = _ts(_pick(ev, "ts", "t", "time", "timestamp"))
        if ts is not None:
            self.first_ts = ts if self.first_ts is None else min(self.first_ts, ts)
            self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)
        if _pick(ev, "lane") and t not in TASK_TYPES:
            self.lane = str(ev["lane"])
        a = self._agent(ev)
        if a is not None and ts is not None:
            a["last_ts"] = ts
        name = (a or {}).get("name")

        if t in START_TYPES:
            self.absorb_meta({k: v for k, v in ev.items() if k not in self.meta or k == "team"})
        elif t in LLM_TYPES:
            usd = _num(_pick(ev, "usd", "cost_usd", "cost", "dollars")) or 0.0
            role = str(_pick(ev, "role", default=(a or {}).get("role") or "unknown"))
            self.usd += usd
            self.usd_by_role[role] += usd
            for k in ("prompt_tokens", "cached_tokens", "completion_tokens"):
                self.tokens[k] += int(_num(ev.get(k)) or 0)
            if a is not None:
                a["calls"] += 1
                a["usd"] += usd
            if ts is not None and usd:
                self.spend_points.append((ts, usd, role))
        elif t in DECISION_TYPES:
            self.decisions.append({
                "ts": ts, "mission": self.id, "agent": name,
                "decision": str(_pick(ev, "decision", "choice", "value", default="?")).lower(),
                "rationale": _short(_pick(ev, "rationale", "reason", "why", "plan", "text"), 280),
                "evidence": _short(_pick(ev, "evidence", "evidence_ids", "trigger_ids"), 120),
            })
        elif t in ASK_TYPES:
            self.decisions.append({"ts": ts, "mission": self.id, "agent": name, "decision": "ask_human",
                                   "rationale": _short(_pick(ev, "question", "text", "rationale"), 280),
                                   "evidence": _short(_pick(ev, "question_id"), 60)})
        elif t in ALERT_TYPES:
            self.alerts.append(_alert(ev, ts, self.id, name))
        elif t in GATE_REQ_TYPES:
            gid = _pick(ev, "gate_id", "request_id")
            if gid is not None:
                rid = str(gid)
                self.gate_requests[rid] = {"request_id": rid, "ts": ts, "mission": self.id,
                                           "bundle": _short(_pick(ev, "bundle_hash", "bundle"), 16),
                                           "machine": _pick(ev, "machine", default=self.meta.get("machine")),
                                           "claim": _short(_pick(ev, "claim", "change", "summary"), 120)}
        elif t in GATE_VERDICT_TYPES:
            v = normalize_verdict(ev, machine=self.meta.get("machine"))
            v["mission"] = self.id
            v["recipe"] = v["recipe"] or self.meta.get("recipe")
            self.verdicts.append(v)
            self.gate_requests.pop(v["verdict_id"], None)
            rid = _pick(ev, "request_id")
            if rid is not None:
                self.gate_requests.pop(str(rid), None)
        elif t in FACT_TYPES:
            self.facts += 1
        elif t in TASK_TYPES and a is not None:
            a["lane"] = _pick(ev, "lane", default=a["lane"])
            a["task"] = _short(_pick(ev, "task", "summary"), 120)
            if a["lane"]:
                self.lane = str(a["lane"])
        elif t in ENDING_TYPES:
            self.ending = str(_pick(ev, "reason", default="ending"))
        elif t in END_TYPES:
            self.ended = ev
            if isinstance(ev.get("row"), dict):
                self.row = ev["row"]

        if t in STEP_TYPES:
            self.steps += 1
            if a is not None:
                a["tools"] += 1
        if a is not None:
            step = _pick(ev, "step", "step_n", "turn")
            if step is not None:
                a["step"] = step
            elif t in STEP_TYPES:
                a["step"] = a["tools"]
            tool = _pick(ev, "tool", "tool_name") or (ev.get("name") if t in STEP_TYPES else None)
            if tool is not None:
                a["tool"] = str(tool)

    def view(self, now: float) -> dict:
        m = self.meta
        started = _ts(_pick(m, "started_at", "start_ts")) or self.first_ts
        box_min = _num(_pick(m, "time_box_min", "time_box_minutes", "timebox_min"))
        deadline = _ts(_pick(m, "deadline", "deadline_ts"))
        if deadline is None and started is not None and box_min:
            deadline = started + box_min * 60
        status = str(_pick(m, "status", default="running")).lower()
        if self.ending and not self.ended:
            status = "ending"
        if self.ended or self.row:
            row = self.row or {}
            done = row.get("done") if isinstance(row.get("done"), dict) else {}
            status = str(_pick(done, "status", default=_pick(self.ended or {}, "status", "outcome",
                                                              default="done"))).lower()
        team_meta = _pick(m, "team", "members", default=[])
        if isinstance(team_meta, dict):
            team_meta = _pick(team_meta, "members", default=[])
        team = []
        for meta_m in team_meta or []:
            if isinstance(meta_m, dict):
                name = str(_pick(meta_m, "name", "agent", "id", "role", default="?"))
                team.append(self.agents.get(name) or {"name": name, "role": meta_m.get("role"),
                                                       "model": meta_m.get("model")})
        seen = {t["name"] for t in team}
        team += [a for n, a in self.agents.items() if n not in seen and n not in ("human", "system", "gate")]
        team.sort(key=lambda a: (ROLE_ORDER.get(str(a.get("role")), 9), a["name"]))
        usd = self.usd
        if not usd and self.row:
            usd = _num(_pick(self.row, "cost_usd")) or 0.0
        return {
            "id": self.id,
            "recipe": _pick(m, "recipe", "recipe_id"),
            "model": _pick(m, "model"),
            "machine": _pick(m, "machine", "building"),
            "gpu_class": _pick(m, "gpu_class", "gpu"),
            "gpus": _pick(m, "gpus"),
            "lane": _pick(m, "lane") or self.lane,
            "team": team,
            "usd": round(usd, 4),
            "budget_usd": _num(_pick(m, "budget_usd", "llm_budget_usd", "budget")),
            "started_at": started,
            "deadline": deadline,
            "time_left_s": None if deadline is None else deadline - now,
            "status": status,
            "ending": self.ending,
            "steps": self.steps,
            "facts": self.facts,
            "last_ts": self.last_ts,
            "pending_gate": len(self.gate_requests),
            "tokens": dict(self.tokens),
            "playbook": _short(_pick(m, "playbook"), 12) or None,
        }


ROLE_ORDER = {"lead": 0, "worker": 1, "verifier": 2, "mechanic": 3}


def _role_from_name(name: str) -> str | None:
    n = name.lower()
    for role in ROLE_ORDER:
        if n.startswith(role):
            return role
    return "worker" if n[:1] == "w" and n[1:].isdigit() else None


def _alert(ev: dict, ts: float | None, mission: str | None, agent: str | None) -> dict:
    et = _etype(ev)
    fault = str(_pick(ev, "fault", "class", "category", default="")).lower()
    kind = str(_pick(ev, "detector", "signal", "trigger_kind", "kind_detail", "what", default=et))
    infra = ev.get("infra")
    if not fault:
        if isinstance(infra, bool):
            fault = "infra" if infra else "agent"
        else:
            fault = "infra" if ("infra" in et or et in ("tool_bug", "error") or
                                kind in ("env_blocked", "tool_bug")) else "agent"
    sev = str(_pick(ev, "severity", "level", default="")).lower()
    if not sev:
        sev = "critical" if fault == "infra" else "warning"
    return {"ts": ts, "mission": mission, "agent": agent, "fault": fault, "detector": kind,
            "severity": sev, "action": _pick(ev, "action", "response"),
            "message": _short(_pick(ev, "message", "msg", "summary", "detail", "text", "error"), 220),
            "machine": _pick(ev, "machine", "building")}


def normalize_verdict(v: dict, machine: str | None = None) -> dict:
    lossless = _pick(v, "lossless", "lossless_ok")
    if isinstance(lossless, dict):
        ok = _pick(lossless, "ok", "pass", "passed")
        detail = {k: lossless.get(k) for k in ("top1", "kl", "max_regret") if k in lossless}
    else:
        ok, detail = lossless, {}
    win = _pick(v, "win", "passed", "pass")
    verdict = _pick(v, "verdict", "result")
    if isinstance(verdict, str):
        vl = verdict.lower()
        if win is None:
            win = vl in ("win", "pass", "passed", "accept", "accepted")
        if ok is None and vl == "lossy":
            ok = False
    pairs = _pick(v, "pairs", "n_pairs", "n")
    pair_gains = _pick(v, "pair_gains", "pair_log_gains", "gains")
    if isinstance(pairs, list):
        pair_gains, pairs = pairs, len(pairs)
    gain = _num(_pick(v, "mean_log_gain", "log_gain", "gain"))
    delta_pct = _num(_pick(v, "delta_pct"))
    if delta_pct is None and gain is not None:
        delta_pct = (math.exp(gain) - 1) * 100
    return {
        "verdict_id": str(_pick(v, "verdict_id", "gate_id", "evidence_id", "id", default="?")),
        "ts": _ts(_pick(v, "ts", "t", "time", "finished_at")),
        "machine": _pick(v, "machine", default=machine),
        "recipe": _pick(v, "recipe", "recipe_id"),
        "mission": _pick(v, "mission", "mission_id"),
        "kind": _pick(v, "gate_kind", "kind", default="candidate") if _etype(v) not in GATE_VERDICT_TYPES
        else _pick(v, "gate_kind", default="candidate"),
        "verdict": verdict if isinstance(verdict, str) else None,
        "bundle": _short(_pick(v, "bundle_hash", "bundle"), 16),
        "pairs": int(_num(pairs)) if _num(pairs) is not None else None,
        "pairs_positive": _pick(v, "pairs_positive", "positive_pairs"),
        "pair_gains": [g for g in (_num(x) for x in pair_gains or []) if g is not None][:32],
        "mean_log_gain": gain,
        "delta_pct": delta_pct,
        "threshold": _num(_pick(v, "threshold", "bar")),
        "win": None if win is None else bool(win),
        "lossless": None if ok is None else bool(ok),
        "lossless_detail": detail,
        "confirmed": _pick(v, "confirmed"),
        "tok_s_ref": _num(_pick(v, "tok_s_ref", "ref_tok_s", "baseline_tok_s")),
        "tok_s_cand": _num(_pick(v, "tok_s_cand", "cand_tok_s", "candidate_tok_s")),
        "reason": _short(_pick(v, "reason", "why", "note"), 160),
    }


# ---------------------------------------------------------------- the reader

class Reader:
    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.tails: dict[str, JsonlTail] = {}
        self.missions: dict[str, MissionState] = {}
        self.mission_tail_paths: dict[str, str] = {}
        self.lines: dict[str, list[dict]] = {}  # non-mission jsonl sources, path -> records (bounded)
        self.sources: dict[str, dict] = {}
        self.meta_mtimes: dict[str, float] = {}
        self.spend_series: list[tuple[float, float]] = []

    # -- file plumbing
    def _p(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def _glob(self, pattern: str) -> list[str]:
        """Glob under the root, and under root/state for state files (code under forge/ only at the root)."""
        prefixes = ("",) if pattern.startswith("forge/") else PREFIXES
        out = []
        for pre in prefixes:
            out += glob.glob(self._p(pre + pattern))
        return sorted(dict.fromkeys(out))

    def _note(self, name: str, path: str | None, n: int, err: str | None = None) -> None:
        rel = os.path.relpath(path, self.root) if path else None
        cur = self.sources.get(name)
        if cur and cur["n"] and not n and not err:
            return
        self.sources[name] = {"path": rel, "n": n, "ok": err is None and n > 0, "err": err}

    def _jsonl(self, path: str, keep: int = 5000) -> list[dict]:
        tail = self.tails.get(path)
        if tail is None:
            tail = self.tails[path] = JsonlTail(path)
        new, reset = tail.poll()
        buf = [] if reset else self.lines.get(path, [])
        buf.extend(new)
        if len(buf) > keep:
            buf = buf[-keep:]
        self.lines[path] = buf
        return buf

    def _jsonl_many(self, name: str, patterns: Iterable[str], keep: int = 5000) -> list[dict]:
        out: list[dict] = []
        paths = list(dict.fromkeys(p for pat in patterns for p in self._glob(pat)))
        for p in paths:
            try:
                recs = self._jsonl(p, keep)
                for r in recs:
                    r.setdefault("_src", os.path.relpath(p, self.root))
                out.extend(recs)
            except OSError as e:
                self._note(name, p, 0, str(e))
        self._note(name, paths[0] if paths else None, len(out))
        return out

    def _json_any(self, name: str, patterns: Iterable[str]) -> list[tuple[str, Any]]:
        out = []
        for pat in patterns:
            for p in self._glob(pat):
                try:
                    out.append((p, _read_json(p)))
                except (OSError, ValueError) as e:  # half-written file: skip this poll
                    self._note(name, p, 0, f"{type(e).__name__}")
        if out:
            self._note(name, out[0][0], len(out))
        elif name not in self.sources:
            self._note(name, None, 0)
        return out

    # -- sources
    def _poll_missions(self) -> None:
        dirs = [d for d in self._glob("missions/*/")
                if any(os.path.exists(os.path.join(d, f)) for f in ("events.jsonl", "spec.json", "mission.json",
                                                                       "row.json"))]  # skip bare workspaces
        for d in dirs:
            mid = os.path.basename(os.path.normpath(d))
            ms = self.missions.get(mid)
            if ms is None:
                ms = self.missions[mid] = MissionState(mid)
            for fname in ("spec.json", "mission.json", "row.json"):
                path = os.path.join(d, fname)
                try:
                    mtime = os.stat(path).st_mtime
                except OSError:
                    continue
                if self.meta_mtimes.get(path) == mtime:
                    continue
                try:
                    doc = _read_json(path)
                except (OSError, ValueError):
                    continue  # half-written: try again next poll
                self.meta_mtimes[path] = mtime
                if fname == "row.json":
                    ms.row = doc if isinstance(doc, dict) else None
                else:
                    ms.absorb_meta(doc)
            ev_path = os.path.join(d, "events.jsonl")
            tail = self.tails.get(ev_path)
            if tail is None:
                tail = self.tails[ev_path] = JsonlTail(ev_path)
            new, reset = tail.poll()
            if reset:
                meta, row = ms.meta, ms.row
                ms = self.missions[mid] = MissionState(mid)
                ms.meta, ms.row = meta, row
            for ev in new:
                try:
                    ms.feed(ev)
                except Exception:  # one odd event must not take the panel down
                    tail.bad_lines += 1
        self._note("missions", dirs[0] if dirs else None, len(dirs))

    def _recipes(self) -> list[dict]:
        paths = self._glob("forge/recipes/*.yaml") + self._glob("forge/recipes/*.yml") + \
            self._glob("forge/recipes/*.json")
        out = []
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    doc = json.load(f) if p.endswith(".json") else (yaml.safe_load(f) if yaml else None)
            except Exception:
                continue
            if isinstance(doc, dict):
                doc.setdefault("id", os.path.splitext(os.path.basename(p))[0])
                out.append(doc)
        self._note("recipes", paths[0] if paths else None, len(out))
        return out

    # -- assembly
    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        self._poll_missions()
        missions = [ms.view(now) for ms in self.missions.values()]
        missions.sort(key=lambda m: (m["status"] not in LIVE_STATUSES, -(m["last_ts"] or 0)))

        rows = self._jsonl_many("leaderboard", ["results/leaderboard.jsonl", "results/rows/*.jsonl",
                                                "results/*/rows.jsonl"])
        for _, d in self._json_any("rows", ["results/rows/*.json", "results/boxes/*/rows/*.json",
                                           "results/leaderboard.json"]):
            items = d if isinstance(d, list) else (_pick(d, "rows", default=None) if isinstance(d, dict) else None)
            rows += [x for x in (items if items is not None else [d]) if isinstance(x, dict)]
        for ms in self.missions.values():  # missions/<id>/row.json, or the row in mission_end
            if ms.row:
                rows.append({**ms.meta, "mission_id": ms.id, **ms.row})
        rows = _dedupe_rows(rows)
        queue = []
        for _, q in self._json_any("queue", ["results/queue.json", "fleet/queue.json"]):
            queue += q if isinstance(q, list) else (_pick(q, "queued", "missions", "queue", default=[]) or [])

        leaderboard = build_leaderboard(self._recipes(), rows, queue, missions)
        gates = self._gates()
        decisions, alerts = self._feeds()
        brain = self._brain()
        n1 = self._n1()
        spend = self._spend(now)
        fixture = os.path.exists(self._p("FIXTURE"))
        return {
            "generated_at": now,
            "root": self.root,
            "fixture": fixture,
            "leaderboard": leaderboard,
            "missions": missions[:24],
            "gates": gates,
            "decisions": decisions,
            "alerts": alerts,
            "n1": n1,
            "brain": brain,
            "spend": spend,
            "sources": self.sources,
        }

    def _gates(self) -> list[dict]:
        by_m: dict[str, dict] = {}

        def g(machine: str) -> dict:
            return by_m.setdefault(machine, {"machine": machine, "running": None, "queued": [],
                                             "verdicts": {}, "quiet": None})

        for d in self._glob("results/gate/*/"):
            machine = os.path.basename(os.path.normpath(d))
            g(machine)
        for r in self._jsonl_many("gate_verdicts", ["results/gate/*/verdicts.jsonl", "results/gate/*.jsonl"]):
            src = r.get("_src", "")
            parts = src.replace("\\", "/").split("/")
            machine = _pick(r, "machine", default=parts[2] if len(parts) > 3 else "?")
            v = normalize_verdict(r, machine)
            g(str(v["machine"]))["verdicts"][v["verdict_id"]] = v
        for p, q in self._json_any("gate_queue", ["results/gate/*/queue.json"]):
            machine = os.path.basename(os.path.dirname(p))
            if isinstance(q, dict):
                gm = g(_pick(q, "machine", default=machine))
                gm["_has_queue"] = True
                gm["running"] = _pick(q, "running", "current")
                gm["queued"] = list(_pick(q, "queued", "queue", "pending", default=[]) or [])
                gm["quiet"] = _pick(q, "quiet", "quiet_ok")
            elif isinstance(q, list):
                g(machine).update(queued=q, _has_queue=True)
        for ms in self.missions.values():
            for v in ms.verdicts:
                gm = g(str(v.get("machine") or ms.meta.get("machine") or "?"))
                gm["verdicts"].setdefault(v["verdict_id"], v)
            for req in ms.gate_requests.values():  # only used when the gate publishes no queue.json
                gm = g(str(req.get("machine") or ms.meta.get("machine") or "?"))
                gm.setdefault("_pending", []).append(req)
        out = []
        for gm in by_m.values():
            pending = gm.pop("_pending", [])
            if not gm.pop("_has_queue", False):
                gm["queued"] = sorted(pending, key=lambda r: r.get("ts") or 0)
            vs = sorted(gm["verdicts"].values(), key=lambda v: -(v["ts"] or 0))
            gm["n_verdicts"] = len(vs)
            gm["n_wins"] = sum(1 for v in vs if v["win"] and v["lossless"] is not False)
            gm["verdicts"] = vs[:10]
            out.append(gm)
        out.sort(key=lambda x: x["machine"])
        return out

    def _feeds(self) -> tuple[list[dict], list[dict]]:
        decisions = [d for ms in self.missions.values() for d in ms.decisions]
        alerts = [a for ms in self.missions.values() for a in ms.alerts]
        for r in self._jsonl_many("alerts", ["alerts.jsonl", "results/alerts.jsonl", "fleet/alerts.jsonl"], 500):
            alerts.append(_alert(r, _ts(_pick(r, "ts", "time")), _pick(r, "mission"), _pick(r, "agent")))
        for r in self._jsonl_many("decisions", ["results/decisions.jsonl"], 500):
            decisions.append({"ts": _ts(_pick(r, "ts")), "mission": _pick(r, "mission"), "agent": _pick(r, "agent"),
                              "decision": str(_pick(r, "decision", default="?")).lower(),
                              "rationale": _short(_pick(r, "rationale", "reason"), 280),
                              "evidence": _short(_pick(r, "evidence"), 120)})
        decisions.sort(key=lambda d: -(d["ts"] or 0))
        alerts.sort(key=lambda a: -(a["ts"] or 0))
        return decisions[:FEED_LEN], alerts[:FEED_LEN]

    def _brain(self) -> dict:
        facts = self._jsonl_many("brain", ["brain/facts.jsonl", "results/brain/facts.jsonl", "brain/*.jsonl"])
        latest: dict[str, dict] = {}
        for i, f in enumerate(facts):  # later lines update earlier ones with the same id
            fid = str(_pick(f, "fact_id", "id", default=f"#{i}"))
            prev = latest.get(fid, {})
            latest[fid] = {**prev, **{k: v for k, v in f.items() if v is not None}, "_id": fid}
        items = []
        for f in latest.values():
            items.append({"id": f["_id"], "claim": _short(_pick(f, "claim", "text", "fact"), 200),
                          "recipe": _pick(f, "recipe_id", "recipe"), "lane": _pick(f, "lane"),
                          "gpu_class": _pick(f, "gpu_class"), "by": _pick(f, "by"),
                          "status": str(_pick(f, "status", default="open")).lower(),
                          "mission": _pick(f, "mission_id", "mission", "source_mission"),
                          "evidence": _short(_pick(f, "evidence", "evidence_ids"), 80),
                          "ts": _ts(_pick(f, "t", "ts", "updated_at", "time"))})
        items.sort(key=lambda f: -(f["ts"] or 0))
        counts = Counter(f["status"] for f in items)
        return {"total": len(items), "by_status": dict(counts.most_common()), "latest": items[:14]}

    def _n1(self) -> dict:
        recs = self._jsonl_many("n1", ["n1/log.jsonl", "n1/versions.jsonl", "results/n1/versions.jsonl"])
        latest: dict[str, dict] = {}
        for r in recs:  # a version can get several lines (proposed, then compared); merge them
            vid = _pick(r, "version", "new_hash", "new", "hash", "playbook", "id")
            vid = str(vid if vid is not None else len(latest))
            latest[vid] = {**latest.get(vid, {}), **{k: v for k, v in r.items() if v is not None}}
        out = []
        for vid, r in latest.items():
            unseen = []
            per = _pick(r, "unseen", "per_recipe", "heldout", "held_out", default=[]) or []
            if isinstance(per, dict):
                per = [dict(v, recipe=k) if isinstance(v, dict) else {"recipe": k, "after": v} for k, v in per.items()]
            for u in per:
                if isinstance(u, dict):
                    unseen.append({"recipe": _pick(u, "recipe", "recipe_id"),
                                   "before": _num(_pick(u, "before", "old", "prev", "base")),
                                   "after": _num(_pick(u, "after", "new", "cand"))})
            status = str(_pick(r, "status", "verdict", "decision", default="candidate")).lower()
            status = {"promote": "promoted", "reject": "rejected", "proposed": "candidate",
                      "propose": "candidate"}.get(status, status)
            label = vid if len(vid) <= 10 else vid[:8]
            parent = _pick(r, "parent", "old_hash", "old", "prev")
            out.append({"version": label, "full": vid,
                        "parent": parent if not isinstance(parent, str) or len(parent) <= 10 else parent[:8],
                        "status": status,
                        "summary": _short(_pick(r, "change_summary", "summary", "change", "reason"), 220),
                        "paired_delta": _num(_pick(r, "paired_delta", "delta", "mean_delta")),
                        "missions": _count(_pick(r, "n_missions", "missions")),
                        "unseen": unseen, "ts": _ts(_pick(r, "t", "ts", "time"))})
        out.sort(key=lambda v: (v["ts"] or 0))
        return {"versions": out[-12:], "current": next((v["version"] for v in reversed(out)
                                                        if v["status"] == "promoted"), None)}

    def _spend(self, now: float) -> dict:
        by_role: Counter = Counter()
        points: list[tuple[float, float]] = []
        for ms in self.missions.values():
            by_role.update(ms.usd_by_role)
            points += [(t, u) for t, u, _ in ms.spend_points]
        for r in self._jsonl_many("llm_calls", ["results/llm_calls.jsonl", "llm/calls.jsonl"]):
            usd = _num(_pick(r, "usd", "cost_usd")) or 0.0
            by_role[str(_pick(r, "role", default="unknown"))] += usd
            t = _ts(_pick(r, "ts"))
            if t is not None and usd:
                points.append((t, usd))
        points.sort()
        series, acc = [], 0.0
        for t, u in points:
            acc += u
            series.append((t, acc))
        if len(series) > 240:  # downsample, keeping the last point
            step = len(series) / 240
            series = [series[int(i * step)] for i in range(240)] + [series[-1]]

        rentals: dict[str, dict] = {}
        for _, ms_doc in self._json_any("machines", ["fleet/machines.json", "fleet/machines/*.json"]):
            items = ms_doc if isinstance(ms_doc, list) else (_pick(ms_doc, "machines", default=[]) or [])
            if isinstance(ms_doc, dict) and not items:
                if "name" in ms_doc:  # fleet/machines/<name>.json: one machine
                    items = [ms_doc]
                else:  # {name: machine}
                    items = [dict(v, name=k) for k, v in ms_doc.items() if isinstance(v, dict)]
            for m in items:
                if not isinstance(m, dict):
                    continue
                key = str(_pick(m, "rental_id", "instance", "name", default="?"))
                rentals[key] = {**rentals.get(key, {}), **m}
        for r in self._jsonl_many("vast_spend", ["fleet/spend.jsonl", "results/spend.jsonl"]):
            key = str(_pick(r, "rental_id", "instance", "machine", default="?"))
            rentals[key] = {**rentals.get(key, {}), **{k: v for k, v in r.items() if v is not None}}
        vast = []
        for key, m in rentals.items():
            rate = _num(_pick(m, "usd_per_hour", "usd_per_hr", "dph", "usd_hr", "rate"))
            if rate is None and not _pick(m, "rental_id", "instance", "spent_usd"):
                continue  # a local machine, not a rental
            started = _ts(_pick(m, "started_at", "start_ts"))
            ended = _ts(_pick(m, "returned_at", "ended_at"))
            active = ended is None and str(_pick(m, "status", default="active")).lower() not in (
                "returned", "destroyed", "stopped")
            spent = _num(_pick(m, "spent_usd", "total_usd", "spent"))
            if spent is None and rate is not None and started is not None:
                spent = rate * max(0.0, ((ended or now) - started)) / 3600
            vast.append({"rental": key, "machine": _pick(m, "name", "machine", "building", default=key),
                         "gpu_class": _pick(m, "gpu_class", "gpu"), "usd_per_hr": rate,
                         "spent_usd": spent, "active": active,
                         "hours_left": _num(_pick(m, "hours_left"))})
        vast.sort(key=lambda v: (not v["active"], v["machine"] or ""))
        crusoe_total = sum(by_role.values())
        return {
            "crusoe": {"by_role": {k: round(v, 4) for k, v in by_role.most_common()},
                       "total": round(crusoe_total, 4), "series": series},
            "vast": {"burn_usd_per_hr": round(sum(v["usd_per_hr"] or 0 for v in vast if v["active"]), 3),
                     "total_usd": round(sum(v["spent_usd"] or 0 for v in vast), 3),
                     "machines": vast[:16]},
        }


# ---------------------------------------------------------------- leaderboard

STATUS_RANK = {"verified": 4, "running": 3, "no-gain": 2, "failed": 1, "queued": 0}


def _norm_status(s: Any) -> str:
    s = str(s or "").lower().replace("_", "-")
    if s in ("verified", "win", "won", "pass", "champion"):
        return "verified"
    if s in ("no-gain", "nogain", "loss", "no-win", "rejected", "lossy", "fail-gate"):
        return "no-gain"
    if s in ("running", "active", "started", "live"):
        return "running"
    if s in ("failed", "error", "crashed", "infra"):
        return "failed"
    return "queued" if s in ("", "queued", "pending", "todo") else s


def _gpu_classes(recipe: dict) -> list[str]:
    hw = _pick(recipe, "hardware", "gpus", "gpu_classes", default=[])
    if isinstance(hw, dict):
        hw = _pick(hw, "classes", "gpu_classes", "fits", default=list(hw.keys()))
    if isinstance(hw, str):
        hw = [hw]
    return [str(h if not isinstance(h, dict) else _pick(h, "class", "gpu_class", "name")) for h in hw or []]


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    """The same mission's row can arrive from several files; keep the last one seen per mission."""
    by_id: dict[str, dict] = {}
    anon = []
    for r in rows:
        mid = _pick(r, "mission_id", "mission")
        if mid is None:
            anon.append(r)
        else:
            by_id[str(mid)] = {**by_id.get(str(mid), {}), **r}
    return anon + list(by_id.values())


def _row_status(row: dict) -> str:
    if _pick(row, "status") is not None:
        return _norm_status(row["status"])
    done = row.get("done")
    if isinstance(done, dict) and done.get("status"):
        return _norm_status(done["status"])
    return "queued"


def _row_gate(row: dict) -> dict:
    """The winning gate verdict of a row, if the row lists its verdicts."""
    vs = [v for v in (_pick(row, "gate_verdicts", "verdicts", default=[]) or []) if isinstance(v, dict)]
    wins = [v for v in vs if str(_pick(v, "verdict", default="")).lower() == "win" or v.get("win") is True]
    pool = wins or vs
    if not pool:
        return {}
    best = max(pool, key=lambda v: (_num(_pick(v, "delta_pct")) or -1e9))
    lossless = _pick(best, "lossless")
    if lossless is None:  # the gate only calls a win when the lossless check passed
        verdict = str(_pick(best, "verdict", default="")).lower()
        lossless = False if verdict == "lossy" else (True if best in wins else None)
    return {"verdict_id": _pick(best, "verdict_id", "gate_id", "evidence_id", "id"),
            "pairs": _pick(best, "pairs", "n_pairs"), "lossless": lossless}


def build_leaderboard(recipes: list[dict], rows: list[dict], queue: list, missions: list[dict]) -> list[dict]:
    table: dict[tuple[str, str], dict] = {}

    def cell(recipe: Any, gpu: Any) -> dict:
        key = (str(recipe or "?"), str(gpu or "?"))
        if key not in table:
            table[key] = {"recipe": key[0], "gpu_class": key[1], "model": None, "metric": None,
                          "baseline_tok_s": None, "best_tok_s": None, "delta_pct": None, "change": None,
                          "verdict_id": None, "pairs": None, "lossless": None, "status": "queued",
                          "mission": None, "machine": None, "ts": None, "attempts": 0, "running": 0}
        return table[key]

    for r in recipes:
        variants = [v for v in (_pick(r, "variants", default=[]) or []) if isinstance(v, dict) and v.get("name")]
        met = _pick(r, "metric", default=_pick(_pick(r, "workload", default={}) or {}, "metric"))
        met = _pick(met, "name", "kind") if isinstance(met, dict) else met
        expanded = [(f"{r.get('id')}.{v['name']}", _pick(v, "metric", default=met)) for v in variants] or \
            [(r.get("id"), met)]
        for rid, metric in expanded:
            for gpu in _gpu_classes(r) or ["?"]:
                c = cell(rid, gpu)
                c["model"] = _pick(r, "model")
                c["metric"] = metric
    for q in queue or []:
        if isinstance(q, dict):
            cell(_pick(q, "recipe", "recipe_id"), _pick(q, "gpu_class", "gpu"))
    for row in sorted(rows, key=lambda r: _ts(_pick(r, "ts", "t", "ended", "finished_at")) or 0):
        rid = _pick(row, "recipe_id", "recipe")
        if isinstance(rid, dict):
            rid = _pick(rid, "id")
        c = cell(rid, _pick(row, "gpu_class", "gpu"))
        c["attempts"] += 1
        status = _row_status(row)
        base = _num(_pick(row, "baseline_tok_s", "baseline", "nvidia_tok_s"))
        best = _num(_pick(row, "best_tok_s", "best", "verified_tok_s"))
        if base is not None:
            c["baseline_tok_s"] = base
        c["metric"] = c["metric"] or _pick(row, "metric")
        gate = _pick(row, "gate", "proof", default=None)
        if not isinstance(gate, dict):
            gate = _row_gate(row)
        if status != "verified":
            best = None  # rows report best = 0 when nothing won
        is_better = best is not None and (c["best_tok_s"] is None or best > c["best_tok_s"])
        if status == "verified" and (is_better or c["status"] != "verified"):
            c["best_tok_s"] = best
            c["change"] = _short(_pick(row, "change", "change_summary", "summary"), 140)
            c["verdict_id"] = _pick(row, "verdict_id", default=_pick(gate, "verdict_id", "id"))
            c["pairs"] = _pick(row, "pairs", default=_pick(gate, "pairs"))
            ll = _pick(row, "lossless", default=_pick(gate, "lossless"))
            c["lossless"] = _pick(ll, "ok", "pass") if isinstance(ll, dict) else ll
            c["mission"] = _pick(row, "mission_id", "mission")
            c["machine"] = _name(_pick(row, "machine"))
            c["ts"] = _ts(_pick(row, "ts", "t", "ended", "finished_at"))
        if STATUS_RANK.get(status, 0) >= STATUS_RANK.get(c["status"], 0):
            c["status"] = status
            if status != "verified" and c["change"] is None:
                c["change"] = _short(_pick(row, "change", "change_summary", "summary"), 140) or None
                c["mission"] = _pick(row, "mission_id", "mission")
    for m in missions:
        if m["status"] in LIVE_STATUSES and m.get("recipe"):
            c = cell(m["recipe"], m.get("gpu_class"))
            c["running"] += 1
            if c["status"] != "verified":
                c["status"] = "running"
                c["mission"] = c["mission"] or m["id"]
                c["machine"] = c["machine"] or m.get("machine")
    for c in table.values():
        b, x = c["baseline_tok_s"], c["best_tok_s"]
        c["delta_pct"] = round((x - b) / b * 100, 2) if b and x is not None else None
    out = list(table.values())
    out.sort(key=lambda c: (-STATUS_RANK.get(c["status"], 0), -(c["delta_pct"] or -1e9), c["recipe"], c["gpu_class"]))
    return out
