"""Version N+1: the Mechanic (Ultra) proposes a playbook revision from trace stats after each wave; the next
unseen recipes run with old vs new; a paired comparison promotes or rejects it. Every step is logged.

APIs (fleet schedules the waves):
  current(root) -> hash of the promoted playbook (base if none)
  trace_stats(mission_ids, root) -> per-mission and aggregate stats from events.jsonl / row.json
  propose(mission_ids, base_hash, root, client=None, profile='crusoe') -> new hash | None
  compare(old_hash, new_hash, rows, root) -> {verdict: promote|reject, per_recipe, reason}   (logged)
  score(row) -> float, the deterministic partial-credit score used by compare
Log: <state>/n1/log.jsonl rows {event: proposed|rejected_proposal|promote|reject, ...}.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from forge.agents import events, playbook as pb_mod
from forge.agents.config import profile as get_profile, load_secrets, state_root
from forge.agents.llm import LLMClient, Meter

EDITABLE = ("lead", "worker", "verifier", "protocol")
KNOB_BOUNDS = {"verifier_bar_pct": (1.0, 10.0), "facts_refresh_turns": (2, 30), "keep_turns": (4, 40)}
PROMOTE_MIN_MEAN = 0.25          # mean paired score delta needed
PROMOTE_NO_WORSE_FRAC = 0.8      # share of paired recipes that must be no worse
MIN_PAIRS = 2


def _root(root) -> Path:
    return Path(root or state_root())


def log_path(root=None) -> Path:
    return _root(root) / "n1" / "log.jsonl"


def current(root=None) -> str:
    cur = pb_mod.content_hash(pb_mod.DEFAULT)
    for r in events.read(log_path(root)):
        if r.get("event") == "promote":
            cur = r["new"]
    return cur


# ---------------------------------------------------------------------------------------------- stats
def mission_stats(mission_id: str, root=None) -> dict:
    d = _root(root) / "missions" / mission_id
    ev = events.read(d / "events.jsonl")
    row = json.loads((d / "row.json").read_text()) if (d / "row.json").exists() else {}
    tools = Counter()
    classes = Counter()
    signals = Counter()
    first_bench_step = None
    steps = 0
    t0 = ev[0]["t"] if ev else 0
    first_bench_min = None
    for e in ev:
        k = e.get("kind")
        if k == "tool_call":
            steps += 1
            tools[e.get("tool")] += 1
            if e.get("cls"):
                classes[e["cls"]] += 1
            if e.get("tool") == "bench" and e.get("ok") and first_bench_step is None:
                first_bench_step = steps
                first_bench_min = round((e["t"] - t0) / 60, 1)
        elif k == "detector":
            signals[e.get("signal")] += 1
    return {"mission_id": mission_id, "recipe_id": row.get("recipe_id"), "playbook": row.get("playbook"),
            "status": (row.get("done") or {}).get("status"), "delta_pct": row.get("delta_pct"),
            "quick_bench_best": (row.get("quick_bench_best") or [None])[0], "cost_usd": row.get("cost_usd"),
            "tool_calls": steps, "tools": dict(tools), "error_classes": dict(classes), "signals": dict(signals),
            "first_bench_step": first_bench_step, "first_bench_min": first_bench_min,
            "n_benches": row.get("n_benches"), "n_gate_requests": row.get("n_gate_requests"),
            "verdicts": [v.get("verdict") for v in row.get("gate_verdicts") or []],
            "compactions": sum(1 for e in ev if e.get("kind") == "compaction"),
            "minutes": row.get("minutes"), "score": score(row) if row else 0.0}


def trace_stats(mission_ids: list[str], root=None) -> dict:
    per = [mission_stats(m, root) for m in mission_ids]
    agg: dict = {"missions": len(per)}
    for key in ("tools", "error_classes", "signals"):
        c = Counter()
        for p in per:
            c.update(p[key])
        agg[key] = dict(c.most_common(20))
    for key in ("tool_calls", "n_benches", "n_gate_requests", "cost_usd", "first_bench_step", "compactions",
                "score"):
        vals = [p[key] for p in per if isinstance(p.get(key), (int, float))]
        agg[f"mean_{key}"] = round(sum(vals) / len(vals), 3) if vals else None
    agg["statuses"] = dict(Counter(p["status"] for p in per))
    agg["verdicts"] = dict(Counter(v for p in per for v in p["verdicts"]))
    return {"aggregate": agg, "missions": per}


def score(row: dict) -> float:
    """Deterministic partial credit for one mission row (higher is better)."""
    s = 0.0
    done = (row.get("done") or {}).get("status")
    if done == "win":
        s += 2.0 + min(float(row.get("delta_pct") or 0), 20.0) / 10
    if row.get("gate_verdicts"):
        s += 1.0
    elif row.get("n_gate_requests"):
        s += 0.5
    q = row.get("quick_bench_best")
    q = q[0] if isinstance(q, (list, tuple)) else q
    if isinstance(q, (int, float)) and q > 0:
        s += min(q, 10.0) / 20
    if row.get("n_benches"):
        s += 0.25
    if row.get("errors"):
        s -= 0.25
    return round(s, 3)


# ---------------------------------------------------------------------------------------------- propose
MECHANIC_SYSTEM = (
    "You are the Mechanic of an agent harness. Agent teams run missions that try to make verified, lossless "
    "speedups of serving recipes. You improve the harness itself, not the speedups: the role prompts, the "
    "experiment protocol wording and a few knobs. You see trace statistics from the last wave (tool use, error "
    "classes, detector signals, how fast teams measured, gate outcomes).\n"
    "Rules: change procedure, structure, clarity and knobs only. Never add advice about what to change in a "
    "serving stack (no flags, settings, kernels, models or techniques): the agents must discover those. Keep "
    "each text under 1800 characters. Make few, targeted changes that address a pattern in the stats. "
    "Call propose_playbook once, or call no_change if the stats show no harness problem."
)

PROPOSE_TOOL = {"type": "function", "function": {
    "name": "propose_playbook", "description": "Propose a revised playbook.",
    "parameters": {"type": "object", "properties": {
        "lead": {"type": "string"}, "worker": {"type": "string"}, "verifier": {"type": "string"},
        "protocol": {"type": "string"}, "knobs": {"type": "object"},
        "rationale": {"type": "string", "description": "which stats motivate which change"}},
        "required": ["rationale"]}}}
NO_CHANGE_TOOL = {"type": "function", "function": {
    "name": "no_change", "description": "Keep the playbook as is.",
    "parameters": {"type": "object", "properties": {"rationale": {"type": "string"}}, "required": ["rationale"]}}}

_FLAG = re.compile(r"(^|\s)--[a-z][a-z0-9-]+")


def leak_check(pb: dict, recipes: list[dict] | None = None) -> list[str]:
    """Reject agent-visible text that names serving settings (NO-LEAK). Returns problems."""
    problems = []
    terms = set()
    for r in recipes or []:
        for a in (r.get("serve") or {}).get("args") or []:
            a = str(a)
            if a.startswith("--"):
                terms.add(a.lower())
    for k in EDITABLE:
        t = pb.get(k) or ""
        if len(t) > 2400:
            problems.append(f"{k}: too long ({len(t)} chars)")
        if _FLAG.search(t):
            problems.append(f"{k}: names a command-line flag")
        low = t.lower()
        for term in terms:
            if term in low:
                problems.append(f"{k}: mentions {term}")
    return problems


def _apply(base: dict, args: dict) -> dict:
    new = {k: v for k, v in base.items() if k not in ("hash",)}
    for k in EDITABLE:
        if isinstance(args.get(k), str) and args[k].strip():
            new[k] = args[k].strip()
    knobs = dict(base.get("knobs") or {})
    for k, v in (args.get("knobs") or {}).items():
        if k in KNOB_BOUNDS:
            lo, hi = KNOB_BOUNDS[k]
            try:
                v = type(lo)(v)
            except (TypeError, ValueError):
                continue
            knobs[k] = min(hi, max(lo, v))
    new["knobs"] = knobs
    return new


def propose(mission_ids: list[str], base_hash: str | None = None, root=None, client: LLMClient | None = None,
            profile: str = "crusoe", recipes: list[dict] | None = None) -> str | None:
    root = _root(root)
    base_hash = base_hash or current(root)
    base = pb_mod.load(base_hash, root / "playbooks")
    stats = trace_stats(mission_ids, root)
    if client is None:
        load_secrets()
        client = LLMClient(get_profile(profile), Meter(root / "n1" / "usage.jsonl"))
    model = client.profile.model_for("ultra")
    view = {k: base.get(k) for k in (*EDITABLE, "knobs")}
    msgs = [{"role": "system", "content": MECHANIC_SYSTEM},
            {"role": "user", "content": "CURRENT PLAYBOOK:\n" + json.dumps(view, indent=1) +
             "\n\nKNOB BOUNDS: " + json.dumps(KNOB_BOUNDS) +
             "\n\nWAVE TRACE STATS:\n" + json.dumps(stats["aggregate"], indent=1) +
             "\n\nPER MISSION:\n" + json.dumps([{k: v for k, v in m.items() if k != "tools"}
                                                for m in stats["missions"]])[:12000]}]
    resp = client.chat(model, msgs, [PROPOSE_TOOL, NO_CHANGE_TOOL], role="mechanic", max_out=3000)
    call = next((c for c in resp.tool_calls if c.name in ("propose_playbook", "no_change")), None)
    if call is None or call.name == "no_change":
        events.append(log_path(root), {"event": "no_change", "base": base_hash, "missions": mission_ids,
                                       "rationale": (call.args.get("rationale") if call else resp.content)[:1000]})
        return None
    new = _apply(base, call.args)
    new["name"] = f"n1-from-{base_hash}"
    new["parent"] = base_hash
    problems = leak_check(new, recipes)
    if problems:
        events.append(log_path(root), {"event": "rejected_proposal", "base": base_hash, "problems": problems,
                                       "rationale": call.args.get("rationale", "")[:1000]})
        return None
    h = pb_mod.save(new, root / "playbooks")
    if h == base_hash:
        return None
    events.append(log_path(root), {"event": "proposed", "base": base_hash, "new": h, "missions": mission_ids,
                                   "rationale": call.args.get("rationale", "")[:1500],
                                   "changed": [k for k in (*EDITABLE, "knobs") if new.get(k) != base.get(k)],
                                   "stats": stats["aggregate"]})
    return h


# ---------------------------------------------------------------------------------------------- compare
def compare(old_hash: str, new_hash: str, rows: list[dict], root=None) -> dict:
    """Paired per-recipe comparison on held-out recipes. rows: mission rows carrying playbook + recipe_id.
    Promote when >= MIN_PAIRS recipes are paired, the mean paired score delta >= PROMOTE_MIN_MEAN, at least
    PROMOTE_NO_WORSE_FRAC of recipes are no worse, and no recipe loses a gate win."""
    by = defaultdict(lambda: {"old": [], "new": []})
    for r in rows:
        side = "old" if r.get("playbook") == old_hash else "new" if r.get("playbook") == new_hash else None
        if side:
            by[r["recipe_id"]][side].append(r)
    per = {}
    for rid, d in by.items():
        if not d["old"] or not d["new"]:
            continue
        so = sum(score(r) for r in d["old"]) / len(d["old"])
        sn = sum(score(r) for r in d["new"]) / len(d["new"])
        lost_win = any((r.get("done") or {}).get("status") == "win" for r in d["old"]) and not any(
            (r.get("done") or {}).get("status") == "win" for r in d["new"])
        per[rid] = {"old": round(so, 3), "new": round(sn, 3), "delta": round(sn - so, 3), "lost_win": lost_win}
    n = len(per)
    mean = sum(p["delta"] for p in per.values()) / n if n else 0.0
    no_worse = sum(1 for p in per.values() if p["delta"] >= 0) / n if n else 0.0
    lost = [rid for rid, p in per.items() if p["lost_win"]]
    if n < MIN_PAIRS:
        verdict, reason = "reject", f"only {n} paired recipes (need {MIN_PAIRS})"
    elif lost:
        verdict, reason = "reject", f"lost a gate win on {', '.join(lost)}"
    elif mean < PROMOTE_MIN_MEAN:
        verdict, reason = "reject", f"mean paired delta {mean:+.3f} < {PROMOTE_MIN_MEAN}"
    elif no_worse < PROMOTE_NO_WORSE_FRAC:
        verdict, reason = "reject", f"no worse on only {no_worse:.0%} of recipes"
    else:
        verdict, reason = "promote", f"mean paired delta {mean:+.3f} over {n} recipes, no worse on {no_worse:.0%}"
    out = {"verdict": verdict, "reason": reason, "old": old_hash, "new": new_hash, "pairs": n,
           "mean_delta": round(mean, 3), "no_worse_frac": round(no_worse, 3), "per_recipe": per}
    events.append(log_path(root), {"event": verdict, **out})
    return out
