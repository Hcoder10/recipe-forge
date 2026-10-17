"""Gate requests and the append-only gate ledger (FORGE_ROOT/gate/ledger.jsonl).

The latest row per gate_id is its status. Rows hold no prompt text and no server logs.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import sys
import time
import uuid
from contextlib import contextmanager

from forge.sched.machine import forge_root

MIN_NULL_PAIRS = 8
REF_REPS = 4
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gate_dir() -> str:
    d = os.path.join(forge_root(), "gate")
    os.makedirs(os.path.join(d, "requests"), mode=0o700, exist_ok=True)
    return d


@contextmanager
def _locked():
    with open(os.path.join(gate_dir(), "ledger.lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def rows() -> list[dict]:
    p = os.path.join(gate_dir(), "ledger.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def latest() -> dict[str, dict]:
    d: dict[str, dict] = {}
    for r in rows():
        d[r["gate_id"]] = {**d.get(r["gate_id"], {}), **r}
    return d


def _write_row(row: dict) -> dict:
    row = {**row, "t": time.time()}
    with open(os.path.join(gate_dir(), "ledger.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


PUBLIC = ("gate_id", "mission_id", "recipe_id", "recipe_key", "machine", "bundle_hash", "kind", "status",
          "verdict", "delta_pct", "evidence_id", "confirmed", "detail", "parent", "requester", "claim",
          "gain_pct", "threshold_pct", "lossless_pass", "pairs", "confirm_gate")


def append(req: dict, **fields) -> dict:
    base = {k: req.get(k) for k in PUBLIC if k in req}
    with _locked():
        return _write_row({**base, **fields})


# -- recipes -------------------------------------------------------------------
def load_recipe(recipe_id: str) -> dict:
    import yaml
    path = os.path.join(PKG, "recipes", recipe_id + ".yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(recipe_id: str, recipe: dict | None = None) -> tuple[dict, str, dict]:
    """'<id>' or '<id>:<variant>' -> (recipe, recipe_key, workload dict)."""
    rid, _, variant = recipe_id.partition(":")
    if not variant and "." in rid and not os.path.exists(os.path.join(PKG, "recipes", rid + ".yaml")):
        rid, _, variant = rid.rpartition(".")      # fleet's older '<base>.<variant>' form
    recipe = recipe or load_recipe(rid)
    wl = dict(recipe.get("workload") or {})
    variants = recipe.get("variants") or []
    if variants:
        pick = next((v for v in variants if v.get("name") == variant), None) if variant else variants[0]
        if pick is None:
            raise ValueError(f"recipe {rid} has no variant {variant!r}; variants: {[v.get('name') for v in variants]}")
        wl.update({k: v for k, v in pick.items() if k != "name"})
        variant = pick.get("name")
    conc = int(wl.get("concurrency") or 1)
    workload = {"concurrency": conc, "max_tokens": int(wl.get("max_tokens") or 512),
                "metric": wl.get("metric") or ("decode_tok_s" if conc == 1 else "output_tok_s"),
                "prompts_per_rep": int(wl.get("gate_prompts_per_rep") or (8 if conc == 1 else 2 * conc)),
                "ignore_eos": bool(wl.get("gate_ignore_eos", True))}
    key = f"{recipe.get('id', rid)}:{variant}" if variant else recipe.get("id", rid)
    return recipe, key, workload


# -- requests ------------------------------------------------------------------
def _save_request(req: dict) -> None:
    p = os.path.join(gate_dir(), "requests", req["gate_id"] + ".json")
    with open(p + ".tmp", "w") as f:
        json.dump(req, f)
    os.replace(p + ".tmp", p)


def get_request(gate_id: str) -> dict:
    with open(os.path.join(gate_dir(), "requests", gate_id + ".json")) as f:
        return json.load(f)


def submit(kind: str, recipe_id: str, bundle_dir: str | None = None, bundle_hash: str | None = None,
           mission_id: str | None = None, machine: str | None = None, requester: str | None = None,
           claim: str | None = None, recipe: dict | None = None, python: str | None = None,
           reps: int | None = None, parent: str | None = None, confirm_index: int = 0,
           lock: str | None = None, gpu_uuid: str | None = None) -> dict:
    recipe, key, workload = resolve(recipe_id, recipe)
    if kind != "reference":
        if not bundle_dir or not os.path.isdir(bundle_dir):
            raise ValueError(f"bundle_dir does not exist: {bundle_dir}")
        bundle_dir = os.path.realpath(bundle_dir)
        if not bundle_hash:
            from pathlib import Path
            from forge.tools.bundle import hash_dir
            bundle_hash = hash_dir(Path(bundle_dir))
    python = python or recipe.get("python") or os.environ.get("FORGE_GATE_PYTHON") or sys.executable
    history_key = key + ("@" + gpu_uuid[:16] if gpu_uuid else "")
    with _locked():
        cur = latest()
        if kind in ("candidate", "confirm"):
            for r in cur.values():
                if (r.get("recipe_key") == key and r.get("bundle_hash") == bundle_hash and r.get("kind") == kind
                        and r.get("status") in ("queued", "running", "done")
                        and r.get("verdict") not in ("invalid", "error")):
                    return {**r, "dedup": True}
        gid = f"g{time.strftime('%m%d%H%M')}-{uuid.uuid4().hex[:6]}"
        req = {"gate_id": gid, "kind": kind, "recipe_id": recipe_id, "recipe_key": key, "recipe": recipe,
               "workload": workload, "bundle_dir": bundle_dir, "bundle_hash": bundle_hash,
               "mission_id": mission_id, "machine": machine, "requester": requester, "claim": (claim or "")[:2000],
               "python": python, "reps": reps or REF_REPS, "parent": parent, "confirm_index": confirm_index,
               "lock": lock or os.environ.get("FORGE_GATE_LOCK", "main"), "history_key": history_key,
               "submitted": time.time()}
        _save_request(req)
        row = _write_row({**{k: req.get(k) for k in PUBLIC if k in req}, "status": "queued", "verdict": None,
                          "evidence_id": gid, "confirmed": False})
    return row


def _n_null(history_key: str) -> int:
    p = os.path.join(gate_dir(), "history", history_key + ".json")
    try:
        with open(p) as f:
            return len(json.load(f)["null_gains"])
    except (OSError, ValueError, KeyError):
        return 0


def next_request() -> dict | None:
    """Pick the next queued request, queueing reference controls a recipe still needs."""
    cur = latest()
    queued = [r for r in cur.values() if r.get("status") == "queued"]
    if not queued:
        return None
    reqs = {r["gate_id"]: get_request(r["gate_id"]) for r in queued}
    pending_refs: dict[str, int] = {}
    for r in cur.values():
        if r.get("kind") == "reference" and r.get("status") in ("queued", "running"):
            hk = get_request(r["gate_id"])["history_key"]
            pending_refs[hk] = pending_refs.get(hk, 0) + 1
    for req in list(reqs.values()):
        if req["kind"] == "reference":
            continue
        hk = req["history_key"]
        need = math.ceil(max(0, MIN_NULL_PAIRS - _n_null(hk)) / REF_REPS) - pending_refs.get(hk, 0)
        for _ in range(max(0, need)):
            row = submit("reference", req["recipe_id"], recipe=req["recipe"], python=req["python"],
                         machine=req.get("machine"), requester="gate/setup",
                         claim=f"reference control needed before judging {req['gate_id']}",
                         lock=req.get("lock"), gpu_uuid=None)
            ref = get_request(row["gate_id"])
            ref["history_key"] = hk
            _save_request(ref)
            reqs[ref["gate_id"]] = ref
            pending_refs[hk] = pending_refs.get(hk, 0) + 1
    done_per_mission: dict[str, int] = {}
    for r in cur.values():
        if r.get("status") == "done" and r.get("mission_id"):
            done_per_mission[r["mission_id"]] = done_per_mission.get(r["mission_id"], 0) + 1
    rank = {"reference": 0, "confirm": 1, "candidate": 2}
    order = sorted(reqs.values(), key=lambda q: (rank.get(q["kind"], 3),
                                                  done_per_mission.get(q.get("mission_id") or "", 0),
                                                  q["submitted"]))
    return order[0]


def finish(req: dict, v: dict) -> dict:
    """Write the final row; a candidate win queues its confirmation, a confirmation updates its parent."""
    test = v.get("test") or {}
    row = append(req, status="done" if v.get("verdict") not in ("error",) else "error", verdict=v.get("verdict"),
                 delta_pct=v.get("delta_pct"), detail=v.get("detail"), gain_pct=test.get("gain_pct"),
                 threshold_pct=test.get("threshold_pct"), lossless_pass=(v.get("lossless") or {}).get("pass"),
                 pairs=[round(p["log_gain"], 5) for p in v.get("pairs", [])], evidence_id=req["gate_id"],
                 minutes=v.get("minutes"))
    if req["kind"] == "candidate" and v.get("verdict") == "win":
        n_conf = sum(1 for r in latest().values() if r.get("kind") == "confirm")
        c = submit("confirm", req["recipe_id"], bundle_dir=req["bundle_dir"], bundle_hash=req["bundle_hash"],
                   mission_id=req.get("mission_id"), machine=req.get("machine"), requester="gate/confirm",
                   claim=f"confirmation of {req['gate_id']}", recipe=req["recipe"], python=req["python"],
                   parent=req["gate_id"], confirm_index=n_conf, lock=req.get("lock"))
        if not c.get("dedup"):
            r2 = get_request(c["gate_id"])
            r2["history_key"] = req["history_key"]
            _save_request(r2)
        append(req, confirm_gate=c["gate_id"])
    if req["kind"] == "confirm" and req.get("parent"):
        parent = get_request(req["parent"])
        append(parent, confirmed=v.get("verdict") == "win",
               detail=(latest().get(req["parent"], {}).get("detail") or "")
               + f" | confirmation {req['gate_id']}: {v.get('verdict')} ({v.get('detail')})")
    return row
