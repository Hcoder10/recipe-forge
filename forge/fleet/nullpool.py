"""Pool compatible machines' reference null pairs (operator-only; never log values).

Pooling is a disclosed methodology choice: each pair remains a within-machine
AB/BA difference; pooling only enlarges the null sample. Local lossless bands
remain local. Pair provenance lives in the extra ``nullpool_pairs`` field.

The current gate holds service.lock for its entire lifetime, and History has no
independent lock. Writes and queue replacements therefore acquire service.lock
nonblocking, then ledger.lock, and defer while a service is alive. Never stop a
service to obtain this lock. Legacy pair identities are reconstructed from exact
reference verdicts, never inferred from rounded ledger values or requested reps.

Run on the operator laptop: python -m forge.fleet.nullpool --loop 120
Use --shrink once to replace queued target reference controls with reps=4.
Only counts, identities, hardware and fixed error codes reach the operator log.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

TARGETS = {"super3-120b-a12b-nvfp4", "nano3-30b-a3b-nvfp4",
           "lightning-30b-a3b-nvfp4-dspark-h100", "bolt-74b-lilicorr-5090"}


def compatible(a, b):
    if not a.get("gpu_class") or a.get("gpu_class") != b.get("gpu_class"):
        return False
    if a.get("name") and a.get("name") == b.get("name"):
        return True
    x, y = a.get("power"), b.get("power")
    return bool(x and y and abs(x-y) / min(x, y) <= .10)


def pair_id(p):
    return p["gate_id"], p["pair_index"]


def extract_pairs(history, machine, hardware, load_verdict):
    """Validate provenance against the numeric array; fail closed on ambiguity."""
    gains = history.get("null_gains", [])
    pairs = [dict(p) for p in history.get("nullpool_pairs", [])]
    if len(pairs) > len(gains):
        raise ValueError("provenance length")
    covered = {p["gate_id"] for p in pairs}
    for gid in history.get("controls", []):
        if gid in covered:
            continue
        v = load_verdict(gid)
        if v.get("kind") != "reference" or v.get("gate_id") != gid:
            raise ValueError("not a reference")
        for i, p in enumerate(v["pairs"]):
            pairs.append({"gate_id": gid, "pair_index": i, "value": p["log_gain"],
                          "pooled_from": machine, "hardware": hardware})
        covered.add(gid)
    if len(pairs) != len(gains) or any(p["value"] != g for p, g in zip(pairs, gains)):
        raise ValueError("unmapped history")
    if len({pair_id(p) for p in pairs}) != len(pairs):
        raise ValueError("duplicate identity")
    if any(not math.isfinite(p["value"]) for p in pairs):
        raise ValueError("nonfinite pair")
    return pairs


def merge_pairs(local, incoming, hardware):
    out = list(local)
    seen = {pair_id(p): p for p in out}
    for p in incoming:
        if not compatible(hardware, p.get("hardware", {})):
            continue
        key = pair_id(p)
        if key in seen:
            if seen[key] != p:
                raise ValueError("conflicting identity")
        else:
            out.append(p)
            seen[key] = p
    return out


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".nullpool-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def gate_locks(root):
    import fcntl
    # Same outer lock as service.main, then records._locked's ledger lock.
    with (root / "gate/service.lock").open("a") as service:
        fcntl.flock(service, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "gate/ledger.lock").open("a") as ledger:
            fcntl.flock(ledger, fcntl.LOCK_EX)
            yield


def hardware_for(machine):
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,power.limit,uuid",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if r.returncode:
        raise ValueError("hardware unavailable")
    rows = list(csv.reader(io.StringIO(r.stdout), skipinitialspace=True))
    timing = (machine.get("gpus") or {}).get("timing", [0])
    devices = []
    for row in rows:
        if int(row[0]) in timing:
            try:
                power = float(row[2])
            except ValueError:
                power = None
            devices.append({"name": row[1].strip(), "power": power,
                            "gpu_class": machine.get("gpu_class"), "uuid": row[3].strip()})
    if len(devices) != len(timing) or not devices:
        raise ValueError("missing timing GPU")
    if any(not compatible(a, b) for a in devices for b in devices):
        raise ValueError("heterogeneous timing GPUs")
    return devices[0]


def read_json(path):
    return json.loads(path.read_text())


def current_rows(root):
    rows = {}
    p = root / "gate/ledger.jsonl"
    for line in p.read_text().splitlines() if p.exists() else []:
        try:
            r = json.loads(line)
            rows.setdefault(r["gate_id"], {}).update(r)
        except ValueError:
            continue
    return rows


def shrink_locked(root, rows):
    changes = []
    for gid, row in rows.items():
        if row.get("kind") != "reference" or row.get("status") != "queued":
            continue
        req = read_json(root / "gate/requests" / (gid + ".json"))
        if req.get("recipe_key", "").split(":")[0] not in TARGETS or req.get("reps", 4) <= 4:
            continue
        new_id = f"g{time.strftime('%m%d%H%M')}-{uuid.uuid4().hex[:6]}"
        new = {**req, "gate_id": new_id, "reps": 4, "submitted": time.time(),
               "supersedes": gid}
        atomic_json(root / "gate/requests" / (new_id + ".json"), new)
        # Latest-row semantics in records.py; one append under ledger.lock.
        replacement = {k: v for k, v in row.items() if k in
                       ("kind", "recipe_id", "recipe_key", "machine", "mission_id", "requester")}
        with (root / "gate/ledger.jsonl").open("a") as f:
            f.write(json.dumps({"gate_id": gid, "status": "superseded", "superseded_by": new_id,
                                "t": time.time()}) + "\n")
            f.write(json.dumps({**replacement, "gate_id": new_id, "status": "queued",
                                "evidence_id": new_id, "confirmed": False, "verdict": None,
                                "t": time.time()}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        changes.append({"old": gid, "new": new_id, "recipe": req["recipe_key"], "reps": 4})
    return changes


def remote(payload):
    machine = payload["machine"]
    root = Path(os.path.expanduser(machine["forge_root"]))
    if not (root / "gate").is_dir():
        return {"status": "no_gate"}
    hardware = hardware_for(machine)
    def pairs(h):
        return extract_pairs(h, machine["name"], hardware,
                             lambda gid: read_json(root / "gate/runs" / gid / "verdict.json"))
    if payload["op"] == "collect":
        histories, counts, skipped = {}, {}, []
        for p in sorted((root / "gate/history").glob("*.json")):
            try:
                h = read_json(p)
                counts[p.stem] = len(h.get("null_gains", []))
                histories[p.stem] = pairs(h)
            except (ValueError, KeyError, OSError, TypeError):
                skipped.append(p.stem)
        controls = []
        for gid, row in current_rows(root).items():
            if row.get("kind") == "reference" and row.get("status") in ("queued", "running"):
                req = read_json(root / "gate/requests" / (gid + ".json"))
                controls.append({"gate_id": gid, "recipe": req.get("recipe_key"),
                                 "status": row["status"], "reps": req.get("reps")})
        return {"status": "ok", "hardware": hardware, "histories": histories,
                "counts": counts, "skipped": skipped, "controls": controls}
    try:
        with gate_locks(root):
            rows = current_rows(root)
            if any(r.get("status") == "running" for r in rows.values()):
                return {"status": "running_deferred"}
            if payload["op"] == "shrink":
                return {"status": "ok", "changes": shrink_locked(root, rows)}
            reports = {}
            for key, incoming in payload["histories"].items():
                if Path(key).name != key or key in (".", ".."):
                    raise ValueError("unsafe key")
                p = root / "gate/history" / (key + ".json")
                h = read_json(p) if p.exists() else {"null_gains": [], "null_probe": [],
                                                     "null_decode": [], "controls": []}
                local = pairs(h)
                merged = merge_pairs(local, incoming, hardware)
                if merged != local:
                    h["null_gains"] = [v["value"] for v in merged]
                    h["nullpool_pairs"] = merged
                    # controls stays local: pooled IDs are represented in metadata.
                    atomic_json(p, h)
                reports[key] = {"before": len(local), "after": len(merged),
                                "sources": sorted({v["pooled_from"] for v in merged})}
            return {"status": "ok", "counts": reports}
    except BlockingIOError:
        return {"status": "service_lock_busy"}


def call(machine, op, **kw):
    from forge.fleet.executor import FleetExecutor
    ex = FleetExecutor(machine)
    payload = {"machine": {k: machine[k] for k in ("name", "forge_root", "gpu_class", "gpus") if k in machine},
               "op": op, **kw}
    # Source and payload go over stdin, not argv, logs, or persistent remote files.
    code = Path(__file__).read_text(encoding="utf-8")
    boot = "import json,sys; d=json.load(sys.stdin); ns={'__name__':'nullpool_remote'}; exec(d['code'],ns); " \
           "\ntry: print(json.dumps(ns['remote'](d['payload'])))\nexcept Exception: print(json.dumps({'status':'remote_error'}))"
    r = ex.run([ex.python, "-B", "-c", boot], stdin=json.dumps({"code": code, "payload": payload}).encode(),
               timeout_s=40)
    if r.rc:
        return {"status": "transport_error", "rc": r.rc}
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {"status": "protocol_error"}


def machines(state):
    from forge.fleet.machines import static_machines
    out = static_machines()
    for p in sorted((state / "fleet/machines").glob("*.json")):
        m = read_json(p)
        out[m["name"]] = m
    return {n: {**m, "name": n} for n, m in out.items()
            if not m.get("retired") and (m.get("ssh") or {}).get("host")}


def emit(**data):
    print(json.dumps({"t": time.time(), **data}), flush=True)


def cycle(state, shrink=False):
    from concurrent.futures import ThreadPoolExecutor
    ms = machines(state)
    with ThreadPoolExecutor(max_workers=8) as pool:
        snaps = dict(zip(ms, pool.map(lambda m: call(m, "collect"), ms.values())))
    for name, s in snaps.items():
        emit(machine=name, phase="collect", **{k: v for k, v in s.items() if k != "histories"})
    if shrink:
        for n, m in ms.items():
            if n != "rig" and snaps[n].get("status") == "ok":
                emit(machine=n, phase="shrink", **call(m, "shrink"))
    for n, m in ms.items():
        s = snaps[n]
        if s.get("status") != "ok":
            continue
        union = {}
        for other in snaps.values():
            if other.get("status") != "ok" or not compatible(s["hardware"], other["hardware"]):
                continue
            for key, ps in other["histories"].items():
                union[key] = merge_pairs(union.get(key, []), ps, s["hardware"])
        emit(machine=n, phase="merge", **call(m, "merge", histories=union))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", type=float, default=0)
    ap.add_argument("--shrink", action="store_true")
    ap.add_argument("--state-root", type=Path, default=Path(os.environ.get("FORGE_STATE", "state")))
    args = ap.parse_args(argv)
    if args.loop < 0:
        ap.error("loop must be nonnegative")
    emit(phase="start", pid=os.getpid(), interval=args.loop)
    while True:
        try:
            cycle(args.state_root, args.shrink)
        except Exception:
            emit(phase="cycle", status="error")
        if not args.loop:
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
