"""sha256 lock over the gate's code.

    python -m forge.gate.lock write [--name main]     # freeze the current gate code
    python -m forge.gate.lock verify [--name main]    # exit 1 if anything changed
    python -m forge.gate.lock show [--name main]

Locks live in FORGE_ROOT/gate/locks/<name>.json (outside the deployed code). Rehearsals use their own
name (e.g. --name rehearsal) so freezing one never invalidates the other. The gate refuses to judge
when the lock it runs under does not verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

from forge.sched.machine import forge_root

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # .../forge
LOCKED = ["gate", "sandbox", "sched", "tools/bundle.py"]


def files() -> list[str]:
    out = []
    for item in LOCKED:
        p = os.path.join(PKG, item)
        if os.path.isfile(p):
            out.append(p)
            continue
        for dp, dns, fns in os.walk(p):
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            out += [os.path.join(dp, f) for f in sorted(fns) if f.endswith(".py")]
    return out


def digest() -> dict:
    per = {}
    for p in files():
        with open(p, "rb") as f:
            per[os.path.relpath(p, PKG).replace(os.sep, "/")] = hashlib.sha256(f.read()).hexdigest()
    total = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
    return {"sha256": total, "files": per}


def lock_path(name: str) -> str:
    return os.path.join(forge_root(), "gate", "locks", f"{name}.json")


def write(name: str = "main") -> dict:
    d = {**digest(), "name": name, "t": time.time()}
    p = lock_path(name)
    os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump(d, f, indent=1)
    os.replace(p + ".tmp", p)
    return d


def verify(name: str = "main") -> tuple[bool, str, str | None]:
    """(ok, message, locked sha256)."""
    p = lock_path(name)
    if not os.path.exists(p):
        return False, f"no gate lock '{name}' (run: python -m forge.gate.lock write --name {name})", None
    with open(p) as f:
        locked = json.load(f)
    now = digest()
    if now["sha256"] == locked["sha256"]:
        return True, "ok", locked["sha256"]
    changed = sorted(k for k in set(now["files"]) | set(locked["files"])
                     if now["files"].get(k) != locked["files"].get(k))
    return False, f"gate code changed since lock '{name}': {', '.join(changed[:8])}", locked["sha256"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forge.gate.lock")
    ap.add_argument("cmd", choices=["write", "verify", "show"])
    ap.add_argument("--name", default=os.environ.get("FORGE_GATE_LOCK", "main"))
    a = ap.parse_args(argv)
    if a.cmd == "write":
        d = write(a.name)
        print(f"locked '{a.name}': {d['sha256']} ({len(d['files'])} files)")
        return 0
    if a.cmd == "show":
        print(json.dumps(digest(), indent=1))
        return 0
    ok, msg, sha = verify(a.name)
    print(f"{'OK' if ok else 'FAIL'} {a.name} {sha or ''}: {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
