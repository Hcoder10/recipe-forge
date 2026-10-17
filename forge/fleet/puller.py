"""Pull results off rented boxes as they appear (boxes are destroyed with their disks on return).

Runs on the rig (it can ssh to every box with ~/.ssh/id_ed25519_forge):

    python -m forge.fleet.puller            # loop forever, every 60 s
    python -m forge.fleet.puller --once

For each active box it copies new or changed files from the box's `$FORGE_ROOT/results/` to the rig's
`~/forge/results/boxes/<machine>/`, and the box's gate ledger dir `$FORGE_ROOT/state/results/gate/` to the rig's
`~/forge/state/results/gate/<machine>/` (where the dashboard reads gate verdicts). Status goes to
`<FORGE_STATE>/fleet/pull_status.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from forge.fleet.executor import ExecutorError, executor_for
from forge.fleet.machines import all_machines, state_dir

RIG_ROOT = Path(os.environ.get("FORGE_ROOT", str(Path.home() / "forge")))


def pairs(m: dict) -> list[tuple[str, Path]]:
    r = m.get("forge_root", "/root/forge")
    return [(f"{r}/results", RIG_ROOT / "results" / "boxes" / m["name"]),
            (f"{r}/state/results/gate", RIG_ROOT / "state" / "results" / "gate" / m["name"])]


def ledger_pairs(m: dict) -> list[tuple[str, Path]]:
    """Single files pulled whole when they grow: the box gate ledger (agent-safe rows, per core.md)."""
    r = m.get("forge_root", "/root/forge")
    return [(f"{r}/gate/ledger.jsonl", RIG_ROOT / "results" / "boxes" / m["name"] / "gate" / "ledger.jsonl")]


def local_manifest(d: Path) -> dict[str, int]:
    out = {}
    if d.is_dir():
        for p in d.rglob("*"):
            if p.is_file():
                out[p.relative_to(d).as_posix()] = p.stat().st_size
    return out


def pull_machine(m: dict) -> dict:
    ex = executor_for(m)
    got = 0
    for remote, local in pairs(m):
        remote_files = ex.stat_tree(remote)
        have = local_manifest(local)
        want = [rel for rel, (size, _) in remote_files.items() if have.get(rel) != size]
        if want:
            got += ex.get_files(remote, want, local)
    for remote, local in ledger_pairs(m):
        r = ex.run(f"stat -c %s {remote} 2>/dev/null", timeout_s=60)
        size = int(r.stdout.strip()) if r.ok and r.stdout.strip().isdigit() else None
        if size is not None and (not local.exists() or local.stat().st_size != size):
            data = ex.read_bytes(remote)
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, local)
            got += 1
    return {"files": got, "t": time.time(), "ok": True}


def once() -> dict:
    status_p = state_dir() / "pull_status.json"
    try:
        status = json.loads(status_p.read_text())
    except (OSError, ValueError):
        status = {}
    for name, m in all_machines().items():
        if m.get("kind") != "vast" or not (m.get("ssh") or {}).get("host"):
            continue
        try:
            st = pull_machine(m)
            prev = status.get(name) or {}
            st["total"] = (prev.get("total") or 0) + st["files"]
            status[name] = st
        except (ExecutorError, OSError) as e:
            status[name] = {**(status.get(name) or {}), "ok": False, "error": str(e)[:300], "t_err": time.time()}
    tmp = status_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1))
    os.replace(tmp, status_p)
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=60)
    a = ap.parse_args(argv)
    while True:
        st = once()
        print(time.strftime("%H:%M:%S"), {k: (v.get("files"), v.get("ok")) for k, v in st.items()}, flush=True)
        if a.once:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
