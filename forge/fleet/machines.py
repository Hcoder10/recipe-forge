"""Machine registry: the static rig (forge/fleet/machines.yaml) plus rented boxes in
<FORGE_STATE>/fleet/machines/<name>.json.

A machine dict: name, hostname, building, ssh {host, user, port, key?}, forge_root, gpu_class
(rtxpro6000_95g | rtx5090_32g), gpus {timing, work}, python, sched_python, python_for {base_id: python},
recipes [base ids it can serve], env {..}, results_dir, and for rentals: rental_id, instance, usd_per_hour,
image, rented_at, kind: "vast".
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def state_dir() -> Path:
    root = os.environ.get("FORGE_STATE") or os.path.join(os.environ.get("FORGE_ROOT", str(Path.home() / "forge")), "state")
    p = Path(root) / "fleet"
    p.mkdir(parents=True, exist_ok=True)
    return p


def static_machines() -> dict[str, dict]:
    with open(HERE / "machines.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dynamic_dir() -> Path:
    d = state_dir() / "machines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def all_machines(include_retired: bool = False) -> dict[str, dict]:
    out = dict(static_machines())
    for p in sorted(glob.glob(str(dynamic_dir() / "*.json"))):
        try:
            m = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("retired") and not include_retired:
            continue
        out[m["name"]] = m
    return out


def get_machine(name: str) -> dict:
    ms = all_machines(include_retired=True)
    if name not in ms:
        raise KeyError(f"no machine {name!r}; known: {sorted(ms)}")
    return ms[name]


def save_machine(m: dict) -> Path:
    p = dynamic_dir() / f"{m['name']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    return p


def retire_machine(name: str, why: str) -> None:
    p = dynamic_dir() / f"{name}.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        m["retired"] = why
        save_machine(m)


def python_for(m: dict, recipe: dict) -> str:
    return (m.get("python_for") or {}).get(recipe.get("base_id", recipe["id"])) or m.get("python") or "python3"


def machine_json(m: dict, smi: list[tuple[int, str, float]] | None = None) -> dict:
    """The $FORGE_ROOT/machine.json core's scheduler reads (forge.sched.machine.Machine fields)."""
    g = m.get("gpus") or {}
    d = {"name": m["name"], "building": m["building"],
         "timing_gpus": list(g.get("timing", [0])), "work_gpus": list(g.get("work", []))}
    if smi:
        d["vram_gb"] = {str(i): round(mem, 2) for i, _, mem in smi}
        d["gpu_uuids"] = {str(i): u for i, u, _ in smi}
    return d
