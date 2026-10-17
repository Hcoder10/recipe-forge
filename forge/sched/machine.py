"""Machine description used by the scheduler, sandbox and gate.

Loaded from $FORGE_ROOT/machine.json when present (the fleet writes it on deploy), otherwise from
nvidia-smi with defaults: on the rig GPUs 0,1 are timing and 2 is work; elsewhere every GPU is timing.
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess


def forge_root() -> str:
    return os.path.realpath(os.path.expanduser(os.environ.get("FORGE_ROOT", "~/forge")))


@dataclasses.dataclass
class Machine:
    name: str
    building: str                       # landlord building name
    timing_gpus: list[int]
    work_gpus: list[int]
    vram_gb: dict[int, float]
    gpu_uuids: dict[int, str]
    gate_cpus: list[int]                # gate client + gate servers
    agent_cpus: list[int]               # everything else
    port_base: int = 20000              # grant port blocks start here
    port_block: int = 100
    port_blocks: int = 40
    vram_reserve_gb: float = 1.5        # never hand out the last bit of a work GPU

    def all_gpus(self) -> list[int]:
        return sorted(set(self.timing_gpus) | set(self.work_gpus))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["vram_gb"] = {str(k): v for k, v in self.vram_gb.items()}
        d["gpu_uuids"] = {str(k): v for k, v in self.gpu_uuids.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Machine":
        d = dict(d)
        d["vram_gb"] = {int(k): float(v) for k, v in d.get("vram_gb", {}).items()}
        d["gpu_uuids"] = {int(k): v for k, v in d.get("gpu_uuids", {}).items()}
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


def _smi() -> list[tuple[int, str, float, bool]]:
    """(index, uuid, GiB, power_capped). A card capped below its default power limit never times."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,power.limit,power.default_limit",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        i, u, m, lim, dflt = [x.strip() for x in line.split(",")]
        try:
            capped = float(lim) < 0.97 * float(dflt)
        except ValueError:
            capped = False
        rows.append((int(i), u, float(m) / 1024.0, capped))
    return rows


def tailnet_ips() -> list[str]:
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [w.split("/")[0] for w in out.split() if w.startswith("100.")]


def _split_cpus(n: int) -> tuple[list[int], list[int]]:
    """Gate gets the first quarter of physical cores plus their SMT siblings."""
    try:
        with open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") as f:
            sib = f.read()
        smt = "," in sib or "-" in sib
    except OSError:
        smt = False
    phys = n // 2 if smt else n
    g = max(1, phys // 4)
    gate = list(range(g)) + ([c + phys for c in range(g)] if smt else [])
    agent = [c for c in range(n) if c not in gate]
    return gate, agent


def load(path: str | None = None) -> Machine:
    path = path or os.path.join(forge_root(), "machine.json")
    if os.path.exists(path):
        with open(path) as f:
            return Machine.from_dict(json.load(f))
    rows = _smi()
    host = socket.gethostname()
    timing = [r[0] for r in rows if not r[3]]
    work = [r[0] for r in rows if r[3]]
    gate, agent = _split_cpus(os.cpu_count() or 1)
    return Machine(name=host, building=os.environ.get("LANDLORD_BUILDING", host), timing_gpus=timing,
                   work_gpus=work, vram_gb={r[0]: round(r[2], 2) for r in rows},
                   gpu_uuids={r[0]: r[1] for r in rows}, gate_cpus=gate, agent_cpus=agent)
