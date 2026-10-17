"""Model ids, prices, context caps and profiles for the mission runner.

Everything tunable lives here so a profile switch never touches code. Model ids are verified against
/v1/models with `python -m forge.agents probe` once the key exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CRUSOE_BASE_URL = "https://api.inference.crusoecloud.com/v1"
SECRETS_FILE = Path.home() / "forge_secrets.env"

# $ per million tokens: (input, cached input, output)
PRICES: dict[str, tuple[float, float, float]] = {
    "ultra": (1.00, 0.25, 3.20),
    "super": (0.30, 0.15, 2.40),
    "nano": (0.05, 0.03, 0.20),
}


@dataclass
class ModelSpec:
    tier: str            # ultra | super | nano (price table key)
    model_id: str
    context: int         # hard context limit of the served model
    temperature: float = 0.2
    max_out: int = 4096


@dataclass
class Profile:
    name: str
    base_url: str
    api_key_env: str
    models: dict[str, ModelSpec]   # role tier -> model
    mock: bool = False

    def model_for(self, tier: str) -> ModelSpec:
        return self.models[tier]


def _crusoe_models() -> dict[str, ModelSpec]:
    return {
        "ultra": ModelSpec("ultra", os.environ.get("FORGE_ULTRA_MODEL",
                                                   "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"), 131072),
        "super": ModelSpec("super", os.environ.get("FORGE_SUPER_MODEL",
                                                   "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"), 131072),
        "nano": ModelSpec("nano", os.environ.get("FORGE_NANO_MODEL",
                                                 "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"), 131072),
    }


def _local_models() -> dict[str, ModelSpec]:
    mid = os.environ.get("FORGE_LOCAL_MODEL", "local")
    ctx = int(os.environ.get("FORGE_LOCAL_CONTEXT", "65536"))
    return {t: ModelSpec(t, mid, ctx) for t in ("ultra", "super", "nano")}


def profile(name: str) -> Profile:
    if name == "crusoe":
        return Profile("crusoe", os.environ.get("FORGE_CRUSOE_URL", CRUSOE_BASE_URL), "CRUSOE_API_KEY",
                       _crusoe_models())
    if name == "local":
        return Profile("local", os.environ.get("FORGE_LOCAL_URL", "http://127.0.0.1:8000/v1"),
                       "FORGE_LOCAL_KEY", _local_models())
    if name == "mock":
        return Profile("mock", "mock://", "", {t: ModelSpec(t, f"mock-{t}", 131072)
                                               for t in ("ultra", "super", "nano")}, mock=True)
    raise ValueError(f"unknown profile {name!r} (crusoe | local | mock)")


def load_secrets(path: Path = SECRETS_FILE) -> None:
    """KEY=VALUE lines into os.environ (does not override what is already set)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


@dataclass
class RoleConfig:
    tier: str
    context_cap: int      # enforced on the full request (prefix + history + tools)
    max_steps: int = 400


@dataclass
class TeamConfig:
    workers: int = 3
    lead: RoleConfig = field(default_factory=lambda: RoleConfig("ultra", 48_000))
    worker: RoleConfig = field(default_factory=lambda: RoleConfig("super", 64_000))
    verifier: RoleConfig = field(default_factory=lambda: RoleConfig("ultra", 48_000))
    worker_step_gap_s: float = 0.0
    lead_every_s: float = 60.0       # Lead turn cadence (it also wakes on triggers)

    @classmethod
    def from_dict(cls, d: dict | None) -> "TeamConfig":
        t = cls()
        for k, v in (d or {}).items():
            if k in ("lead", "worker", "verifier") and isinstance(v, dict):
                setattr(t, k, RoleConfig(**{**getattr(t, k).__dict__, **v}))
            elif hasattr(t, k):
                setattr(t, k, v)
        return t


def state_root() -> Path:
    """Where missions, the shared brain and playbooks live (internal disk, small)."""
    root = os.environ.get("FORGE_STATE")
    if root:
        return Path(root)
    return Path(os.environ.get("FORGE_ROOT", Path.home() / "forge")) / "state"
