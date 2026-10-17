"""Quick-bench runners: the real one submits forge.tools.benchjob through the executor as a GPU job;
the mock one returns synthetic numbers for tests and dry runs."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import PurePosixPath

from forge.tools.bundle import canon
from forge.tools.registry import EnvBlocked, ToolError


class RemoteBenchRunner:
    def __init__(self, executor, forge_root: str, python: str, vram_gb: float = 90.0,
                 timeout_s: float = 2400, sandbox: bool = True):
        self.ex = executor
        self.sandbox = sandbox
        self.forge_root = forge_root
        self.python = python
        self.vram_gb = vram_gb
        self.timeout_s = timeout_s

    def run(self, mission, *, config_delta=None, env=None, bundle_dir=None, n_prompts=None,
            fresh_reference=False, label="bench") -> dict:
        ws = self.ex.workspace(mission.id)
        stamp = f"{label}-{int(time.time() * 1000)}"
        work = str(PurePosixPath(ws) / "bench" / stamp)
        rec = mission.recipe
        spec = {
            "python": mission.python,
            "recipe": {"model": rec["model"], "serve_args": rec["serve"]["args"], "env": rec["serve"].get("env") or {},
                       "served_model_name": rec["serve"].get("served_model_name")},
            "workload": rec["workload"], "prompts": mission.bench_prompts, "n_prompts": n_prompts,
            "ref_cache_dir": str(PurePosixPath(self.forge_root) / "state" / "refcache"),
            "cache_dir": str(PurePosixPath(self.forge_root) / "state" / "benchcache"),
            "work_dir": work, "fresh_reference": fresh_reference,
            "serve_cwd": getattr(mission, "serve_cwd", None), "sandbox": self.sandbox,
            "hf_home": getattr(mission, "hf_home", None),
            "candidate": None if (config_delta is None and bundle_dir is None and env is None)
            else {"config_delta": config_delta, "env": env, "bundle_dir": bundle_dir},
        }
        spec_path = str(PurePosixPath(work) / "spec.json")
        self.ex.write_text(spec_path, json.dumps(spec))
        r = self.ex.run([self.python, "-m", "forge.tools.benchjob", "--spec", spec_path], cwd=self.forge_root,
                        env={"PYTHONPATH": self.forge_root}, timeout_s=self.timeout_s, kind="bench",
                        vram_gb=self.vram_gb, gpus=1, mission_id=mission.id, label=label)
        if r.infra_error:
            raise EnvBlocked(f"bench could not run: {r.infra_error}")
        line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("BENCH_RESULT ")), None)
        if line is None:
            err = (r.stderr or r.stdout).strip().splitlines()
            raise EnvBlocked(f"bench job failed (rc={r.rc}): {(err[-1] if err else 'no output')[:300]}")
        out = json.loads(line[len("BENCH_RESULT "):])
        if not out.get("ok"):
            # server failing to start with the candidate's settings is the agent's to fix
            raise ToolError(f"bench failed: {out.get('error', '?')}")
        out["work_dir"] = work
        return out


class MockBenchRunner:
    """Deterministic synthetic bench: the reference is ~100 with ±1% noise; a candidate's effect is a
    hash of its content in [-6%, +8%]. For tests only."""

    def __init__(self, base: float = 100.0, noise: float = 0.01, seed: int = 0, fixed_effects: dict | None = None):
        self.base = base
        self.noise = noise
        self.rng = random.Random(seed)
        self.fixed = fixed_effects or {}
        self.calls = []
        self.ref_cached = False

    def effect(self, key: str) -> float:
        if key in self.fixed:
            return self.fixed[key]
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return -0.06 + 0.14 * h

    def run(self, mission, *, config_delta=None, env=None, bundle_dir=None, n_prompts=None,
            fresh_reference=False, label="bench") -> dict:
        self.calls.append({"config_delta": config_delta, "env": env, "bundle_dir": bundle_dir})
        metric = mission.recipe["workload"].get("metric", "decode_tok_s")
        ref = self.base * (1 + self.rng.gauss(0, self.noise))
        out = {"ok": True, "gpu": {"uuid": "GPU-mock", "name": "mock"},
               "reference": {"metric": metric, "value": round(ref, 3), "n": n_prompts or 16},
               "reference_cached": self.ref_cached and not fresh_reference}
        self.ref_cached = True
        if config_delta is not None or bundle_dir is not None or env is not None:
            d, e = config_delta, env
            if bundle_dir:
                try:
                    from forge.tools.bundle import load_manifest
                    m = load_manifest(bundle_dir)
                    d, e = {**(m.get("config_delta") or {}), **(d or {})}, {**(m.get("env") or {}), **(e or {})}
                    d, e = d or None, e or None
                except OSError:
                    pass
            key = canon({"d": d, "e": e, "b": None})
            v = self.base * (1 + self.effect(key)) * (1 + self.rng.gauss(0, self.noise))
            out["candidate"] = {"metric": metric, "value": round(v, 3), "n": n_prompts or 16}
            out["delta_pct"] = round(100 * (v - ref) / ref, 3)
        return out
