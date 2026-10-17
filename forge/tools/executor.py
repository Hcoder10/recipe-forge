"""Executor interface: run commands and jobs on a mission's target machine.

The fleet provides the real one (forge.fleet.executor: ssh/daemon, sandbox, scheduler grants). Tools only
use this interface. LocalExecutor runs on this machine (for the rig itself and for tests).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class ExecResult:
    rc: int
    stdout: str
    stderr: str
    elapsed_s: float
    infra_error: str | None = None   # set when the command could not run at all (not the command's fault)

    @property
    def ok(self) -> bool:
        return self.rc == 0 and self.infra_error is None


@runtime_checkable
class Executor(Protocol):
    name: str

    def run(self, cmd: str | list[str], *, cwd: str | None = None, env: dict | None = None,
            timeout_s: float = 600, kind: str = "work", vram_gb: float | None = None,
            gpus: int = 0, mission_id: str = "", label: str = "") -> ExecResult:
        """kind: work (CPU or work-GPU job) | bench (needs a GPU grant through the scheduler).
        A GPU job blocks while its grant is queued (never refused just because a lease is pending)."""
        ...

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, text: str) -> None: ...

    def exists(self, path: str) -> bool: ...

    def list_files(self, root: str) -> list[str]:
        """Files under root, recursively, as paths relative to root ([] if root is missing)."""
        ...

    def listdir(self, path: str) -> list[str]: ...

    def remove(self, path: str) -> None: ...

    def workspace(self, mission_id: str) -> str:
        """Absolute path of the mission's writable workspace on the target."""
        ...


class LocalExecutor:
    """Runs on the local machine. GPU jobs go through `gpu_runner` when given (core's scheduler);
    otherwise kind='bench' is refused so nothing ever touches a GPU without a lease."""

    def __init__(self, root: str | Path, name: str = "local", gpu_runner=None, base_env: dict | None = None):
        self.name = name
        self.root = Path(root)
        self.gpu_runner = gpu_runner
        self.base_env = base_env or {}

    def workspace(self, mission_id: str) -> str:
        p = self.root / "missions" / mission_id / "ws"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def run(self, cmd, *, cwd=None, env=None, timeout_s=600, kind="work", vram_gb=None, gpus=0,
            mission_id="", label=""):
        full_env = {**os.environ, **self.base_env, **(env or {})}
        if kind == "bench" or gpus:
            if self.gpu_runner is None:
                return ExecResult(126, "", "", 0.0, infra_error="no GPU scheduler on this executor")
            return self.gpu_runner(cmd, cwd=cwd, env=full_env, timeout_s=timeout_s, vram_gb=vram_gb,
                                   gpus=gpus or 1, mission_id=mission_id, label=label)
        t0 = time.time()
        try:
            p = subprocess.run(cmd if isinstance(cmd, list) else ["bash", "-lc", cmd], cwd=cwd, env=full_env,
                               capture_output=True, text=True, timeout=timeout_s)
            return ExecResult(p.returncode, p.stdout, p.stderr, time.time() - t0)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecResult(124, out, f"timed out after {timeout_s:.0f}s", time.time() - t0)
        except FileNotFoundError as e:
            return ExecResult(127, "", str(e), time.time() - t0)
        except OSError as e:
            return ExecResult(126, "", "", time.time() - t0, infra_error=f"{type(e).__name__}: {e}")

    def read_text(self, path):
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def write_text(self, path, text):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")

    def exists(self, path):
        return Path(path).exists()

    def list_files(self, root):
        r = Path(root)
        if not r.is_dir():
            return []
        return sorted(p.relative_to(r).as_posix() for p in r.rglob("*") if p.is_file())

    def listdir(self, path):
        p = Path(path)
        return sorted(x.name for x in p.iterdir()) if p.is_dir() else []

    def remove(self, path):
        Path(path).unlink(missing_ok=True)


def quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
