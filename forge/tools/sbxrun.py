"""Run an agent's sh/py command inside forge.sandbox on the target (no GPU, no network).

    python -m forge.tools.sbxrun --ws <workspace> --venv <venv dir> [--ro path ...] [--timeout s] -- <cmd ...>

Writable: the mission workspace (plus the job's private home/tmp). Read-only: the venv, extra --ro paths, system
dirs. Output passes through; exit code is the command's. Exit 199 + "FORGE_INFRA:" means the sandbox itself
could not start.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--ro", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE")
    ap.add_argument("--gpus-from-env", action="store_true",
                    help="grant the job the GPUs in CUDA_VISIBLE_DEVICES (set by the scheduler grant)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    try:
        from forge.sandbox import Spec, launch
    except ImportError as e:
        print(f"FORGE_INFRA: sandbox not installed ({e})", file=sys.stderr)
        return 199
    jobs = os.path.join(os.path.dirname(os.path.abspath(a.ws)), "sbx")
    job_dir = os.path.join(jobs, f"{int(time.time())}-{uuid.uuid4().hex[:6]}")
    env = dict(kv.split("=", 1) for kv in a.env if "=" in kv)
    gpus = []
    if a.gpus_from_env:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not vis:
            print("FORGE_INFRA: --gpus-from-env but no CUDA_VISIBLE_DEVICES (no grant)", file=sys.stderr)
            return 199
        gpus = [int(x) for x in vis.split(",") if x.strip()]
        for k in ("LANDLORD_LEASE_ID", "FORGE_LEASE_IDS", "FORGE_GRANT_ID"):
            if os.environ.get(k):
                env.setdefault(k, os.environ[k])
    try:
        job = launch(Spec(cmd=cmd, job_dir=job_dir, ro=[a.venv] + [p for p in a.ro if p and os.path.exists(p)],
                          rw=[a.ws], env=env, gpus=gpus, path_prepend=[os.path.join(a.venv, "bin")], cwd=a.ws,
                          name=f"agent-{uuid.uuid4().hex[:8]}"))
    except Exception as e:  # noqa: BLE001
        print(f"FORGE_INFRA: sandbox launch failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 199
    rc = job.wait(timeout=a.timeout)
    sys.stdout.write(job.tail(12000))
    job.cleanup(keep_log=False)
    return 124 if rc in (-15, 143) and time.time() - job.started >= a.timeout else rc


if __name__ == "__main__":
    sys.exit(main())
