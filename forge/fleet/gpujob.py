"""Run one command on a GPU grant from this machine's scheduler (forge.sched), then release it.

    python -m forge.fleet.gpujob --kind bench --vram 90 --mission m1 --label bench -- <cmd ...>

Runs ON the target machine (the executor wraps GPU commands in it). Waits while the grant is queued, runs the
command with the grant's env (CUDA_VISIBLE_DEVICES, LANDLORD_LEASE_ID, FORGE_PORT_LO/HI, ...), passes its
output through and exits with its return code. If the command cannot run at all (scheduler down, request
impossible, wait too long) it prints `FORGE_INFRA: <reason>` on stderr and exits 199.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys

INFRA_RC = 199


def infra(msg: str) -> int:
    print(f"FORGE_INFRA: {msg}", file=sys.stderr, flush=True)
    return INFRA_RC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forge.fleet.gpujob")
    ap.add_argument("--kind", default="bench", choices=["bench", "serve", "gate", "work"])
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--vram", type=float, default=None)
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--mission", default="-")
    ap.add_argument("--label", default="job")
    ap.add_argument("--timeout", type=int, default=0, help="seconds the command may run once granted")
    ap.add_argument("--wait", type=int, default=3600, help="seconds to wait for the grant")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        return infra("no command given")
    try:
        from forge.sched.client import SchedClient, SchedError
    except ImportError as e:
        return infra(f"scheduler client not installed on this machine ({e})")
    sc = SchedClient()
    if not sc.healthy():
        return infra("the scheduler daemon on this machine is not running (start: python -m forge.sched.daemon ensure)")
    owner = f"mission:{a.mission}" if a.mission != "-" else "fleet"
    purpose = f"{a.label} for {owner}: {' '.join(cmd)[:160]}"
    kw = {"minutes": a.minutes}
    if a.vram is not None:
        kw["vram_gb"] = a.vram
    waited = []

    def on_wait(g):
        if g.wait_reason and (not waited or waited[-1] != g.wait_reason):
            waited.append(g.wait_reason)
            print(f"[gpujob] waiting for a {a.kind} grant: {g.wait_reason}", file=sys.stderr, flush=True)

    try:
        grant = sc.acquire(a.kind, owner, purpose, timeout=a.wait, on_wait=on_wait, **kw)
    except SchedError as e:
        return infra(str(e))
    except Exception as e:  # scheduler answered with an error, or the request was refused
        return infra(f"grant request failed: {type(e).__name__}: {e}")
    env = {**os.environ, **{k: str(v) for k, v in (grant.env or {}).items()}}
    rc = 1
    proc = None

    def _forward(signum, _frame):
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signum)
            except OSError:
                pass

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _forward)
    try:
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=a.timeout or None)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            print(f"[gpujob] timed out after {a.timeout}s", file=sys.stderr, flush=True)
            rc = 124
    except OSError as e:
        rc = infra(f"could not start the command: {e}")
    finally:
        try:
            grant.release(f"{a.label}: rc={rc}")
        except Exception as e:
            print(f"[gpujob] release failed (the scheduler expires it): {e}", file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
