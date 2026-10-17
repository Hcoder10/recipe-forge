"""Set up a rented box for its recipes. Runs ON the box (python3 of the vLLM image), as a detached job:

    python3 -m forge.fleet.boxsetup --plan /root/forge/state/box_plan.json

Plan: {name, building, gpu_class, recipes: [base ids], landlord_url}. Steps, each recorded in
$FORGE_ROOT/state/box_status.json as it finishes (the controller polls this file):
  python deps -> per-recipe extras (e.g. backend kernels the recipe page installs) -> model downloads (resumable,
  with a socket timeout) -> recipe files (reasoning-parser plugins) -> machine.json -> scheduler daemon.
Idempotent: rerunning skips finished steps.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback

ROOT = os.environ.get("FORGE_ROOT", "/root/forge")
STATE = os.path.join(ROOT, "state")
STATUS = os.path.join(STATE, "box_status.json")


def log(msg: str) -> None:
    print(f"[boxsetup {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_status() -> dict:
    try:
        with open(STATUS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"steps": {}, "started": time.time()}


def save_status(st: dict) -> None:
    os.makedirs(STATE, exist_ok=True)
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATUS)


def step(st: dict, name: str, fn) -> None:
    if (st["steps"].get(name) or {}).get("ok"):
        log(f"skip {name} (done)")
        return
    t0 = time.time()
    st["steps"][name] = {"ok": False, "started": t0}
    st["current"] = name
    save_status(st)
    log(f"start {name}")
    try:
        info = fn() or {}
        st["steps"][name] = {"ok": True, "s": round(time.time() - t0, 1), **info}
        log(f"done {name} in {time.time() - t0:.0f}s")
    except Exception as e:
        st["steps"][name] = {"ok": False, "s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}
        st["failed"] = name
        save_status(st)
        traceback.print_exc()
        raise
    finally:
        save_status(st)


def sh(cmd: str, timeout: int = 3600) -> str:
    log(f"$ {cmd}")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = (p.stdout + p.stderr)[-4000:]
    if p.returncode != 0:
        raise RuntimeError(f"rc={p.returncode}: {out[-800:]}")
    return out


def download(repo: str, tries: int = 4) -> dict:
    socket.setdefaulttimeout(60)
    from huggingface_hub import snapshot_download
    last = None
    for i in range(tries):
        try:
            p = snapshot_download(repo, max_workers=16)
            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)
            return {"path": p, "gb": round(size / 1e9, 2)}
        except Exception as e:  # network hiccups: retry, the download resumes
            last = e
            log(f"download {repo} attempt {i + 1} failed: {e}")
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"download {repo} failed: {last}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    a = ap.parse_args(argv)
    with open(a.plan) as f:
        plan = json.load(f)
    sys.path.insert(0, ROOT)
    from forge.recipes import get as get_recipe
    st = load_status()
    st["plan"] = plan
    st.pop("failed", None)
    save_status(st)
    recipes = [get_recipe(b, plan.get("gpu_class")) for b in plan["recipes"]]

    def deps():
        out = sh("python3 -c 'import yaml, huggingface_hub, vllm; print(vllm.__version__)'")
        return {"vllm": out.strip().splitlines()[-1]}

    step(st, "deps", deps)

    if any(r["base_id"].startswith("bolt") for r in recipes):
        # The Bolt recipe page installs these inside the container before `vllm serve` (needed by its backends).
        step(st, "bolt_extras", lambda: {"out": sh("pip install -q b12x && vllm download-kernels", timeout=3600)[-300:]})

    models = []
    for r in recipes:
        models.append(r["model"])
        if r.get("drafter"):
            models.append(r["drafter"]["model"])
    for m in dict.fromkeys(models):
        step(st, f"download:{m}", lambda m=m: download(m))

    def recipe_files():
        placed = []
        for r in recipes:
            d = os.path.join(STATE, "recipe_files", r["base_id"])
            os.makedirs(d, exist_ok=True)
            for ef in (r.get("vllm") or {}).get("extra_files") or []:
                dest = os.path.join(d, ef["dest"])
                if not os.path.exists(dest):
                    import urllib.request
                    urllib.request.urlretrieve(ef["url"], dest + ".tmp")
                    os.replace(dest + ".tmp", dest)
                placed.append(dest)
        return {"files": placed}

    step(st, "recipe_files", recipe_files)

    def machine_json():
        from forge.sched import machine as mm
        path = os.path.join(ROOT, "machine.json")
        if os.path.exists(path):
            os.remove(path)
        m = mm.load(path)              # nvidia-smi + cpu split defaults
        m.name, m.building = plan["name"], plan["building"]
        m.timing_gpus, m.work_gpus = list(plan.get("timing", m.all_gpus())), list(plan.get("work", []))
        with open(path, "w") as f:
            json.dump(m.to_dict(), f, indent=1)
        return {"machine": m.to_dict()}

    step(st, "machine_json", machine_json)

    def sched():
        from forge.sched.client import SchedClient
        if SchedClient().healthy():
            return {"already": True}
        env = {**os.environ, "FORGE_ROOT": ROOT, "PYTHONPATH": ROOT, "LANDLORD_URL": plan["landlord_url"],
               "LANDLORD_BUILDING": plan["building"]}
        os.makedirs(os.path.join(STATE, "logs"), exist_ok=True)
        lf = open(os.path.join(STATE, "logs", "sched.log"), "ab")
        subprocess.Popen([sys.executable, "-m", "forge.sched.daemon"], env=env, stdout=lf, stderr=lf,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=ROOT)
        for _ in range(60):
            time.sleep(1)
            if SchedClient().healthy():
                return {"started": True}
        raise RuntimeError("scheduler daemon did not answer within 60 s (see state/logs/sched.log)")

    step(st, "sched", sched)

    def gate_service():
        env = {**os.environ, "FORGE_ROOT": ROOT, "PYTHONPATH": ROOT, "LANDLORD_URL": plan["landlord_url"]}
        if not os.path.exists(os.path.join(ROOT, "gate", "locks", "main.json")):
            out = subprocess.run([sys.executable, "-m", "forge.gate.lock", "write", "--name", "main"], env=env,
                                 cwd=ROOT, capture_output=True, text=True, timeout=300)
            if out.returncode != 0:
                raise RuntimeError(f"gate lock: {(out.stdout + out.stderr)[-500:]}")
        ps = subprocess.run(["bash", "-c", "pgrep -f '[f]orge.gate.service' || true"], capture_output=True, text=True)
        if ps.stdout.strip():
            return {"already": ps.stdout.split()}
        lf = open(os.path.join(STATE, "logs", "gate.log"), "ab")
        p = subprocess.Popen([sys.executable, "-m", "forge.gate.service"], env=env, stdout=lf, stderr=lf,
                             stdin=subprocess.DEVNULL, start_new_session=True, cwd=ROOT)
        time.sleep(5)
        if p.poll() is not None:
            raise RuntimeError(f"gate service exited rc={p.returncode} (see state/logs/gate.log)")
        return {"pid": p.pid}

    if plan.get("gate", True):
        step(st, "gate_service", gate_service)
    st["ready"] = True
    st["ready_at"] = time.time()
    st["current"] = None
    save_status(st)
    log("box ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
