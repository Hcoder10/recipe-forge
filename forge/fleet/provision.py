"""Bring a rented box into the fleet: register it, wait for ssh, let the rig in, install forge, start setup.

    m = register_rental(rental, gpu_class, recipes, image)   # from landlord rent_machine / list_rentals
    provision(m)        # idempotent; returns the box status (setup runs detached ON the box)
    box_status(m)       # {ready, current, failed, steps}

The rig gets in with its own key (~/.ssh/id_ed25519_forge on the rig), appended to the box's authorized_keys
after normalising the file to one key per line.
"""
from __future__ import annotations

import json
import posixpath
import re
import time
from pathlib import Path

from forge.fleet.executor import ExecutorError, executor_for
from forge.fleet.machines import get_machine, save_machine

REPO = Path(__file__).resolve().parents[2]
BOX_ROOT = "/root/forge"
BOX_LANDLORD = "http://127.0.0.1:7788/mcp"   # reverse tunnel to the coordinator, on the box's loopback


def register_rental(rental: dict, gpu_class: str, recipes: list[str], image: str, name: str | None = None) -> dict:
    inst = str(rental.get("provider_id") or rental.get("instance"))
    m = {
        "name": name or f"vast-{inst}", "kind": "vast", "hostname": None, "building": rental.get("building") or f"vast-{inst}",
        "rental_id": rental["id"], "instance": inst, "label": rental.get("label"),
        "ssh": {"host": rental.get("ssh_host"), "port": rental.get("ssh_port"), "user": "root"},
        "forge_root": BOX_ROOT, "gpu_class": gpu_class, "gpus": {"timing": [0], "work": []},
        "python": "python3", "sched_python": "python3", "python_for": {}, "recipes": list(recipes),
        "image": image, "usd_per_hour": rental.get("usd_per_hour"), "rented_at": rental.get("created_at"), "started_at": rental.get("created_at"),
        "expires_at": rental.get("expires_at"),
        "env": {"HF_HOME": "/root/hf", "FORGE_ROOT": BOX_ROOT, "LANDLORD_URL": BOX_LANDLORD,
                "LANDLORD_BUILDING": rental.get("building") or f"vast-{inst}", "PYTHONPATH": BOX_ROOT},
        "results_dir": f"{BOX_ROOT}/results",
        "status": "registered",
    }
    save_machine(m)
    return m


def update_ssh(m: dict, rental: dict) -> dict:
    if rental.get("ssh_host"):
        m["ssh"] = {"host": rental["ssh_host"], "port": rental.get("ssh_port"), "user": "root"}
        if rental.get("usd_per_hour"):
            m["usd_per_hour"] = rental["usd_per_hour"]
        save_machine(m)
    return m


def rig_pubkey() -> str:
    rig = executor_for(get_machine("rig"))
    r = rig.sh("cat ~/.ssh/id_ed25519_forge.pub")
    if r.rc != 0:
        raise ExecutorError("rig has no ~/.ssh/id_ed25519_forge.pub")
    return r.stdout.strip()


def wait_ssh(m: dict, timeout_s: float = 600) -> bool:
    ex = executor_for(m)
    end = time.time() + timeout_s
    while time.time() < end:
        r = ex.run("echo ok", timeout_s=30)
        if r.ok and "ok" in r.stdout:
            return True
        time.sleep(10)
    return False


def allow_rig(m: dict, pub: str) -> None:
    ex = executor_for(m)
    cur = ex.run("cat ~/.ssh/authorized_keys 2>/dev/null", timeout_s=30).stdout
    keys = [k.strip() for k in re.split(r"(?=(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-\S+) )", cur) if k.strip()]
    if pub not in keys:
        keys.append(pub)
    ex.sh("mkdir -p ~/.ssh && chmod 700 ~/.ssh")
    ex.write_bytes("/root/.ssh/authorized_keys", ("\n".join(keys) + "\n").encode(), mode="600")


def install(m: dict) -> int:
    ex = executor_for(m)
    n = ex.put_tree(REPO, BOX_ROOT, excludes=("__pycache__", ".git", "state", "results", "*.pyc", "runs", ".claude"))
    ex.sh(f"mkdir -p {BOX_ROOT}/results {BOX_ROOT}/state")
    return n


def restart_services(m: dict) -> None:
    """Redeploy-time restart: stop the sched daemon and gate service this fleet started on a box (single
    tenant), then rerun setup's sched/gate steps on the new code, rewriting the gate lock."""
    ex = executor_for(m)
    ex.sh("pkill -TERM -f '[f]orge.gate.service' ; pkill -TERM -f '[f]orge.sched.daemon' ; sleep 8; "
          "pgrep -f '[f]orge.(gate.service|sched.daemon)' || true", timeout_s=60)
    start_setup(m, redo="sched,gate_lock,gate_service")


def start_setup(m: dict, redo: str = "") -> None:
    ex = executor_for(m)
    plan = {"name": m["name"], "building": m["building"], "gpu_class": m["gpu_class"], "recipes": m["recipes"],
            "landlord_url": BOX_LANDLORD, "timing": m["gpus"]["timing"], "work": m["gpus"]["work"]}
    ex.write_text(f"{BOX_ROOT}/state/box_plan.json", json.dumps(plan, indent=1))
    st = ex.job_status("boxsetup")
    if st["state"] == "running":
        return
    ex.start_job("boxsetup", f"python3 -m forge.fleet.boxsetup --plan {BOX_ROOT}/state/box_plan.json"
                 + (f" --redo {redo}" if redo else ""),
                 cwd=BOX_ROOT, env=m.get("env"))


def box_status(m: dict) -> dict:
    ex = executor_for(m)
    try:
        st = json.loads(ex.read_text(f"{BOX_ROOT}/state/box_status.json"))
    except (FileNotFoundError, ValueError):
        st = {}
    except ExecutorError as e:
        return {"reachable": False, "error": str(e)}
    st["reachable"] = True
    st["job"] = ex.job_status("boxsetup")
    st.pop("plan", None)
    return st


def provision(m: dict, pub: str | None = None, log=print) -> dict:
    """ssh up -> rig key -> forge tree -> setup job. Safe to rerun."""
    if not (m.get("ssh") or {}).get("host"):
        raise ExecutorError(f"{m['name']}: no ssh address yet (still provisioning at the provider)")
    if not wait_ssh(m, timeout_s=120):
        raise ExecutorError(f"{m['name']}: ssh does not answer")
    allow_rig(m, pub or rig_pubkey())
    n = install(m)
    log(f"{m['name']}: forge installed ({n / 1e6:.1f} MB)")
    start_setup(m)
    m["status"] = "setting_up"
    m["setup_started"] = time.time()
    save_machine(m)
    return box_status(m)


def rig_can_reach(m: dict) -> bool:
    """True when the rig itself can ssh to the box (the mission runner needs this)."""
    rig = executor_for(get_machine("rig"))
    s = m["ssh"]
    cmd = (f"ssh -i ~/.ssh/id_ed25519_forge -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 "
           f"-o StrictHostKeyChecking=accept-new -p {s['port']} root@{s['host']} echo reach-ok")
    r = rig.run(cmd, timeout_s=60)
    return "reach-ok" in r.stdout
