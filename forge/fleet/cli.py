"""forge: fleet CLI.

  forge doctor                         check rig, landlord, schedulers, boxes, reachability
  forge deploy <machine>|all           tar forge to a machine (+ machine.json, machine registry on the rig)
  forge rent --offer N --gpu-class C --recipes a,b [--hours 6 --image I]   rent through landlord + register
  forge rent --rental rent_x --gpu-class C --recipes a,b --image I         register an existing rental
  forge boxes [--provision]            rentals -> ssh addresses -> install + setup; show setup progress
  forge missions [list|add <recipe> <machine> [--wave N]|plan --wave N]
  forge run-fleet [--once] [--every 30]  dispatch / monitor / leaderboard loop (+ result puller sync)
  forge status                         machines, missions, spend
  forge leaderboard                    rebuild and print the leaderboard
  forge stop [<mission_id>]            stop running missions this controller started
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from forge.fleet import provision as prov
from forge.fleet.executor import ExecutorError, executor_for
from forge.fleet.machines import all_machines, get_machine, machine_json, save_machine, state_dir
from forge.recipes import fits, load_all

TENANT = "fleet@squaredcube/recipe-forge"


def landlord():
    from forge.sched.landlord import LandlordClient
    return LandlordClient(tenant=TENANT, timeout=600)


def p(obj) -> None:
    print(json.dumps(obj, indent=1, default=str) if not isinstance(obj, str) else obj)


# ---------------------------------------------------------------------- doctor
def cmd_doctor(a) -> int:
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            res = fn()
            print(f"  ok   {name}: {res}")
        except Exception as e:
            ok = False
            print(f"  FAIL {name}: {type(e).__name__}: {str(e)[:300]}")

    rig = executor_for(get_machine("rig"))
    print("rig")
    check("ssh", lambda: rig.sh("hostname").stdout.strip())
    check("disk / free GB", lambda: _disk(rig))
    check("forge deployed", lambda: rig.sh("test -f ~/forge/forge/__init__.py && echo yes || echo NO").stdout.strip())
    check("scheduler daemon", lambda: rig.sh("curl -s -m 5 127.0.0.1:7791/health || echo DOWN").stdout.strip()[:120])
    check("secrets file present", lambda: rig.sh("test -f ~/forge_secrets.env && stat -c %a ~/forge_secrets.env || echo MISSING").stdout.strip())
    check("forge key", lambda: rig.sh("test -f ~/.ssh/id_ed25519_forge && echo yes || echo MISSING").stdout.strip())
    print("landlord")
    check("overview", lambda: [(b["building"], b["trust"]) for b in landlord().call("landlord_overview")["buildings"]])
    for name, m in all_machines().items():
        if m.get("kind") != "vast":
            continue
        print(name)
        check("box status", lambda m=m: _short_status(prov.box_status(m)))
        check("rig can reach", lambda m=m: prov.rig_can_reach(m))
    return 0 if ok else 1


def _disk(ex) -> str:
    return ex.sh("df -BG --output=avail / | tail -1").stdout.strip()


def _short_status(st: dict) -> dict:
    return {"reachable": st.get("reachable"), "ready": st.get("ready"), "current": st.get("current"),
            "failed": st.get("failed"), "job": (st.get("job") or {}).get("state"),
            "done": [k for k, v in (st.get("steps") or {}).items() if v.get("ok")],
            "error": ((st.get("steps") or {}).get(st.get("failed") or "", {}) or {}).get("error")}


# ---------------------------------------------------------------------- deploy
def cmd_deploy(a) -> int:
    names = [n for n in all_machines()] if a.machine == "all" else [a.machine]
    rc = 0
    for n in names:
        try:
            deploy(get_machine(n))
        except (ExecutorError, OSError) as e:
            print(f"{n}: deploy FAILED: {e}")
            rc = 1
    return rc


def deploy(m: dict) -> None:
    ex = executor_for(m)
    root = m["forge_root"]
    n = ex.put_tree(prov.REPO, root, excludes=("__pycache__", ".git", "state", "results", "*.pyc", "runs", ".claude"))
    # verify: every .py file on the target matches the bytes that were sent
    local = {k: v for k, v in ex.last_manifest.items() if k.endswith(".py") and k.startswith(("forge/", "bin/"))}
    r = ex.sh(f"cd {root} && find forge bin -name '*.py' -not -path '*/__pycache__/*' -exec md5sum {{}} +", timeout_s=120)
    remote = {line.split()[1]: line.split()[0] for line in r.stdout.splitlines() if len(line.split()) == 2}
    bad = [k for k, v in local.items() if remote.get(k) != v]
    if bad:
        raise OSError(f"{len(bad)} files differ after deploy, e.g. {bad[:3]}")
    # machine.json for core's scheduler/gate (only when the machine is static or the box is set up)
    if m.get("kind") != "vast":
        code = ("import json,sys; from forge.sched import machine as mm; import os; "
                "p=os.path.join(mm.forge_root(),'machine.json'); m=mm.load('/nonexistent'); d=json.loads(sys.argv[1]); "
                "m.name=d['name']; m.building=d['building']; m.timing_gpus=d['timing_gpus']; m.work_gpus=d['work_gpus']; "
                "open(p,'w').write(json.dumps(m.to_dict(),indent=1)); print(p)")
        ex.sh(f"cd {root} && FORGE_ROOT={root} PYTHONPATH={root} {m.get('sched_python') or m['python']} -c "
              f"{_sq(code)} {_sq(json.dumps(machine_json(m)))}")
        # the machine registry (boxes) so the mission runner on the rig can reach them
        sync_registry(ex, root)
    print(f"{m['name']}: deployed {n / 1e6:.2f} MB to {root}, {len(local)} .py files verified")


def sync_registry(ex, root: str) -> None:
    """Copy the box registry to a machine in one transfer (one ssh call)."""
    from forge.fleet.machines import dynamic_dir
    ex.put_tree(dynamic_dir(), f"{root}/state/fleet/machines", excludes=("*.tmp",))


def _sq(s: str) -> str:
    import shlex
    return shlex.quote(s)


# ---------------------------------------------------------------------- rent / boxes
def cmd_rent(a) -> int:
    recipes = [r for r in a.recipes.split(",") if r]
    for b in recipes:
        r = next((x for x in load_all() if x["base_id"] == b), None)
        if r is None:
            print(f"unknown recipe {b}")
            return 2
        if not fits(r, a.gpu_class):
            print(f"{b} does not fit {a.gpu_class}")
            return 2
    ll = landlord()
    if a.rental:
        rentals = ll.call("list_rentals")["rentals"]
        rental = next((x for x in rentals if x["id"] == a.rental), None)
        if not rental:
            print(f"no active rental {a.rental}")
            return 2
    else:
        if not a.purpose:
            print("--purpose is required when renting (what runs and why)")
            return 2
        res = ll.call("rent_machine", offer=a.offer, purpose=a.purpose, hours=a.hours, image=a.image,
                      disk_gb=a.disk, label=a.label, save=["/root/forge/results"], wait_seconds=20, tenant=TENANT)
        rental = res["rental"]
    m = prov.register_rental(rental, a.gpu_class, recipes, a.image, name=a.name)
    print(f"registered {m['name']} ({rental['id']}, {rental.get('status')})")
    return 0


def sync_rentals(log=print) -> list[dict]:
    """Update every registered box from landlord's rental list; retire returned ones."""
    try:
        res = landlord().call("list_rentals")
    except Exception as e:
        log(f"list_rentals failed: {e}")
        return []
    by_id = {r["id"]: r for r in res.get("rentals", [])}
    out = []
    for name, m in all_machines().items():
        if m.get("kind") != "vast":
            continue
        r = by_id.get(m["rental_id"])
        if r is None:
            from forge.fleet.machines import retire_machine
            retire_machine(name, "rental no longer active")
            log(f"{name}: rental gone, retired")
            continue
        prov.update_ssh(m, r)
        m["rental_status"] = r.get("status")
        m["started_at"] = r.get("created_at") or m.get("started_at")
        m["usd_per_hour"] = r.get("usd_per_hour") or m.get("usd_per_hour")
        m["spent_usd"] = r.get("spent_usd")
        m["hours_left"] = r.get("hours_left")
        save_machine(m)
        out.append(m)
    try:
        with open(state_dir() / "spend.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "source": "vast", "burn": res.get("burn"),
                                "credit_usd": res.get("credit_usd"),
                                "rentals": [{k: r.get(k) for k in ("id", "building", "label", "usd_per_hour",
                                                                    "spent_usd", "hours_left", "status")}
                                            for r in res.get("rentals", [])]}) + "\n")
    except OSError:
        pass
    return out


def boxes(provision: bool, log=print) -> list[dict]:
    rows = []
    pub = None
    for m in sync_rentals(log):
        st = {}
        if (m.get("ssh") or {}).get("host"):
            try:
                st = prov.box_status(m)
                if provision and st.get("reachable") and not st.get("job", {}).get("state") in ("running",) \
                        and not st.get("ready"):
                    pub = pub or prov.rig_pubkey()
                    st = prov.provision(m, pub, log=log)
                deps = (st.get("steps") or {}).get("deps") or {}
                if "landlock_abi" in deps and m.get("landlock_abi") != deps["landlock_abi"]:
                    m["landlock_abi"], m["kernel"] = deps["landlock_abi"], deps.get("kernel")
                    save_machine(m)
                if st.get("ready") and not m.get("ready"):
                    m["ready"] = True
                    m["status"] = "ready"
                    m["ready_at"] = time.time()
                    save_machine(m)
                    log(f"{m['name']}: READY")
            except (ExecutorError, OSError) as e:
                st = {"reachable": False, "error": str(e)[:200]}
        rows.append({"name": m["name"], "rental": m["rental_id"], "status": m.get("rental_status"),
                     "ssh": f"{m['ssh'].get('host')}:{m['ssh'].get('port')}", "recipes": m["recipes"],
                     "spent": m.get("spent_usd"), "hours_left": m.get("hours_left"), **_short_status(st)})
    return rows


def cmd_boxes(a) -> int:
    for r in boxes(a.provision):
        p(r)
    return 0


# ---------------------------------------------------------------------- missions
def cmd_missions(a) -> int:
    from forge.fleet.controller import Controller
    c = Controller()
    if a.action == "add":
        p(c.add(a.recipe, a.machine, wave=a.wave, playbook=a.playbook))
    elif a.action == "n1":
        pairs = [tuple(x.split("@", 1)) for x in (a.pairs or "").split(",") if "@" in x]
        p(c.set_n1_plan(a.wave, pairs, a.time_box))
    elif a.action == "plan":
        n = 0
        for name, m in all_machines().items():
            for r in load_all(m.get("gpu_class")):
                if r["base_id"] in (m.get("recipes") or []) and fits(r, m["gpu_class"]) \
                        and (not a.variants or r["variant"] in a.variants.split(",")):
                    c.add(r["id"], name, wave=a.wave)
                    n += 1
        print(f"queued {n} missions in wave {a.wave}")
    else:
        for it in c.queue():
            print(f"{it['status']:9} w{it.get('wave')} {it['mission_id']:60} {it['recipe_id']:34} {it['machine']} "
                  f"{json.dumps(it.get('row') or {})}")
    return 0


def cmd_run_fleet(a) -> int:
    from forge.fleet.controller import Controller
    c = Controller(defaults={"profile": a.profile} if a.profile else None)
    _judged: set = set()
    while True:
        try:
            boxes(provision=True)
            sync_registry(c.rig, get_machine("rig")["forge_root"])
            started = c.dispatch()
            finished = c.monitor()
            c.n1_tick()
            for w in sorted({i.get("wave") for i in c.queue() if i.get("arm")} - set(_judged)):
                if c.wave_done(w):
                    _judged.add(w)
                    v = c.judge_wave(w)
                    print(time.strftime("%H:%M:%S"), f"N+1 verdict for wave {w}:", v, flush=True)
            c.snapshot()
            if started or finished:
                print(time.strftime("%H:%M:%S"), "started", started, "finished", finished, flush=True)
        except Exception as e:  # keep the loop alive; the error is an alert
            c.alert("controller_error", "run-fleet", f"{type(e).__name__}: {e}")
            print("controller error:", e, flush=True)
        if a.once:
            return 0
        time.sleep(a.every)


def cmd_status(a) -> int:
    from forge.fleet.controller import Controller
    c = Controller()
    snap = c.snapshot()
    for m in snap["machines"]:
        print(f"{m['name']:16} {m['kind']:6} {m['gpu_class'] or '':15} ready={m['ready']!s:5} "
              f"running={m['running']} queued={m['queued']} ${m['usd_per_hour'] or 0}/h")
    q = c.queue()
    counts = {}
    for it in q:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    print("missions:", counts)
    try:
        r = landlord().call("list_rentals")
        print("vast burn:", r.get("burn"), "credit:", r.get("credit_usd"))
    except Exception as e:
        print("landlord:", e)
    return 0


def cmd_leaderboard(a) -> int:
    from forge.fleet.controller import Controller
    lb = Controller().leaderboard()
    print((state_dir().parent / "results" / "leaderboard.md").read_text(encoding="utf-8"))
    return 0 if lb else 1


def cmd_stop(a) -> int:
    from forge.fleet.controller import Controller
    p(Controller().stop(a.mission))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    d = sub.add_parser("deploy")
    d.add_argument("machine")
    r = sub.add_parser("rent")
    r.add_argument("--offer", type=int)
    r.add_argument("--rental")
    r.add_argument("--gpu-class", required=True, choices=["rtxpro6000_95g", "rtx5090_32g"])
    r.add_argument("--recipes", required=True)
    r.add_argument("--hours", type=float, default=6)
    r.add_argument("--image", default="vllm/vllm-openai:latest")
    r.add_argument("--disk", type=float, default=100)
    r.add_argument("--label")
    r.add_argument("--name")
    r.add_argument("--purpose")
    b = sub.add_parser("boxes")
    b.add_argument("--provision", action="store_true")
    ms = sub.add_parser("missions")
    ms.add_argument("action", nargs="?", default="list", choices=["list", "add", "plan", "n1"])
    ms.add_argument("--pairs", help="n1: held-out recipe@machine list run under old and new playbook")
    ms.add_argument("--time-box", type=int, default=45)
    ms.add_argument("recipe", nargs="?")
    ms.add_argument("machine", nargs="?")
    ms.add_argument("--wave", type=int, default=1)
    ms.add_argument("--playbook")
    ms.add_argument("--variants", help="comma list, e.g. c1")
    rf = sub.add_parser("run-fleet")
    rf.add_argument("--once", action="store_true")
    rf.add_argument("--every", type=float, default=30)
    rf.add_argument("--profile", choices=["crusoe", "local", "mock"])
    sub.add_parser("status")
    sub.add_parser("leaderboard")
    s = sub.add_parser("stop")
    s.add_argument("mission", nargs="?")
    a = ap.parse_args(argv)
    fn = {"doctor": cmd_doctor, "deploy": cmd_deploy, "rent": cmd_rent, "boxes": cmd_boxes,
          "missions": cmd_missions, "run-fleet": cmd_run_fleet, "status": cmd_status,
          "leaderboard": cmd_leaderboard, "stop": cmd_stop}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
