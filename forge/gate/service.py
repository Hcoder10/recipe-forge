"""The gate service: one per machine, runs queued gate requests one at a time.

    python -m forge.gate.service            # foreground
    python -m forge.gate.service ensure     # start as systemd --user unit forge-gate if not running
    python -m forge.gate.service once <gate_id>   # run one queued request now (tests / rehearsals)

Order: reference controls first (a recipe needs >= 8 null pairs before its candidate gates count;
they are queued automatically), then confirmations, then candidates with the fewest finished gates per
mission first. Each run holds a scheduler `gate` grant (all timing GPUs, gate cores).
"""
from __future__ import annotations

import fcntl
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

from forge.gate import records as L
from forge.gate import lock as gate_lock
from forge.gate import pool as pool_mod
from forge.gate import runner
from forge.gate.integrity import canon_hash, hash_tree, venv_hash
from forge.gate.server import GateServer, GpuWatch
from forge.sched.machine import forge_root

RETAIN_BYTES = 20 * 1024 * 1024
ENV_DENY_PREFIX = ("FORGE_", "LANDLORD_", "HF_", "CUDA_VISIBLE", "VLLM_CACHE", "VLLM_PORT", "XDG_",
                   "TORCHINDUCTOR_CACHE", "TRITON_CACHE", "CUDA_CACHE")
ENV_DENY = {"HOME", "TMPDIR", "TMP", "TEMP", "PATH", "LD_PRELOAD", "PYTHONHOME"}


def hf_home() -> str:
    return os.path.realpath(os.path.expanduser(os.environ.get("HF_HOME", "~/hf")))


def snapshot_dir(model: str) -> str:
    base = os.path.join(hf_home(), "hub", "models--" + model.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime)
    if not snaps:
        raise FileNotFoundError(f"{model} is not in {hf_home()}/hub")
    return snaps[-1]


def _cpu_busy(cpus: list[int], secs: float = 1.0) -> float:
    def snap():
        out = {}
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu") and line[3].isdigit():
                    p = line.split()
                    c = int(p[0][3:])
                    vals = list(map(int, p[1:]))
                    out[c] = (sum(vals), vals[3] + vals[4])
        return out
    a = snap()
    time.sleep(secs)
    b = snap()
    busy = []
    for c in cpus:
        if c in a and c in b:
            tot = b[c][0] - a[c][0]
            idle = b[c][1] - a[c][1]
            busy.append(1 - idle / tot if tot else 0)
    return sum(busy) / len(busy) if busy else 0.0


class RealCtx:
    """Builds servers, tokenizer and checks for one gate run on this machine."""

    def __init__(self, req: dict, grant, run_dir: str, log):
        from transformers import AutoTokenizer   # heavy; only in the service
        self.req = req
        self.recipe = req["recipe"]
        self.grant = grant
        self.run_dir = run_dir
        self.log = log
        self.python = req["python"]
        self.venv = os.path.dirname(os.path.dirname(self.python))
        self.model = self.recipe["model"]
        self.snap = snapshot_dir(self.model)
        self.tok = AutoTokenizer.from_pretrained(self.snap, local_files_only=True, trust_remote_code=False)
        chat = self.recipe.get("chat") or {}
        self.chat_kwargs = chat.get("chat_template_kwargs") or {}
        self.system = chat.get("system_prompt")
        tp = int((self.recipe.get("hardware") or {}).get("tensor_parallel") or 1)
        self.gpus = list(grant.gpus)[:tp]
        self.uuids = [u for u in grant.gpu_uuids][:tp]
        root = forge_root()
        key = req["recipe_key"].replace("/", "_").replace(":", "_")
        self.ref_cache = os.path.join(root, "gate", "cache", "ref", key)
        self.cand_cache = os.path.join(root, "gate", "cache", "cand")
        shutil.rmtree(self.cand_cache, ignore_errors=True)
        self.extra = {}
        for ef in (self.recipe.get("vllm") or {}).get("extra_files") or []:
            dest = ef["dest"]
            for src in (os.path.join(root, "assets", self.recipe["id"], dest), os.path.join(self.snap, dest)):
                if os.path.exists(src):
                    self.extra[dest] = src
                    break
            else:
                raise FileNotFoundError(f"extra file {dest} not found for {self.recipe['id']}")
        self.bundle = req.get("bundle_copy")
        self.n = 0

    def chat_ids(self, messages: list[dict]) -> list[int]:
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + messages
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **self.chat_kwargs)
        if not isinstance(ids, list):           # BatchEncoding (a Mapping) in newer transformers
            ids = ids["input_ids"]
        ids = [int(t) for t in ids]
        return ids

    def retok(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def make_server(self, side: str, name: str):
        from forge.tools.bundle import serve_command
        serve = self.recipe.get("serve") or {}
        port = self.grant.ports[0]
        hook = os.path.join(self.run_dir, "hook")
        argv, benv = serve_command(self.python, self.model, serve.get("args") or [],
                                   self.bundle if side == "B" else None, port,
                                   hook_dir=hook if side == "B" else None)
        env = {**((self.recipe.get("vllm") or {}).get("env") or {}), **(serve.get("env") or {})}
        for k, v in benv.items():
            if side == "B" and (k in ENV_DENY or k.startswith(ENV_DENY_PREFIX)) and k not in ("FORGE_OVERLAY_MAP",):
                continue
            env[k] = str(v)
        cache = self.ref_cache if side == "A" else self.cand_cache
        env.update({"HF_HOME": hf_home(), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "HF_MODULES_CACHE": os.path.join(cache, "hf_modules")})
        # private caches start cold, so JIT kernels need the venv's own CUDA toolkit
        nvcc = sorted(glob.glob(os.path.join(self.venv, "lib", "python3*", "site-packages", "nvidia", "cu*", "bin",
                                             "nvcc")))
        if nvcc:
            env["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc[-1]))
            env["PATH"] = os.path.dirname(nvcc[-1])
        ro = [self.venv, hf_home()]
        if side == "B":
            ro += [self.bundle] + ([hook] if os.path.isdir(hook) else [])
        lease = (self.grant.leases or {}).get(str(self.gpus[0]))
        self.n += 1
        return GateServer(f"gate-{self.req['gate_id']}-{name}", argv, env,
                          os.path.join(self.run_dir, "srv", name), cache, ro, self.gpus,
                          tuple(self.grant.ports), list(self.grant.cpus), lease, extra_files=self.extra)

    def quiet(self) -> str | None:
        foreign = GpuWatch(list(self.grant.gpu_uuids), own_unit="").check_once()
        if foreign:
            return f"GPU processes present on gate GPUs: {foreign[:2]}"
        busy = _cpu_busy(list(self.grant.cpus))
        if busy > 0.3:
            return f"gate CPU cores busy (mean {busy:.0%})"
        return None

    def watch(self, srv):
        return GpuWatch(list(self.grant.gpu_uuids), own_unit=srv.job.unit if srv.job else "")


def _retain(run_dir: str) -> None:
    """Keep the run dir under RETAIN_BYTES: drop server private dirs, then trim the biggest logs."""
    for d in glob.glob(os.path.join(run_dir, "srv", "*")):
        for sub in ("home", "tmp", "ws"):
            shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    shutil.rmtree(os.path.join(run_dir, "bundle"), ignore_errors=True)
    shutil.rmtree(os.path.join(run_dir, "hook"), ignore_errors=True)
    files = []
    for dp, _, fns in os.walk(run_dir):
        files += [os.path.join(dp, f) for f in fns]
    total = sum(os.path.getsize(f) for f in files)
    for f in sorted(files, key=os.path.getsize, reverse=True):
        if total <= RETAIN_BYTES:
            break
        size = os.path.getsize(f)
        keep = 256 * 1024
        if size > keep and not f.endswith("verdict.json"):
            with open(f, "rb") as fh:
                fh.seek(size - keep)
                tail = fh.read()
            with open(f, "wb") as fh:
                fh.write(b"[trimmed]\n" + tail)
            total -= size - keep


def run_one(req: dict, grant, log=print, ctx_factory=RealCtx) -> dict:
    """Run one gate request under an active gate grant and return the verdict."""
    root = forge_root()
    gid = req["gate_id"]
    run_dir = os.path.join(root, "gate", "runs", gid)
    os.makedirs(run_dir, mode=0o700, exist_ok=True)
    t0 = time.time()
    v: dict = {"gate_id": gid, "kind": req["kind"], "recipe_key": req["recipe_key"], "started": t0}
    try:
        ok, msg, lock_sha = gate_lock.verify(req.get("lock") or "main")
        v["lock"] = {"name": req.get("lock") or "main", "sha256": lock_sha, "ok": ok}
        if not ok:
            raise runner.GateInvalid(msg)
        venv = os.path.dirname(os.path.dirname(req["python"]))
        v["venv_sha256"] = venv_hash(venv, os.path.join(root, "gate", "venv_hashes.json"))
        v["recipe_sha256"] = canon_hash(req["recipe"])
        if req["kind"] != "reference":
            src = req["bundle_dir"]
            copy = os.path.join(run_dir, "bundle")
            shutil.rmtree(copy, ignore_errors=True)
            shutil.copytree(src, copy, symlinks=True, ignore=shutil.ignore_patterns(".hook"))
            for dp, dns, fns in os.walk(copy):
                for n in dns + fns:
                    p = os.path.join(dp, n)
                    if os.path.islink(p):
                        raise runner.GateInvalid(f"bundle contains a symlink: {os.path.relpath(p, copy)}")
            from forge.tools.bundle import hash_dir
            from pathlib import Path
            h = hash_dir(Path(copy))
            if req.get("bundle_hash") and h != req["bundle_hash"]:
                raise runner.GateInvalid("bundle changed after it was submitted (hash mismatch)")
            v["bundle_sha256"] = h
            req = {**req, "bundle_copy": copy}
        pool = pool_mod.build(root)
        v["pool_sha256"] = pool["sha256"]
        slices = pool_mod.Slices(pool)
        hist = runner.History(os.path.join(root, "gate", "history", req["history_key"] + ".json"))
        ctx = ctx_factory(req, grant, run_dir, log)
        rc = runner.RunCtx(make_server=ctx.make_server, chat_ids=ctx.chat_ids, retok=ctx.retok, quiet=ctx.quiet,
                           history=hist, log=log, watch=getattr(ctx, "watch", None))
        wl = runner.Workload(**req["workload"])
        reps = 2 if req["kind"] == "confirm" else int(req.get("reps") or 4)
        res = runner.run(req["kind"], slices, wl, reps, rc, confirm_index=int(req.get("confirm_index") or 0))
        if req["kind"] == "reference":
            runner.record_reference(hist, res, gid)
            res["null_pairs_after"] = hist.n_null
        v.update(res)
    except runner.GateInvalid as e:
        v["verdict"], v["detail"] = "invalid", str(e)[:500]
        v["servers"] = list(getattr(e, "servers", []) or [])
    except Exception as e:  # infra/tool bug: never a win
        v["verdict"], v["detail"] = "error", f"{type(e).__name__}: {str(e)[:400]}"
        v["traceback"] = traceback.format_exc()[-4000:]
    v["finished"] = time.time()
    v["minutes"] = round((v["finished"] - t0) / 60, 2)
    servers = v.pop("servers", [])
    os.makedirs(os.path.join(run_dir, "srv"), exist_ok=True)
    for s in servers:
        with open(os.path.join(run_dir, "srv", f"r{s['rep']}{s['slot']}.log.tail"), "w") as f:
            f.write(s.pop("log_tail", ""))
    v["servers"] = servers
    with open(os.path.join(run_dir, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    _retain(run_dir)
    return v


class Service:
    def __init__(self, log=print):
        self.log = log
        self.stop = threading.Event()

    def _acquire(self, req: dict):
        from forge.sched.client import SchedClient
        mins = int(req.get("minutes") or (25 if req["kind"] == "confirm" else 60))
        g = SchedClient().acquire("gate", owner=f"gate/{req['gate_id']}",
                                  purpose=f"{req['kind']} gate {req['gate_id']} for {req['recipe_key']}: "
                                          f"paired timing + lossless checks, needs all timing GPUs",
                                  minutes=mins, on_wait=lambda g: self.log(f"gate waiting: {g.wait_reason}"))
        return g

    def run_request(self, req: dict) -> dict:
        L.append(req, status="running", started=time.time())
        g = self._acquire(req)
        renew = threading.Event()

        def _renew():
            while not renew.wait(300):
                try:
                    g.renew(30)
                except Exception:
                    pass
        threading.Thread(target=_renew, daemon=True).start()
        try:
            v = run_one(req, g, self.log)
        finally:
            renew.set()
            g.release("gate run finished")
        L.finish(req, v)
        return v

    def loop(self):
        os.sched_setaffinity(0, _gate_cpus())
        while not self.stop.is_set():
            req = L.next_request()
            if req is None:
                self.stop.wait(5)
                continue
            try:
                self.run_request(req)
            except Exception as e:
                L.append(req, status="error", verdict="error", detail=f"{type(e).__name__}: {e}"[:400])
                self.stop.wait(5)


def _gate_cpus() -> list[int]:
    from forge.sched import machine
    return machine.load().gate_cpus


def ensure() -> int:
    r = subprocess.run(["systemctl", "--user", "is-active", "forge-gate.service"], capture_output=True, text=True)
    if r.stdout.strip() == "active":
        print("forge.gate: already running")
        return 0
    pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmd = ["systemd-run", "--user", "--unit=forge-gate", "--collect", f"--working-directory={pkg_parent}",
           f"--setenv=FORGE_ROOT={forge_root()}", f"--setenv=HF_HOME={hf_home()}", "--setenv=HF_HUB_OFFLINE=1",
           sys.executable, "-m", "forge.gate.service"]
    subprocess.run(cmd, check=True)
    print("forge.gate: started (journalctl --user -u forge-gate)")
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "ensure":
        return ensure()
    root = forge_root()
    os.makedirs(os.path.join(root, "gate"), mode=0o700, exist_ok=True)
    os.chmod(os.path.join(root, "gate"), 0o700)
    lockf = open(os.path.join(root, "gate", "service.lock"), "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("forge.gate: another gate is running on this machine (one gate per machine)", file=sys.stderr)
        return 1
    svc = Service(log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    if argv and argv[0] == "once":
        os.sched_setaffinity(0, _gate_cpus())
        req = L.get_request(argv[1])
        v = svc.run_request(req)
        print(json.dumps({k: v.get(k) for k in ("gate_id", "verdict", "detail", "delta_pct", "minutes")}))
        return 0
    svc.loop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
