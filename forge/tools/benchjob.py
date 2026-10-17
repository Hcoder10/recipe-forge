"""Quick bench job. Runs ON the target machine inside a GPU grant (CUDA_VISIBLE_DEVICES set by the scheduler).

    python -m forge.tools.benchjob --spec spec.json

Measures the candidate and, unless a reference for this exact GPU + recipe + prompt set is cached, the
reference first, on the same GPU in the same grant. Prints one JSON line prefixed with BENCH_RESULT.

This is the agents' quick bench on visible prompts. It is not the gate and proves nothing on its own.
Stdlib only (runs in the recipe's venv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from forge.tools.bundle import canon, serve_command


def gpu_identity() -> dict:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    try:
        out = subprocess.run(["nvidia-smi", f"--id={vis}", "--query-gpu=uuid,name,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        uuid, name, power = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"uuid": uuid, "name": name, "power_limit": power}
    except Exception:
        return {"uuid": f"unknown-{vis}", "name": "unknown", "power_limit": ""}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def load_prompts(path: str, n: int | None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append(r if isinstance(r, dict) else {"prompt": r})
    return rows[:n] if n else rows


def wait_ready(port: int, proc, timeout_s: float, log_path: Path) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.poll()} during startup; "
                               f"log tail: {tail(log_path, 6)}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server not ready after {timeout_s:.0f}s; log tail: {tail(log_path, 6)}")


def tail(p: Path, n: int) -> str:
    try:
        lines = p.read_text(errors="replace").splitlines()
        return " | ".join(l.strip()[:200] for l in lines[-n:])
    except Exception:
        return ""


def stream_one(port: int, model: str, row: dict, max_tokens: int, chat: dict | None) -> dict:
    """One streamed request. Returns tokens, ttft, decode time (first to last token)."""
    if chat:
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        msgs = row.get("messages") or [{"role": "user", "content": row.get("prompt", "")}]
        if chat.get("system_prompt") and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": chat["system_prompt"]}] + msgs
        body = {"model": model, "messages": msgs, "max_tokens": row.get("max_tokens", max_tokens),
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        if chat.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = chat["chat_template_kwargs"]
    else:
        url = f"http://127.0.0.1:{port}/v1/completions"
        body = {"model": model, "prompt": row.get("prompt", ""), "max_tokens": row.get("max_tokens", max_tokens),
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            ch = (ev.get("choices") or [{}])[0] if ev.get("choices") else {}
            piece = (ch.get("text") or (ch.get("delta") or {}).get("content") or
                     (ch.get("delta") or {}).get("reasoning_content") or "")
            if piece:
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
    t_end = time.perf_counter()
    n = int(usage.get("completion_tokens") or chunks)
    return {"tokens": n, "ttft_s": (first - t0) if first else None,
            "decode_s": (last - first) if first and last else 0.0, "e2e_s": t_end - t0}


def run_workload(port: int, model: str, prompts: list[dict], wl: dict) -> dict:
    conc = int(wl.get("concurrency", 1))
    max_tokens = int(wl.get("max_tokens", 256))
    chat = wl.get("chat") or None
    if chat is True:
        chat = {"on": True}
    # one untimed warmup request (compile/graph capture paths), not counted
    stream_one(port, model, {"prompt": "Hello", "max_tokens": 16}, 16, chat)
    results: list[dict | None] = [None] * len(prompts)
    t0 = time.perf_counter()
    if conc <= 1:
        for i, p in enumerate(prompts):
            results[i] = stream_one(port, model, p, max_tokens, chat)
    else:
        idx = iter(range(len(prompts)))
        lock = threading.Lock()

        def worker():
            while True:
                with lock:
                    i = next(idx, None)
                if i is None:
                    return
                results[i] = stream_one(port, model, prompts[i], max_tokens, chat)

        ts = [threading.Thread(target=worker) for _ in range(conc)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    wall = time.perf_counter() - t0
    rs = [r for r in results if r]
    tok = sum(r["tokens"] for r in rs)
    dec = [(r["tokens"] - 1) / r["decode_s"] for r in rs if r["decode_s"] > 0 and r["tokens"] > 1]
    out = {"n": len(rs), "tokens": tok, "wall_s": round(wall, 3), "output_tok_s": round(tok / wall, 3) if wall else 0,
           "decode_tok_s": round(sum(dec) / len(dec), 3) if dec else 0.0,
           "ttft_ms_mean": round(1000 * sum(r["ttft_s"] or 0 for r in rs) / max(1, len(rs)), 1),
           "per_prompt_decode": [round(x, 2) for x in dec]}
    metric = wl.get("metric", "decode_tok_s")
    out["metric"] = metric
    out["value"] = out.get(metric, out["decode_tok_s"])
    return out


class _Plain:
    """Unsandboxed server process (only when the spec says sandbox: false)."""

    def __init__(self, argv, env, cwd, log_path):
        self.log_path = log_path
        with open(log_path, "w") as log:
            self.proc = subprocess.Popen(argv, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=cwd or None)

    def poll(self):
        return self.proc.poll()

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except Exception:
                pass


SEED_CACHES = ("flashinfer", "b12x")   # prebuilt JIT kernels; the sandbox has no nvcc to rebuild them


def seed_caches(cache: Path) -> None:
    """One shared, writable bench cache per machine, seeded once from the user's prebuilt kernel caches."""
    dst = cache / "home" / ".cache"
    dst.mkdir(parents=True, exist_ok=True)
    (cache / "vllm").mkdir(parents=True, exist_ok=True)
    for name in SEED_CACHES:
        src = Path.home() / ".cache" / name
        if src.is_dir() and not (dst / name).exists():
            subprocess.run(["cp", "-a", str(src), str(dst / name)], check=False)


class _Sandboxed:
    """Server inside forge.sandbox (Landlock + its own systemd scope); candidates carry agent code."""

    def __init__(self, argv, env, cwd, spec, label, work: Path, ro_extra: list[str], port_range):
        from forge.sandbox import Spec as SbxSpec, launch  # noqa: PLC0415
        venv = str(Path(spec["python"]).parent.parent)
        cache = Path(spec.get("cache_dir") or Path(spec["work_dir"]).parent.parent / "benchcache")
        seed_caches(cache)
        ro = [venv] + [p for p in [spec.get("hf_home"), cwd, *ro_extra] if p]
        gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        job_dir = work / f"sbx-{label}"
        home = job_dir / "home"
        senv = {k: v for k, v in env.items() if k not in ("PATH", "HOME", "CUDA_VISIBLE_DEVICES")}
        senv.update({"VLLM_CACHE_ROOT": str(cache / "vllm"), "HF_MODULES_CACHE": str(home / "hf_modules"),
                     "HF_HUB_OFFLINE": "1", "FLASHINFER_WORKSPACE_BASE": str(cache / "home"),
                     "XDG_CACHE_HOME": str(cache / "home" / ".cache")})
        if spec.get("hf_home"):
            senv["HF_HOME"] = spec["hf_home"]
        for k in ("LANDLORD_LEASE_ID", "FORGE_LEASE_IDS", "FORGE_GRANT_ID"):
            if os.environ.get(k):
                senv[k] = os.environ[k]
        self.job = launch(SbxSpec(cmd=argv, job_dir=str(job_dir), ro=ro, rw=[str(cache)], gpus=[gpu],
                                  ports=port_range, env=senv, path_prepend=[f"{venv}/bin"], cwd=cwd or None,
                                  name=f"bench-{label}-{os.getpid()}"))
        self.log_path = Path(self.job.log_path)

    def poll(self):
        return self.job.poll()

    def stop(self):
        try:
            self.job.stop()
        finally:
            self.job.cleanup(keep_log=True)


def serve_and_measure(spec: dict, bundle_dir, config_delta, env_delta, label: str, work: Path) -> dict:
    rec = spec["recipe"]
    lo, hi = os.environ.get("FORGE_PORT_LO"), os.environ.get("FORGE_PORT_HI")
    port = int(lo) if lo else free_port()
    hook = work / f"hook-{label}"
    argv, extra_env = serve_command(spec["python"], rec["model"], rec["serve_args"], bundle_dir, port,
                                    config_delta, env_delta, hook_dir=hook)
    env = {**(rec.get("env") or {}), **extra_env}
    t0 = time.time()
    if spec.get("sandbox", True):
        ro_extra = [str(hook)] if hook.exists() else []
        if bundle_dir:
            ro_extra.append(str(bundle_dir))
        srv = _Sandboxed(argv, env, spec.get("serve_cwd"), spec, label, work, ro_extra,
                         (int(lo), int(hi)) if lo and hi else (port, port))
    else:
        env.setdefault("VLLM_CACHE_ROOT", str(work / f"cache-{label}"))
        srv = _Plain(argv, env, spec.get("serve_cwd"), work / f"server-{label}.log")
    try:
        wait_ready(port, srv, float(spec.get("startup_timeout_s", 900)), srv.log_path)
        startup = time.time() - t0
        prompts = load_prompts(spec["prompts"], spec.get("n_prompts"))
        res = run_workload(port, rec.get("served_model_name") or rec["model"], prompts, spec["workload"])
        res["startup_s"] = round(startup, 1)
        return res
    finally:
        srv.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    work = Path(spec["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    gpu = gpu_identity()
    out: dict = {"gpu": gpu}
    try:
        prompts_hash = hashlib.sha256(Path(spec["prompts"]).read_bytes()).hexdigest()[:16]
        key = hashlib.sha256(canon({"gpu": gpu["uuid"], "recipe": spec["recipe"], "prompts": prompts_hash,
                                    "n": spec.get("n_prompts"), "wl": spec["workload"],
                                    "sandbox": spec.get("sandbox", True)}).encode()).hexdigest()[:20]
        cache = Path(spec["ref_cache_dir"]) / f"{key}.json"
        out["ref_key"] = key
        if cache.exists() and not spec.get("fresh_reference"):
            out["reference"] = json.loads(cache.read_text())
            out["reference_cached"] = True
        else:
            out["reference"] = serve_and_measure(spec, None, None, None, "ref", work)
            out["reference"]["measured_at"] = time.time()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(out["reference"]))
            out["reference_cached"] = False
        if spec.get("candidate") is not None:
            c = spec["candidate"]
            out["candidate"] = serve_and_measure(spec, c.get("bundle_dir"), c.get("config_delta"),
                                                 c.get("env"), "cand", work)
            r, v = out["reference"]["value"], out["candidate"]["value"]
            out["delta_pct"] = round(100.0 * (v - r) / r, 3) if r else None
        out["ok"] = True
    except Exception as e:  # report one line; the tool layer classifies it
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    print("BENCH_RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
