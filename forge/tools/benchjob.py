"""Quick bench job. Runs ON the target machine inside a GPU grant (CUDA_VISIBLE_DEVICES set by the scheduler).

    python -m forge.tools.benchjob --spec spec.json

Measures the candidate and, unless a reference for this exact GPU + recipe + prompt set is cached, the
reference first, on the same GPU in the same grant. Prints one JSON line prefixed with BENCH_RESULT.

This is the agents' quick bench on visible prompts. It is not the gate and proves nothing on its own.
Stdlib only (runs in the recipe's venv).
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from forge.tools.bundle import canon, serve_command


def gpu_identity() -> dict:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    try:
        out = subprocess.run(["nvidia-smi", f"--id={vis}", "--query-gpu=index,uuid,name,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        index, uuid, name, power = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"index": index, "uuid": uuid, "name": name, "power_limit": power}
    except Exception:
        return {"index": vis if vis.isdigit() else None, "uuid": f"unknown-{vis}",
                "name": "unknown", "power_limit": ""}


def confirmation_threshold(value=None) -> float:
    value = float(value if value is not None else os.environ.get("FORGE_BENCH_CONFIRM_PCT", "2.0"))
    if not math.isfinite(value) or value < 0:
        raise ValueError("confirm_pct must be finite and nonnegative")
    return value


def same_gpu(reference: dict, current: dict) -> bool:
    index = reference.get("index")
    uuid = reference.get("uuid")
    return ((index is not None and current.get("index") is not None
             and str(index) == str(current["index"])) or
            (bool(uuid) and not uuid.startswith("unknown-") and uuid == current.get("uuid")))


def comparison(reference: float, candidate: float, fresh: float | None = None) -> dict:
    delta = 100.0 * (candidate - reference) / reference if reference else None
    out = {}
    noise = 0.0
    if fresh is not None:
        noise = 100.0 * abs(reference - fresh) / fresh if fresh else None
        baseline = (reference + fresh) / 2 if noise is not None and noise <= 2.0 else fresh
        out.update(confirmed=True, delta_pct_vs_cached=round(delta, 3) if delta is not None else None,
                   noise_pct=round(noise, 3) if noise is not None else None)
        delta = 100.0 * (candidate - baseline) / baseline if baseline else None
    out["delta_pct"] = round(delta, 3) if delta is not None else None
    out["verdict_hint"] = (None if delta is None else "within_noise" if abs(delta) < max(noise or 0, 1.0)
                           else "gain" if delta > 0 else "loss")
    return out


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


def stream_one(port: int, model: str, row: dict, max_tokens: int, chat: dict | None,
               token_clock=None) -> dict:
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
    requested_tokens = int(body["max_tokens"])
    body.update(ignore_eos=True, min_tokens=requested_tokens)
    fixed_length = True
    for attempt in range(2):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            response = urllib.request.urlopen(req, timeout=900)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or attempt:
                raise
            exc.close()
            # Older servers may reject the extra sampling parameters.
            body.pop("ignore_eos")
            body.pop("min_tokens")
            fixed_length = False
    first = last = None
    chunks = 0
    usage = {}
    with response as r:
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
                if token_clock is not None:
                    token_clock(time.time_ns() / 1000)
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
    t_end = time.perf_counter()
    reported_tokens = usage.get("completion_tokens")
    n = int(reported_tokens) if reported_tokens is not None else (requested_tokens if fixed_length else chunks)
    mismatch = fixed_length and reported_tokens is not None and abs(n - requested_tokens) * 100 > requested_tokens
    return {"tokens": n, "ttft_s": (first - t0) if first else None,
            "requested_tokens": requested_tokens, "completion_tokens": reported_tokens,
            "streamed_content_pieces": chunks, "fixed_length": fixed_length,
            "token_count_mismatch": mismatch,
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
           "per_prompt_decode": [round(x, 2) for x in dec],
           "per_request": results,
           "fixed_length": all(r["fixed_length"] for r in rs),
           "token_count_mismatch": any(r["token_count_mismatch"] for r in rs)}
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


def cuda_home(python: str) -> str | None:
    """The venv's pip CUDA toolkit (nvidia/cu*/bin/nvcc), for JIT kernel builds when /usr/local/cuda is absent."""
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    venv = Path(python).parent.parent
    for nvcc in sorted(venv.glob("lib/python3*/site-packages/nvidia/cu*/bin/nvcc"), reverse=True):
        return str(nvcc.parent.parent)
    return None


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
        # Short path: vLLM puts ZMQ ipc sockets in TMPDIR and unix socket paths max out at 107 bytes.
        sbx_root = Path(spec.get("sbx_root") or Path(spec["ref_cache_dir"]).parent / "sbx")
        job_dir = sbx_root / f"{os.getpid() % 100000}{label[:1]}{int(time.time()) % 100000}"
        home = job_dir / "home"
        # vLLM's ZMQ ipc sockets live under VLLM_RPC_BASE_PATH / TMPDIR; job_dir is too deep for the 107-byte
        # unix-socket limit, so give each server a short private dir under /tmp.
        short_tmp = Path(f"/tmp/fz{os.getpid() % 100000}{label[:1]}{int(time.time()) % 100000}")
        short_tmp.mkdir(mode=0o700, parents=True, exist_ok=True)
        senv = {k: v for k, v in env.items() if k not in ("PATH", "HOME", "CUDA_VISIBLE_DEVICES")}
        senv.update({"VLLM_RPC_BASE_PATH": str(short_tmp), "TMPDIR": str(short_tmp), "TMP": str(short_tmp),
                     "TEMP": str(short_tmp)})
        senv.update({"VLLM_CACHE_ROOT": str(cache / "vllm"), "HF_MODULES_CACHE": str(home / "hf_modules"),
                     "HF_HUB_OFFLINE": "1", "FLASHINFER_WORKSPACE_BASE": str(cache / "home"),
                     "XDG_CACHE_HOME": str(cache / "home" / ".cache")})
        if spec.get("hf_home"):
            senv["HF_HOME"] = spec["hf_home"]
        cuda = cuda_home(spec["python"])
        if cuda and "CUDA_HOME" not in senv:
            senv["CUDA_HOME"] = cuda
        for k in ("LANDLORD_LEASE_ID", "FORGE_LEASE_IDS", "FORGE_GRANT_ID"):
            if os.environ.get(k):
                senv[k] = os.environ[k]
        writable = [str(cache), str(short_tmp)]
        if spec.get("mode") == "profile":
            writable.append(str(work / "trace"))
        self.job = launch(SbxSpec(cmd=argv, job_dir=str(job_dir), ro=ro, rw=writable, gpus=[gpu],
                                  ports=port_range, env=senv, path_prepend=[f"{venv}/bin"], cwd=cwd or None,
                                  name=f"bench-{label}-{os.getpid()}"))
        self.log_path = Path(self.job.log_path)
        self.job_dir = job_dir

    def poll(self):
        return self.job.poll()

    def stop(self):
        try:
            self.job.stop()
        finally:
            self.job.cleanup(keep_log=True)
            try:  # keep the server log with the bench results
                subprocess.run(["mv", str(self.log_path), str(self.work_log)], check=False)
                subprocess.run(["rm", "-rf", str(self.job_dir)], check=False)
            except Exception:
                pass


def serve_and_measure(spec: dict, bundle_dir, config_delta, env_delta, label: str, work: Path) -> dict:
    rec = spec["recipe"]
    lo, hi = os.environ.get("FORGE_PORT_LO"), os.environ.get("FORGE_PORT_HI")
    port = int(lo) if lo else free_port()
    hook = work / f"hook-{label}"
    argv, extra_env = serve_command(spec["python"], rec["model"], rec["serve_args"], bundle_dir, port,
                                    config_delta, env_delta, hook_dir=hook)
    env = {**(rec.get("env") or {}), **extra_env}
    if spec.get("mode") == "profile":
        trace = work / "trace"
        trace.mkdir(parents=True, exist_ok=True)
        if spec.get("profiler_config"):
            argv += ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(trace),
                                                     **spec.get("profiler_options", {})})]
        else:
            env["VLLM_TORCH_PROFILER_DIR"] = str(trace)
    t0 = time.time()
    if spec.get("sandbox", True):
        ro_extra = [str(hook)] if hook.exists() else []
        if bundle_dir:
            ro_extra.append(str(bundle_dir))
        srv = _Sandboxed(argv, env, spec.get("serve_cwd"), spec, label, work, ro_extra,
                         (int(lo), int(hi)) if lo and hi else (port, port))
        srv.work_log = work / f"server-{label}.log"
    else:
        env.setdefault("VLLM_CACHE_ROOT", str(work / f"cache-{label}"))
        if cuda_home(spec["python"]):
            env.setdefault("CUDA_HOME", cuda_home(spec["python"]))
        srv = _Plain(argv, env, spec.get("serve_cwd"), work / f"server-{label}.log")
    try:
        wait_ready(port, srv, float(spec.get("startup_timeout_s", 900)), srv.log_path)
        startup = time.time() - t0
        if spec.get("mode") == "profile":
            return profile_workload(port, rec, spec, work)
        prompts = load_prompts(spec["prompts"], spec.get("n_prompts"))
        res = run_workload(port, rec.get("served_model_name") or rec["model"], prompts, spec["workload"])
        res["startup_s"] = round(startup, 1)
        return res
    finally:
        srv.stop()


_CATEGORIES = [
    ("communication", r"nccl|all_?reduce"),
    ("memcpy/memset", r"memcpy|memset"),
    ("ssm/mamba", r"selective_scan|ssd|mamba|causal_conv1d|chunk_scan|state_passing"),
    ("moe", r"moe|topk.*expert|expert|grouped_gemm"),
    ("attention", r"flash|fmha|paged|mla"),
    ("norm/act", r"rms_?norm|layer_?norm|silu|gelu|act_and_mul"),
    ("quant/dequant", r"fp4|fp8|nvfp4|scaled|quant"),
    ("dense gemm", r"gemm|cutlass|cublas|nvjet|sm(?:90|100|120).*xmma"),
    ("sampler/topk/softmax", r"sampl|top_?k|softmax"),
]


def parse_trace(trace, steps: int, decode_windows=None) -> dict:
    """Aggregate GPU activities only; busy time is the union, not sum, of intervals.

    Windows and Chrome timestamps are microseconds in the same clock domain.
    Without explicit windows the activity extent is used (useful for trace fixtures).
    """
    if isinstance(trace, (str, Path)):
        path = Path(trace)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            trace = json.load(f)
    activities = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"
                  and re.search(r"kernel|gpu_memcpy|gpu_memset", e.get("cat", ""), re.I)
                  and float(e.get("dur", 0)) > 0]
    if not activities or steps <= 0:
        raise ValueError("no decode GPU activities")
    windows = ([(min(e["ts"] for e in activities), max(e["ts"] + e["dur"] for e in activities))]
               if decode_windows is None else decode_windows)
    def merged(intervals):
        result = []
        for start, end in sorted(intervals):
            if end <= start:
                continue
            if result and start <= result[-1][1]:
                result[-1][1] = max(end, result[-1][1])
            else:
                result.append([start, end])
        return result
    windows = merged(windows)
    totals = {key: 0.0 for key, _ in _CATEGORIES}
    totals["other"] = 0.0
    kernels, busy, launches = {}, [], 0
    for e in activities:
        clips = [(max(e["ts"], a), min(e["ts"] + e["dur"], b)) for a, b in windows]
        clips = [(a, b) for a, b in clips if b > a]
        duration = sum(b - a for a, b in clips)
        if not duration:
            continue
        busy.extend(clips)
        name = e.get("name", "unknown")
        category = next((key for key, regex in _CATEGORIES if re.search(regex, name, re.I)), "other")
        totals[category] += duration
        if "kernel" in e.get("cat", "").lower():
            launches += 1
            item = kernels.setdefault(name, {"name": name, "time_us": 0.0, "count": 0})
            item["time_us"] += duration
            item["count"] += 1
    total = sum(totals.values())
    window = sum(b - a for a, b in windows)
    if not total or not window:
        raise ValueError("trace clock does not overlap decode window")
    busy_pct = 100 * sum(b - a for a, b in merged(busy)) / window
    return {"time_by_category_pct": {k: round(100 * v / total, 4) for k, v in totals.items()},
            "busy_pct": round(busy_pct, 4), "idle_pct": round(100 - busy_pct, 4),
            "launches_per_step": launches / steps,
            "top_kernels": [{**k, "name": k["name"][:160]} for k in
                            sorted(kernels.values(), key=lambda k: -k["time_us"])[:15]]}


def extract_structure(config: dict, recipe=None) -> dict:
    """Read structural fields only; never return arbitrary config or environment text."""
    from forge.agents.winlib import arch_traits
    from forge.tools.bundle import args_to_dict
    recipe = recipe or {}
    recipe = {**recipe, "serve": recipe.get("serve") or {"args": recipe.get("serve_args")}}
    cfg = {**config, **(config.get("text_config") or {})}
    structure = arch_traits(recipe, cfg)
    pattern = cfg.get("hybrid_override_pattern") or cfg.get("layers_block_type") or cfg.get("layer_types") or []
    pattern = list(pattern) if isinstance(pattern, str) and re.fullmatch(r"[M*EAS-]+", pattern) else pattern
    if isinstance(pattern, str):
        pattern = re.split(r"[,\s]+", pattern)
    structure["layer_mix"] = {**structure["layer_mix"], "mlp": sum(
        str(x).lower() in ("-", "mlp", "feed_forward", "ffn") for x in pattern)}
    structure["architectures"] = cfg.get("architectures") or []
    structure["num_experts_per_tok"] = cfg.get("num_experts_per_tok", cfg.get("moe_top_k", cfg.get("top_k")))
    flags = args_to_dict(recipe["serve"].get("args"))
    drafter = flags.get("--speculative-config") or recipe.get("drafter") or {}
    if isinstance(drafter, str):
        try:
            drafter = json.loads(drafter)
        except ValueError:
            drafter = {}
    drafter = dict(drafter) if isinstance(drafter, dict) else {}
    for key in ("model", "method", "num_speculative_tokens"):
        value = flags.get("--speculative-config." + key.replace("_", "-"))
        if value is not None:
            drafter[key] = int(value) if key == "num_speculative_tokens" else value
    structure["drafter_config"] = {k: drafter[k] for k in ("model", "method", "num_speculative_tokens")
                                   if isinstance(drafter, dict) and k in drafter}
    return structure


def inspect_profile_target(spec):
    """Inspect the recipe interpreter's installed sources without importing model code."""
    script = '''import importlib.util,json
from pathlib import Path
s=importlib.util.find_spec("vllm")
print(json.dumps(str(Path(s.origin).parent) if s else None))
'''
    config_path = Path(spec["recipe"]["model"]) / "config.json"
    if not config_path.is_file():
        hf = Path(spec.get("hf_home") or os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
        repo = hf / "hub" / ("models--" + spec["recipe"]["model"].replace("/", "--"))
        revision = (repo / "refs/main").read_text().strip()
        config_path = repo / "snapshots" / revision / "config.json"
    structure = extract_structure(json.loads(config_path.read_text()), spec["recipe"])
    files = []
    try:
        probe = subprocess.run([spec["python"], "-c", script], capture_output=True, text=True, timeout=30)
        root_value = json.loads(probe.stdout)
        root = Path(root_value) if root_value else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return structure, files, False
    if root:
        registry = root / "model_executor/models/registry.py"
        try:
            tree = ast.parse(registry.read_text())
        except (OSError, SyntaxError):
            tree = ast.Module(body=[], type_ignores=[])
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in structure["architectures"]:
                    try:
                        entry = ast.literal_eval(value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], str):
                        path = root / "model_executor/models" / (entry[0] + ".py")
                        if path.is_file():
                            files.append(str(path))
        # Source inspection is done on the actual installed version, not a guessed version number.
        endpoints = legacy = modern = False
        for path in root.rglob("*.py"):
            try:
                source = path.read_text(errors="replace")
            except OSError:
                continue
            endpoints |= "start_profile" in source and "stop_profile" in source
            legacy |= '"VLLM_TORCH_PROFILER_DIR"' in source or "'VLLM_TORCH_PROFILER_DIR'" in source
            modern |= "torch_profiler_dir" in source and "ProfilerConfig" in source
        spec["profiler_config"] = modern
        if modern:
            config_source = (root / "config/profiler.py")
            source = config_source.read_text() if config_source.is_file() else ""
            spec["profiler_options"] = {k: v for k, v in {
                "ignore_frontend": True, "torch_profiler_with_stack": False,
                "torch_profiler_dump_cuda_time_total": False,
                "torch_profiler_use_gzip": True,
            }.items() if re.search(r"\b" + k + r"\s*:", source)}
        available = endpoints and (legacy or modern)
    else:
        available = False
    return structure, sorted(set(files)), available


def profile_workload(port, recipe, spec, work):
    from forge.tools.bundle import args_to_dict
    flags = args_to_dict(recipe.get("serve_args"))
    model = recipe.get("served_model_name") or flags.get("--served-model-name") or recipe["model"]
    chat = spec["workload"].get("chat") or None
    if chat is True:
        chat = {"on": True}
    steps = int(spec.get("steps", 64))
    stream_one(port, model, {"prompt": "Hello"}, steps, chat)
    def post(action):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/{action}_profile", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=180):
            pass
    windows, decode_steps = [], 0
    trace_dir = work / "trace"
    existing = set(trace_dir.rglob("*"))
    post("start")
    try:
        for _ in range(int(spec.get("n_prompts") or 4)):
            stamps = []
            result = stream_one(port, model, {"prompt": "Continue counting: one, two, three,"}, steps,
                                chat, token_clock=stamps.append)
            if not result["fixed_length"] or result["token_count_mismatch"] or len(stamps) < 2:
                raise ValueError("fixed decode window unavailable")
            windows.append((stamps[0], stamps[-1]))
            decode_steps += result["tokens"] - 1
    finally:
        try:
            post("stop")
        finally:
            if sys.exc_info()[0] is not None:
                for path in trace_dir.rglob("*"):
                    if path not in existing and path.is_file() and (
                            path.name.endswith(".json") or path.name.endswith(".json.gz")):
                        path.unlink(missing_ok=True)
    return parse_profile_traces(trace_dir, existing, decode_steps, windows)


def parse_profile_traces(trace_dir, existing, decode_steps, windows):
    """Ignore frontend-only traces and remove only artifacts from this capture."""
    traces = sorted(p for p in trace_dir.rglob("*") if p not in existing and p.is_file()
                    and (p.name.endswith(".json.gz") or p.name.endswith(".json")))
    results = []
    try:
        for path in traces:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as f:
                trace = json.load(f)
            if not any(e.get("ph") == "X" and "kernel" in e.get("cat", "").lower()
                       for e in trace.get("traceEvents", [])):
                continue
            base = trace.get("baseTimeNanoseconds", 0) / 1000
            results.append(parse_trace(trace, decode_steps, [(a - base, b - base) for a, b in windows]))
        if len(results) != 1:
            raise ValueError("expected one GPU trace")
        return {**results[0], "trace_path": None, "trace_removed": True,
                "window_basis": "first-to-last streamed content per request"}
    finally:
        for path in traces:
            path.unlink(missing_ok=True)


def run_profile(spec, work, gpu):
    out = {"structure": {}, "time_by_category_pct": None, "idle_pct": None,
           "launches_per_step": None, "top_kernels": [], "trace_path": None, "model_files": []}
    key = hashlib.sha256(canon({"recipe": spec["recipe"], "gpu": gpu,
                               "steps": spec.get("steps", 64), "n": spec.get("n_prompts"),
                               "version": 2}).encode()).hexdigest()[:20]
    cache = Path(spec["ref_cache_dir"]) / f"arch-{key}.json"
    try:
        if cache.exists() and not gpu["uuid"].startswith("unknown-"):
            cached = json.loads(cache.read_text())
            if time.time() - cached.get("measured_at", 0) < float(spec.get("ref_max_age_s", 1800)):
                return {**cached, "cached": True}
        structure, files, available = inspect_profile_target(spec)
        out.update(structure=structure, model_files=files)
        if not available:
            raise RuntimeError("profiler unavailable")
        out.update(serve_and_measure(spec, None, None, None, "profile", work))
    except Exception:
        # Never expose server logs, config values or exception text through a fallback.
        out["reason"] = "Profiling unavailable; only locally readable model structure is reported."
    out.update(measured_at=time.time(), gpu=gpu)
    if out["time_by_category_pct"] is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    work = Path(spec["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    gpu = gpu_identity()
    if spec.get("mode") == "profile":
        print("BENCH_RESULT " + json.dumps({"ok": True, **run_profile(spec, work, gpu)}), flush=True)
        return
    out: dict = {"gpu": gpu}
    try:
        confirm_pct = confirmation_threshold(spec.get("confirm_pct"))
        prompts_hash = hashlib.sha256(Path(spec["prompts"]).read_bytes()).hexdigest()[:16]
        key = hashlib.sha256(canon({"gpu": gpu["uuid"], "recipe": spec["recipe"], "prompts": prompts_hash,
                                    "n": spec.get("n_prompts"), "wl": spec["workload"],
                                    "sandbox": spec.get("sandbox", True),
                                    "measurement_version": 2}).encode()).hexdigest()[:20]
        cache = Path(spec["ref_cache_dir"]) / f"{key}.json"
        out["ref_key"] = key
        cached = json.loads(cache.read_text()) if cache.exists() else None
        max_age = float(spec.get("ref_max_age_s", 1800))
        if (cached and not spec.get("fresh_reference")
                and same_gpu(cached.get("gpu") or {}, gpu)
                and time.time() - cached.get("measured_at", 0) < max_age):
            out["reference"] = cached
            out["reference_cached"] = True
        else:
            try:
                out["reference"] = serve_and_measure(spec, None, None, None, "ref", work)
            except Exception as e:
                out["infra"] = True     # the unmodified published recipe failed: not the agent's doing
                raise
            out["reference"]["measured_at"] = time.time()
            out["reference"]["gpu"] = gpu
            if cached:   # the previous reference on this GPU: a run-to-run noise sample for the agents
                out["previous_reference_value"] = cached.get("value")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(out["reference"]))
            out["reference_cached"] = False
        out["reference_age_s"] = max(0.0, time.time() - out["reference"]["measured_at"])
        out["reference_gpu"] = out["reference"]["gpu"]
        if spec.get("candidate") is not None:
            c = spec["candidate"]
            out["candidate"] = serve_and_measure(spec, c.get("bundle_dir"), c.get("config_delta"),
                                                 c.get("env"), "cand", work)
            r, v = out["reference"]["value"], out["candidate"]["value"]
            out.update(comparison(r, v))
            if out["reference_cached"] and r and abs(100.0 * (v - r) / r) >= confirm_pct:
                try:
                    fresh = serve_and_measure(spec, None, None, None, "ref-confirm", work)
                except Exception:
                    out["infra"] = True
                    raise
                fresh.update(measured_at=time.time(), gpu=gpu)
                out["reference_fresh"] = fresh
                out.update(comparison(r, v, fresh["value"]))
                cache.write_text(json.dumps(fresh))
        out["ok"] = True
    except Exception as e:  # report one line; the tool layer classifies it
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    print("BENCH_RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
