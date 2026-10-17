"""Gate-started vLLM servers (one private sandbox scope each) and the timing client."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request

from forge.sandbox import Spec, launch


class ServerError(RuntimeError):
    pass


def http_json(url: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:600]
        raise ServerError(f"HTTP {e.code} from {url.rsplit('/', 2)[-1]}: {body}") from None


class GateServer:
    """One vLLM server started by the gate.

    argv/env come from forge.tools.bundle.serve_command. Writable: the job's private home/tmp/ws and
    `cache_dir` (all caches point there). Read-only: venv, HF cache, bundle copy, hook dir.
    """

    def __init__(self, name: str, argv: list[str], env: dict, job_dir: str, cache_dir: str, ro: list[str],
                 gpus: list[int], ports: tuple[int, int], cpus: list[int], lease_id: str | None,
                 extra_files: dict[str, str] | None = None, start_timeout: float = 1200.0):
        self.name = name
        self.argv = argv
        self.env = env
        self.job_dir = job_dir
        self.cache_dir = cache_dir
        self.ro = ro
        self.gpus = gpus
        self.ports = ports
        self.cpus = cpus
        self.lease_id = lease_id
        self.extra_files = extra_files or {}
        self.start_timeout = start_timeout
        self.port = int(argv[argv.index("--port") + 1])
        self.base = f"http://127.0.0.1:{self.port}"
        self.job = None
        self.started_s = None

    def start(self) -> float:
        os.makedirs(self.cache_dir, mode=0o700, exist_ok=True)
        ws = os.path.join(self.job_dir, "ws")
        os.makedirs(ws, mode=0o700, exist_ok=True)
        for dest, src in self.extra_files.items():
            shutil.copyfile(src, os.path.join(ws, dest))
        c = self.cache_dir
        env = {**self.env,
               "VLLM_CACHE_ROOT": os.path.join(c, "vllm"), "TORCHINDUCTOR_CACHE_DIR": os.path.join(c, "inductor"),
               "TRITON_CACHE_DIR": os.path.join(c, "triton"), "CUDA_CACHE_PATH": os.path.join(c, "nv"),
               "FLASHINFER_WORKSPACE_BASE": os.path.join(c, "flashinfer"), "XDG_CACHE_HOME": os.path.join(c, "xdg"),
               "VLLM_PORT": str(self.ports[0] + 1), "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
        if self.lease_id:
            env["LANDLORD_LEASE_ID"] = self.lease_id
        extra_path = [p for p in env.pop("PATH", "").split(":") if p]
        spec = Spec(cmd=self.argv, job_dir=self.job_dir, ro=self.ro, rw=[self.cache_dir], gpus=self.gpus,
                    ports=self.ports, cpus=self.cpus, env=env,
                    path_prepend=[os.path.dirname(self.argv[0])] + extra_path, cwd=ws, name=self.name)
        t0 = time.time()
        self.job = launch(spec)
        end = t0 + self.start_timeout
        while time.time() < end:
            if not self.job.alive():
                raise ServerError(f"server exited during startup (rc={self.job.returncode})")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                    if r.status == 200:
                        self.started_s = time.time() - t0
                        return self.started_s
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        self.stop()
        raise ServerError(f"server not healthy after {self.start_timeout:.0f}s")

    def model_name(self) -> str:
        d = http_json(self.base + "/v1/models", timeout=30)
        return d["data"][0]["id"]

    def complete(self, model: str, prompt_ids: list[int], max_tokens: int, logprobs: int | None = None,
                 prompt_logprobs: int | None = None, timeout: float = 900.0,
                 ignore_eos: bool = False) -> tuple[dict, float]:
        body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0.0,
                "seed": 0, "return_tokens_as_token_ids": True, "skip_special_tokens": False}
        if ignore_eos:
            body["ignore_eos"] = True
        if logprobs is not None:
            body["logprobs"] = logprobs
        if prompt_logprobs is not None:
            body["prompt_logprobs"] = prompt_logprobs
        t0 = time.perf_counter()
        r = http_json(self.base + "/v1/completions", body, timeout=timeout)
        return r, time.perf_counter() - t0

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop(grace=20)

    def log_tail(self, n: int = 256 * 1024) -> str:
        return self.job.tail(n) if self.job else ""

    def cleanup(self) -> None:
        if self.job is not None:
            self.job.cleanup(keep_log=True)


class GpuWatch:
    """Samples the gate GPUs during a timed pass for foreign compute processes."""

    def __init__(self, uuids: list[str], own_unit: str, period: float = 2.0):
        self.uuids = set(u for u in uuids if u)
        self.own_unit = own_unit
        self.period = period
        self.foreign: list[dict] = []
        self._stop = threading.Event()
        self._t = None

    def check_once(self) -> list[dict]:
        import subprocess
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        bad = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or parts[1] not in self.uuids:
                continue
            pid = int(parts[0])
            try:
                with open(f"/proc/{pid}/cgroup") as f:
                    cg = f.read()
            except OSError:
                cg = ""
            if self.own_unit and self.own_unit in cg:
                continue
            bad.append({"pid": pid, "gpu": parts[1], "mib": parts[2]})
        return bad

    def _run(self):
        while not self._stop.wait(self.period):
            self.foreign += self.check_once()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=30)
