"""Run a Landlock job in a systemd user scope, or a supervised process group.

FORGE_SANDBOX_NO_SYSTEMD=1 forces the fallback. Otherwise availability is cached
per process. The fallback adopts orphaned descendants and tracks /proc identities
and a job environment token; it does not provide systemd's MemoryMax accounting.
FORGE_SANDBOX_MIN_ABI defaults to 1. Unsupported Landlock rights are omitted;
FORGE_SANDBOX_ALLOW_NONE=1 explicitly permits kernels without Landlock.
job.sandbox_level and <job_dir>/sandbox_level.json report the applied restrictions.

    from forge.sandbox import Spec, launch
    job = launch(Spec(cmd=["python", "x.py"], job_dir="/home/cube/forge/jobs/j1",
                      ro=["/home/cube/venvs/vllm-bolt"], rw=[], gpus=[2], ports=(20100, 20199),
                      cpus=list(range(8, 32)), env={"LANDLORD_LEASE_ID": "apt_..."}))
    rc = job.wait(timeout=600)      # reaps the whole scope, setsid children included
    print(job.tail())

Layout per job: <job_dir>/{home,tmp,ws} are private and writable, <job_dir>/log is the combined
stdout/stderr, <job_dir>/spec.json the exact sandbox applied. Read-only paths must not sit inside
writable ones (Landlock rights are inherited downward, so they would be writable); launch() refuses.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

from forge.sandbox import landlock

SYSTEM_RO = ["/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc", "/opt", "/sys",
             "/run/systemd/resolve", "/var/lib/dpkg/info"]
DEV_FILES = ["/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty"]
GPU_SHARED_NODES = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"]
BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_systemd_available: bool | None = None
_systemd_lock = threading.Lock()


def _use_systemd() -> bool:
    global _systemd_available
    if os.environ.get("FORGE_SANDBOX_NO_SYSTEMD") == "1":
        return False
    with _systemd_lock:
        if _systemd_available is None:
            try:
                result = subprocess.run(
                    ["systemd-run", "--user", "--scope", "--quiet", "--collect", "/bin/true"],
                    env=_user_bus_env(), capture_output=True, timeout=10)
                _systemd_available = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                _systemd_available = False
        return _systemd_available


@dataclasses.dataclass
class Spec:
    cmd: list[str]
    job_dir: str
    ro: list[str] = dataclasses.field(default_factory=list)      # extra read-only paths
    rw: list[str] = dataclasses.field(default_factory=list)      # extra writable paths
    gpus: list[int] = dataclasses.field(default_factory=list)    # physical nvidia-smi indexes
    ports: tuple[int, int] | None = None                         # inclusive TCP range for bind+connect
    connect_extra: list[int] = dataclasses.field(default_factory=list)
    connect_any: bool = False                                    # leave outbound TCP unrestricted
    cpus: list[int] = dataclasses.field(default_factory=list)
    env: dict[str, str] = dataclasses.field(default_factory=dict)  # added on top of the clean base env
    path_prepend: list[str] = dataclasses.field(default_factory=list)  # e.g. a venv's bin dir
    cwd: str | None = None                                       # default <job_dir>/ws
    shm: bool = True                                             # allow /dev/shm (vLLM/torch need it)
    mem_max: str | None = None                                   # systemd MemoryMax, e.g. "64G"
    name: str | None = None                                      # unit name suffix; default random


class SandboxError(RuntimeError):
    pass


def _norm(p: str) -> str:
    return os.path.realpath(os.path.expanduser(p))


def _inside(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def gpu_nodes(indexes: list[int]) -> list[str]:
    """Map visible nvidia-smi indexes to /dev/nvidiaN through PCI bus ids and the driver's minor numbers.

    In containers /proc/driver/nvidia/gpus lists every host GPU, so list position is not the index."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=10).stdout
        buses = {}
        for line in out.splitlines():
            index, pci = (p.strip() for p in line.split(",", 1))
            domain, bus, slot = pci.lower().split(":")
            buses[int(index)] = f"{int(domain, 16):04x}:{bus}:{slot}"
        nodes = []
        for idx in indexes:
            if idx not in buses:
                raise SandboxError(f"no visible GPU with index {idx}")
            with open(f"/proc/driver/nvidia/gpus/{buses[idx]}/information") as f:
                minors = [int(line.split(":", 1)[1]) for line in f if line.startswith("Device Minor")]
            if len(minors) != 1:
                raise SandboxError(f"cannot resolve device minor for GPU {idx}")
            node = f"/dev/nvidia{minors[0]}"
            if not os.path.exists(node):
                raise SandboxError(f"GPU {idx} device node {node} is not present")
            nodes.append(node)
        return nodes
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        raise SandboxError(f"cannot resolve visible GPU device nodes: {e}") from e


def build_rules(spec: Spec) -> tuple[list[tuple[str, int]], dict]:
    jd = _norm(spec.job_dir)
    rw = [os.path.join(jd, d) for d in ("home", "tmp", "ws")] + [_norm(p) for p in spec.rw]
    ro = [_norm(p) for p in spec.ro]
    for r in ro:
        for w in rw:
            if _inside(r, w):
                raise SandboxError(f"read-only path {r} is inside writable {w}; it would be writable")
    rules: list[tuple[str, int]] = [(p, landlock.RO) for p in SYSTEM_RO + ro]
    rules += [(p, landlock.RW) for p in rw]
    # CUDA names its threads via /proc/self/task/*/comm (O_WRONLY|O_TRUNC) and fails init without it.
    # Writes to other processes' /proc entries still hit Landlock's ptrace scoping.
    rules.append(("/proc", landlock.RO | landlock.WRITE_FILE | landlock.TRUNCATE))
    rules += [(p, landlock.DEV_RW) for p in DEV_FILES]
    if spec.shm:
        rules.append(("/dev/shm", landlock.RW))
    if spec.gpus:
        rules += [(p, landlock.DEV_RW) for p in GPU_SHARED_NODES + gpu_nodes(spec.gpus)]
        rules.append(("/dev/nvidia-caps", landlock.RO))
    net: dict = {"bind_ports": None, "connect_ports": None}
    if spec.ports:
        lo, hi = spec.ports
        rng = list(range(lo, hi + 1))
        net["bind_ports"] = rng + [0]
        net["connect_ports"] = None if spec.connect_any else rng + list(spec.connect_extra)
    elif not spec.connect_any:
        net["bind_ports"] = []
        net["connect_ports"] = list(spec.connect_extra)
    return rules, net


def build_env(spec: Spec) -> dict[str, str]:
    jd = _norm(spec.job_dir)
    home, tmp = os.path.join(jd, "home"), os.path.join(jd, "tmp")
    env = {
        "PATH": ":".join(spec.path_prepend + [BASE_PATH]),
        "HOME": home, "TMPDIR": tmp, "TMP": tmp, "TEMP": tmp,
        "XDG_CACHE_HOME": os.path.join(home, ".cache"), "XDG_CONFIG_HOME": os.path.join(home, ".config"),
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONUNBUFFERED": "1", "USER": os.environ.get("USER", "cube"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(home, ".cache", "inductor"),
        "TRITON_CACHE_DIR": os.path.join(home, ".cache", "triton"),
    }
    if spec.gpus:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(spec.gpus)))
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if spec.ports:
        env["FORGE_PORT_LO"], env["FORGE_PORT_HI"] = str(spec.ports[0]), str(spec.ports[1])
    env.update({k: str(v) for k, v in spec.env.items()})
    return env


class Job:
    def __init__(self, unit: str, job_dir: str, proc: subprocess.Popen, log_path: str):
        self.unit = unit
        self.job_dir = job_dir
        self.proc = proc
        self.log_path = log_path
        self.pid = proc.pid
        self.started = time.time()
        self.returncode: int | None = None
        self.containment = "systemd-scope"
        self.sandbox_level: dict | None = None
        self.survivors: list[int] = []
        self._report_path: str | None = None

    def poll(self) -> int | None:
        rc = self.proc.poll()
        if rc is not None and self.returncode is None:
            self.returncode = rc
            self._reap_scope()
        return rc

    def alive(self) -> bool:
        return self.poll() is None

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the main process; on timeout stop the whole scope. Returns the exit code."""
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.stop()
            rc = self.proc.wait()
        self.returncode = rc
        self._reap_scope()
        return rc

    def stop(self, grace: float = 15.0) -> None:
        """SIGTERM every process in the scope, then SIGKILL after `grace` seconds."""
        if self.containment == "pgroup-fallback":
            if self.proc.poll() is None:
                with open(self._report_path + ".grace", "w") as f:
                    f.write(str(grace))
                self.proc.terminate()
            self.returncode = self.proc.wait()
            self._reap_scope()
            return
        _systemctl("kill", "--signal=SIGTERM", self.unit + ".scope")
        end = time.time() + grace
        while time.time() < end and (self.proc.poll() is None or _scope_active(self.unit)):
            time.sleep(0.25)
        self._reap_scope()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.returncode = self.proc.returncode

    def kill(self, grace: float = 15.0) -> list[int]:
        """Stop the job and return any surviving processes."""
        self.stop(grace)
        return list(self.survivors)

    def _reap_scope(self) -> None:
        if self.containment == "pgroup-fallback":
            with open(self._report_path) as f:
                self.survivors = json.load(f)["survivors"]
            return
        if _scope_active(self.unit):
            _systemctl("kill", "--signal=SIGKILL", self.unit + ".scope")
            for _ in range(40):
                if not _scope_active(self.unit):
                    break
                time.sleep(0.1)
        _systemctl("stop", self.unit + ".scope")

    def tail(self, n_bytes: int = 4000) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - n_bytes))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def cleanup(self, keep_log: bool = True) -> None:
        """Delete the job's private dirs (home, tmp, ws). The log stays unless keep_log=False."""
        for d in ("home", "tmp", "ws"):
            shutil.rmtree(os.path.join(self.job_dir, d), ignore_errors=True)
        if not keep_log:
            shutil.rmtree(self.job_dir, ignore_errors=True)


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, env=_user_bus_env())


def _scope_active(unit: str) -> bool:
    r = _systemctl("is-active", unit + ".scope")
    return r.stdout.strip() in ("active", "activating", "deactivating")


def _user_bus_env() -> dict[str, str]:
    uid = os.getuid()
    rt = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"
    return {"PATH": BASE_PATH, "HOME": os.path.expanduser("~"), "XDG_RUNTIME_DIR": rt,
            "DBUS_SESSION_BUS_ADDRESS": os.environ.get("DBUS_SESSION_BUS_ADDRESS", f"unix:path={rt}/bus")}


def launch(spec: Spec) -> Job:
    """Start spec.cmd sandboxed with automatic containment selection."""
    global _systemd_available
    min_abi = int(os.environ.get("FORGE_SANDBOX_MIN_ABI", "1"))
    allow_none = os.environ.get("FORGE_SANDBOX_ALLOW_NONE") == "1"
    try:
        landlock.checked_abi(min_abi, allow_none)
    except landlock.LandlockUnavailable as exc:
        raise SandboxError(str(exc)) from exc
    jd = _norm(spec.job_dir)
    for d in ("home", "tmp", "ws"):
        os.makedirs(os.path.join(jd, d), mode=0o700, exist_ok=True)
    rules, net = build_rules(spec)
    env = build_env(spec)
    env["FORGE_JOB_TOKEN"] = uuid.uuid4().hex
    unit = "forge-" + (spec.name or uuid.uuid4().hex[:12])
    payload = {"cmd": spec.cmd, "env": env, "cwd": _norm(spec.cwd) if spec.cwd else os.path.join(jd, "ws"),
               "cpus": spec.cpus, "paths": rules, **net, "scope": True, "unit": unit,
               "min_abi": min_abi, "allow_none": allow_none}
    level_path = os.path.join(jd, "sandbox_level.json")
    if os.path.exists(level_path):
        os.unlink(level_path)
    spec_path = os.path.join(jd, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(payload, f, indent=1)
    os.chmod(spec_path, 0o600)
    log_path = os.path.join(jd, "log")
    pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    entry = [sys.executable, "-I", "-c",
             f"import sys; sys.path.insert(0, {pkg_parent!r}); from forge.sandbox.enter import run; "
             f"run({spec_path!r}, containment='pgroup')"]
    use_systemd = _use_systemd()
    # A readiness file distinguishes a scope launch failure from a job failure.
    ready = os.path.join(jd, ".started-" + uuid.uuid4().hex)
    argv = ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", unit]
    if spec.mem_max:
        argv += ["-p", f"MemoryMax={spec.mem_max}"]
    argv += [sys.executable, "-I", "-c",
             f"import sys; sys.path.insert(0, {pkg_parent!r}); from forge.sandbox.enter import run; "
             f"open({ready!r}, 'w').close(); "
             f"run({spec_path!r})"]
    penv = _user_bus_env()
    report_path = os.path.join(jd, ".processes-" + uuid.uuid4().hex + ".json")
    with open(log_path, "ab") as log:
        if use_systemd:
            try:
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        env=penv, start_new_session=True)
            except OSError:
                use_systemd = False
            else:
                deadline = time.monotonic() + 15
                while not os.path.exists(ready) and proc.poll() is None:
                    if time.monotonic() >= deadline:
                        # An ambiguous launch must not execute the command twice.
                        Job(unit, jd, proc, log_path).stop(grace=0)
                        raise SandboxError("systemd scope startup timed out")
                    time.sleep(0.01)
                use_systemd = os.path.exists(ready)
                if use_systemd:
                    os.unlink(ready)
            if not use_systemd:
                with _systemd_lock:
                    _systemd_available = False
        if not use_systemd:
            argv = [sys.executable, "-I", "-c",
                    f"import sys; sys.path.insert(0, {pkg_parent!r}); "
                    f"from forge.sandbox.supervisor import run; run({spec_path!r}, {entry!r}, {report_path!r})"]
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    env=penv, start_new_session=True)
            # Wait until the supervisor has installed its signal handler and recorded its group.
            while not os.path.exists(report_path):
                if proc.poll() is not None:
                    raise SandboxError("fallback supervisor failed to start; see " + log_path)
                time.sleep(0.01)
    job = Job(unit, jd, proc, log_path)
    if not use_systemd:
        job.containment = "pgroup-fallback"
        job._report_path = report_path
    deadline = time.monotonic() + 15
    while True:
        try:
            with open(level_path) as f:
                job.sandbox_level = json.load(f)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if proc.poll() is not None or time.monotonic() >= deadline:
                job.stop(grace=0)
                raise SandboxError("sandbox entry failed to report its level; see " + log_path)
            time.sleep(0.01)
    return job
