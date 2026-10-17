"""Executor: run commands, detached jobs and file transfers on a fleet machine (local, the rig over ssh, or a
Vast box over ssh). It implements the forge.tools.executor.Executor protocol, plus jobs and tree transfer.

    from forge.fleet.executor import executor_for
    ex = executor_for(machine)                 # machine dict or name (forge.fleet.machines)
    r = ex.run("nvidia-smi -L")                # ExecResult(rc, stdout, stderr, elapsed_s, infra_error)
    r = ex.run([py, "-m", "forge.tools.benchjob", "--spec", p], kind="bench", vram_gb=90, gpus=1,
               mission_id="m1", label="bench")  # waits for a scheduler grant on the target, then runs
    j = ex.start_job("serve-1", "python -m ...")  # detached; survives the ssh session
    ex.job_status(j) -> {"state": running|exited|lost, "rc", "pid"}; ex.read_log(j, offset) -> (text, offset)
    ex.put_tree(local_dir, remote_dir); ex.get_tree(remote_path, local_dir)

Every remote command is one `bash -c` script. GPU work (kind="bench"/"serve"/"gate" or gpus>0) is wrapped in
`python -m forge.fleet.gpujob` ON THE TARGET, which takes a grant from that machine's scheduler daemon
(forge.sched) and so holds a landlord lease. infra_error is set when the command could not run at all
(ssh down, scheduler down, grant impossible), never for the command's own failure.
"""
from __future__ import annotations

import io
import os
import posixpath
import shlex
import socket
import subprocess
import tarfile
import time
from pathlib import Path

from forge.tools.executor import ExecResult

INFRA_RC = 199            # gpujob's exit code for "could not run at all"; the reason is on stderr
INFRA_PREFIX = "FORGE_INFRA: "
GPU_KINDS = ("bench", "serve", "gate")


def _q(cmd: str | list[str]) -> str:
    return cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(c)) for c in cmd)


class FleetExecutor:
    """One machine. `machine` keys used: name, forge_root, ssh {host, user, port, key} (absent = local),
    env (base env for every command), python (default python on the target)."""

    def __init__(self, machine: dict):
        self.machine = machine
        self.name = machine["name"]
        self.root = machine.get("forge_root") or "~/forge"
        self.ssh = machine.get("ssh") or None
        if self.ssh and machine.get("hostname") and machine["hostname"] == socket.gethostname():
            self.ssh = None           # we are on that machine: run locally
        self.base_env = dict(machine.get("env") or {})
        self.python = machine.get("python") or "python3"

    # ------------------------------------------------------------------ transport
    def _ssh_argv(self) -> list[str]:
        s = self.ssh
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=4", "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR"]
        key = s.get("key") or os.environ.get("FORGE_SSH_KEY")
        if not key:
            default = Path.home() / ".ssh" / "id_ed25519_forge"
            key = str(default) if default.exists() else None
        if key:
            argv += ["-i", os.path.expanduser(key), "-o", "IdentitiesOnly=yes"]
        if s.get("port"):
            argv += ["-p", str(s["port"])]
        if s.get("jump"):
            argv += ["-J", s["jump"]]
        argv.append(f"{s.get('user', 'root')}@{s['host']}")
        return argv

    def _script(self, cmd: str | list[str], cwd: str | None, env: dict | None, timeout_s: float | None) -> str:
        lines = []
        for k, v in {**self.base_env, **(env or {})}.items():
            lines.append(f"export {k}={shlex.quote(str(v))}")
        if cwd:
            lines.append(f"cd {self._path(cwd)} || exit 127")
        body = _q(cmd)
        if timeout_s:
            body = f"timeout -k 15 {int(timeout_s)} bash -c {shlex.quote(body)}"
        lines.append(body)
        return "\n".join(lines)

    def _path(self, p: str) -> str:
        """Quote a target path, keeping a leading ~ expandable."""
        if p == "~" or p.startswith("~/"):
            return '"$HOME"' + (shlex.quote(p[1:]) if len(p) > 1 else "")
        return shlex.quote(p)

    def _exec(self, script: str, stdin: bytes | None = None, timeout: float | None = None) -> tuple[int, bytes, bytes, float, str | None]:
        argv = (self._ssh_argv() + ["bash -c " + shlex.quote(script)]) if self.ssh else ["bash", "-c", script]
        t0 = time.time()
        try:
            p = subprocess.run(argv, input=stdin if stdin is not None else b"", capture_output=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or b"", (e.stderr or b"") + f"\ntimed out after {timeout:.0f}s".encode(), time.time() - t0, None
        except OSError as e:
            return 126, b"", b"", time.time() - t0, f"{type(e).__name__}: {e}"
        infra = None
        if self.ssh and p.returncode == 255:
            infra = f"ssh to {self.name} failed: {p.stderr.decode(errors='replace').strip()[-300:]}"
        return p.returncode, p.stdout, p.stderr, time.time() - t0, infra

    # ------------------------------------------------------------------ protocol
    def run(self, cmd, *, cwd=None, env=None, timeout_s=600, kind="work", vram_gb=None, gpus=0,
            mission_id="", label="", stdin: bytes | None = None) -> ExecResult:
        if kind in GPU_KINDS or gpus:
            wrap = [self.machine.get("sched_python") or self.python, "-m", "forge.fleet.gpujob",
                    "--kind", kind if kind in GPU_KINDS else "bench", "--gpus", str(gpus or 1),
                    "--minutes", str(int(timeout_s / 60) + 10), "--mission", mission_id or "-",
                    "--label", label or kind, "--timeout", str(int(timeout_s))]
            if vram_gb is not None:
                wrap += ["--vram", str(vram_gb)]
            cmd = _q(wrap) + " -- bash -c " + shlex.quote(_q(cmd))
            env = {"PYTHONPATH": self.root, "FORGE_ROOT": self.root, **(env or {})}
            # the grant can queue: allow an extra hour of waiting on top of the job's own time
            script = self._script(cmd, cwd, env, None)
            rc, out, err, dt, infra = self._exec(script, stdin, timeout=timeout_s + 3600)
        else:
            script = self._script(cmd, cwd, env, timeout_s)
            rc, out, err, dt, infra = self._exec(script, stdin, timeout=timeout_s + 60)
        so, se = out.decode(errors="replace"), err.decode(errors="replace")
        if rc == INFRA_RC and not infra:
            line = next((l for l in reversed(se.splitlines()) if l.startswith(INFRA_PREFIX)), None)
            if line:
                infra = line[len(INFRA_PREFIX):]
        return ExecResult(rc, so, se, dt, infra_error=infra)

    def sh(self, cmd, timeout_s=120, **kw) -> ExecResult:
        """run() that raises on infra errors (for fleet-internal use)."""
        r = self.run(cmd, timeout_s=timeout_s, **kw)
        if r.infra_error:
            raise ExecutorError(r.infra_error)
        return r

    def read_text(self, path: str) -> str:
        r = self.sh(f"cat {self._path(path)}")
        if r.rc != 0:
            raise FileNotFoundError(f"{self.name}:{path}: {r.stderr.strip()[-200:]}")
        return r.stdout

    def read_bytes(self, path: str) -> bytes:
        rc, out, err, _, infra = self._exec(f"cat {self._path(path)}", timeout=600)
        if infra:
            raise ExecutorError(infra)
        if rc != 0:
            raise FileNotFoundError(f"{self.name}:{path}: {err.decode(errors='replace').strip()[-200:]}")
        return out

    def write_text(self, path: str, text: str) -> None:
        self.write_bytes(path, text.encode("utf-8"))

    def write_bytes(self, path: str, data: bytes, mode: str | None = None) -> None:
        d = posixpath.dirname(path) or "."
        tmp = path + ".tmp-forge"
        script = f"mkdir -p {self._path(d)} && cat > {self._path(tmp)} && mv -f {self._path(tmp)} {self._path(path)}"
        if mode:
            script += f" && chmod {mode} {self._path(path)}"
        rc, _, err, _, infra = self._exec(script, stdin=data, timeout=600)
        if infra:
            raise ExecutorError(infra)
        if rc != 0:
            raise OSError(f"{self.name}:{path}: write failed: {err.decode(errors='replace').strip()[-200:]}")

    def exists(self, path: str) -> bool:
        return self.sh(f"test -e {self._path(path)}").rc == 0

    def list_files(self, root: str) -> list[str]:
        r = self.sh(f"test -d {self._path(root)} && find {self._path(root)} -type f -printf '%P\\n' | sort")
        return [l for l in r.stdout.splitlines() if l] if r.rc == 0 else []

    def listdir(self, path: str) -> list[str]:
        r = self.sh(f"test -d {self._path(path)} && ls -A1 {self._path(path)}")
        return sorted(l for l in r.stdout.splitlines() if l) if r.rc == 0 else []

    def remove(self, path: str) -> None:
        self.sh(f"rm -f -- {self._path(path)}")

    def home(self) -> str:
        if not hasattr(self, "_home"):
            self._home = self.sh("echo $HOME").stdout.strip()
        return self._home

    def abs(self, path: str) -> str:
        return self.home() + path[1:] if path.startswith("~") else path

    def workspace(self, mission_id: str) -> str:
        p = posixpath.join(self.abs(self.root), "state", "missions", mission_id, "ws")
        self.sh(f"mkdir -p {shlex.quote(p)}")
        return p

    # ------------------------------------------------------------------ jobs (detached)
    def _job_dir(self, job_id: str) -> str:
        return posixpath.join(self.abs(self.root), "state", "jobs", job_id)

    def start_job(self, job_id: str, cmd, *, cwd=None, env=None) -> str:
        """Start `cmd` detached (setsid nohup) and return job_id. Its process group is its own."""
        d = self._job_dir(job_id)
        body = self._script(cmd, cwd, env, None)
        run_sh = f"{body}\n"
        wrapper = (f"#!/bin/bash\n( {run_sh} )\necho $? > {shlex.quote(d)}/rc.tmp && mv {shlex.quote(d)}/rc.tmp {shlex.quote(d)}/rc\n")
        self.write_bytes(posixpath.join(d, "run.sh"), wrapper.encode(), mode="755")
        r = self.sh(f"cd {shlex.quote(d)} && rm -f rc && setsid nohup bash run.sh > log 2>&1 < /dev/null & "
                    f"echo $! > {shlex.quote(d)}/pid; cat {shlex.quote(d)}/pid")
        if r.rc != 0:
            raise ExecutorError(f"start_job {job_id} on {self.name}: {r.stderr.strip()[-200:]}")
        return job_id

    def job_status(self, job_id: str) -> dict:
        d = shlex.quote(self._job_dir(job_id))
        r = self.sh(f"cat {d}/pid 2>/dev/null; echo; cat {d}/rc 2>/dev/null; echo; "
                    f"p=$(cat {d}/pid 2>/dev/null); [ -n \"$p\" ] && kill -0 $p 2>/dev/null && echo alive")
        parts = r.stdout.split("\n")
        pid = int(parts[0]) if parts and parts[0].strip().isdigit() else None
        rc = int(parts[1]) if len(parts) > 1 and parts[1].strip().lstrip("-").isdigit() else None
        alive = "alive" in r.stdout
        state = "exited" if rc is not None else ("running" if alive else ("lost" if pid else "unknown"))
        return {"job_id": job_id, "state": state, "rc": rc, "pid": pid}

    def read_log(self, job_id: str, offset: int = 0, max_bytes: int = 1 << 20) -> tuple[str, int]:
        """New log text from byte `offset`; returns (text, new_offset). Poll this to stream a log."""
        log = shlex.quote(posixpath.join(self._job_dir(job_id), "log"))
        rc, out, _, _, infra = self._exec(f"tail -c +{int(offset) + 1} {log} 2>/dev/null | head -c {int(max_bytes)}",
                                          timeout=120)
        if infra:
            raise ExecutorError(infra)
        return out.decode(errors="replace"), offset + len(out)

    def stream_log(self, job_id: str, follow: bool = True, poll_s: float = 2.0):
        """Yield log chunks until the job exits (follow=True) or once (follow=False)."""
        off = 0
        while True:
            text, off = self.read_log(job_id, off)
            if text:
                yield text
            if not follow:
                return
            if self.job_status(job_id)["state"] != "running":
                text, off = self.read_log(job_id, off)
                if text:
                    yield text
                return
            time.sleep(poll_s)

    def tail(self, job_id: str, n: int = 100) -> str:
        return self.sh(f"tail -n {int(n)} {shlex.quote(posixpath.join(self._job_dir(job_id), 'log'))} 2>/dev/null").stdout

    def stop_job(self, job_id: str, grace_s: int = 20) -> dict:
        """Stop a job this executor started: TERM its process group, KILL after grace_s."""
        d = shlex.quote(self._job_dir(job_id))
        self.sh(f"p=$(cat {d}/pid 2>/dev/null) || exit 0; kill -TERM -$p 2>/dev/null; "
                f"for i in $(seq {int(grace_s)}); do kill -0 $p 2>/dev/null || exit 0; sleep 1; done; kill -KILL -$p 2>/dev/null; true",
                timeout_s=grace_s + 30)
        return self.job_status(job_id)

    # ------------------------------------------------------------------ transfer
    def put_tree(self, local_dir: str | Path, remote_dir: str, excludes: tuple[str, ...] = ("__pycache__", ".git", "state", "*.pyc")) -> int:
        """Copy local_dir's contents into remote_dir (tar over the transport). Returns bytes sent."""
        data = tar_bytes(Path(local_dir), excludes)
        rc, _, err, _, infra = self._exec(f"mkdir -p {self._path(remote_dir)} && tar xzf - -C {self._path(remote_dir)}",
                                          stdin=data, timeout=1800)
        if infra:
            raise ExecutorError(infra)
        if rc != 0:
            raise OSError(f"put_tree to {self.name}:{remote_dir}: {err.decode(errors='replace').strip()[-300:]}")
        return len(data)

    def get_tree(self, remote_path: str, local_dir: str | Path) -> list[str]:
        """Copy remote_path (file or dir) into local_dir/<basename>. Returns the member names."""
        parent, base = posixpath.split(remote_path.rstrip("/"))
        rc, out, err, _, infra = self._exec(f"tar czf - -C {self._path(parent or '/')} {shlex.quote(base)}", timeout=3600)
        if infra:
            raise ExecutorError(infra)
        if rc != 0:
            raise FileNotFoundError(f"get_tree {self.name}:{remote_path}: {err.decode(errors='replace').strip()[-300:]}")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(out), mode="r:gz") as tf:
            names = tf.getnames()
            for m in tf.getmembers():
                if m.name.startswith(("/", "..")) or "/../" in m.name or m.issym() or m.islnk():
                    raise OSError(f"refusing unsafe tar member {m.name}")
            tf.extractall(local_dir)
        return names


    def stat_tree(self, root: str) -> dict[str, tuple[int, float]]:
        """{relative path: (size, mtime)} for files under root ({} if missing)."""
        r = self.sh(f"test -d {self._path(root)} && find {self._path(root)} -type f -printf '%s %T@ %P\n'", timeout_s=120)
        out = {}
        for line in r.stdout.splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3:
                out[parts[2]] = (int(parts[0]), float(parts[1]))
        return out

    def get_files(self, root: str, rels: list[str], local_dir: str | Path) -> int:
        """Copy the listed files (relative to root) into local_dir, keeping their relative paths."""
        if not rels:
            return 0
        listing = ("\n".join(rels) + "\n").encode()
        rc, out, err, _, infra = self._exec(f"cd {self._path(root)} && tar czf - -T -", stdin=listing, timeout=3600)
        if infra:
            raise ExecutorError(infra)
        if rc not in (0, 1):     # 1: a file changed while being read; it is fetched again next round
            raise OSError(f"get_files {self.name}:{root}: {err.decode(errors='replace').strip()[-300:]}")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(out), mode="r:gz") as tf:
            ms = [m for m in tf.getmembers() if m.isfile() and not m.name.startswith(("/", "..")) and "/../" not in m.name]
            tf.extractall(local_dir, members=ms)
        return len(ms)


class ExecutorError(RuntimeError):
    """The machine could not be reached or the operation could not run (infrastructure)."""


def tar_bytes(src: Path, excludes: tuple[str, ...] = ()) -> bytes:
    import fnmatch
    buf = io.BytesIO()

    def skip(rel: str) -> bool:
        return any(fnmatch.fnmatch(part, pat) for part in rel.split("/") for pat in excludes)

    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src).as_posix()
            if skip(rel) or not p.is_file():
                continue
            info = tf.gettarinfo(str(p), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if rel.startswith("bin/") or rel.endswith(".sh"):
                info.mode = 0o755
            else:
                info.mode = 0o644
            with open(p, "rb") as f:
                data = f.read()
            if rel.endswith((".py", ".sh", ".yaml", ".md", ".json", ".jsonl", ".txt")) or rel.startswith("bin/"):
                data = data.replace(b"\r\n", b"\n")      # code runs on Linux
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def executor_for(machine) -> FleetExecutor:
    """machine: a machine dict, or a name looked up in forge.fleet.machines."""
    if isinstance(machine, str):
        from forge.fleet.machines import get_machine
        machine = get_machine(machine)
    return FleetExecutor(machine)
