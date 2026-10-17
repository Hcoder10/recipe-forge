"""Sandbox entry point, run inside the job's systemd scope:

    python -m forge.sandbox.enter <spec.json>

Pins CPUs, applies Landlock, switches to the job's cwd and env, then execs the command.
The spec is written by forge.sandbox.launch(); nothing here reads the parent's environment.
"""
import json
import os
import shutil
import sys

from forge.sandbox import landlock


def main(spec_path: str, containment: str = "systemd") -> None:
    with open(spec_path) as f:
        spec = json.load(f)
    env = spec["env"]
    if spec.get("cpus"):
        os.sched_setaffinity(0, spec["cpus"])
    os.umask(0o077)
    # Open before restriction: the job directory itself is deliberately not writable.
    level_path = os.path.join(os.path.dirname(spec_path), "sandbox_level.json")
    with open(level_path, "w") as level_file:
        abi = landlock.apply(
            [(p, r) for p, r in spec["paths"]], spec.get("bind_ports"), spec.get("connect_ports"),
            scope=spec.get("scope", True), min_abi=spec.get("min_abi", 1),
            allow_none=spec.get("allow_none", False))
        json.dump({"landlock_abi": abi, "fs_rules": abi >= 1,
                   "net_rules": abi >= 4 and (spec.get("bind_ports") is not None or
                                               spec.get("connect_ports") is not None),
                   "containment": containment}, level_file)
    os.chdir(spec["cwd"])
    cmd = spec["cmd"]
    exe = cmd[0] if os.sep in cmd[0] else shutil.which(cmd[0], path=env.get("PATH", ""))
    if not exe:
        sys.stderr.write(f"forge.sandbox: command not found: {cmd[0]}\n")
        os._exit(127)
    os.execve(exe, cmd, env)


def run(spec_path: str, containment: str = "systemd") -> None:
    try:
        main(spec_path, containment)
    except Exception as e:  # report and fail closed: never exec unsandboxed
        sys.stderr.write(f"forge.sandbox: refused to start: {type(e).__name__}: {e}\n")
        os._exit(126)


if __name__ == "__main__":
    run(sys.argv[1])
