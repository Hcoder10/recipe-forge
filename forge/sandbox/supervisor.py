"""Linux fallback supervisor. Remains outside Landlock so it can reap descendants.

This is best-effort process containment, not a cgroup resource controller. PID
identities include start times to avoid signalling a recycled PID.
"""
import ctypes
import json
import os
import signal
import subprocess
import time


def snapshot():
    result = {}
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        try:
            with open(f'/proc/{name}/stat') as f:
                fields = f.read().rsplit(')', 1)[1].split()
            result[int(name)] = (int(fields[1]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return result


def run(spec_path, argv, report_path):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'prctl(CHILD_SUBREAPER)')
    with open(spec_path) as f:
        token = json.load(f)['env']['FORGE_JOB_TOKEN']
    marker = ('FORGE_JOB_TOKEN=' + token).encode()
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    child = subprocess.Popen(argv, start_new_session=True)
    pgid = child.pid
    seen = {}

    def scan(include_zombies=False):
        table = snapshot()
        owned = {os.getpid()}
        owned.update(pid for pid, start in seen.items()
                     if pid in table and table[pid][1] == start)
        for pid in table:
            try:
                with open(f'/proc/{pid}/environ', 'rb') as f:
                    if marker in f.read().split(b'\0'):
                        owned.add(pid)
            except OSError:
                pass
        while True:
            added = {pid for pid, (parent, _, _) in table.items() if parent in owned} - owned
            if not added:
                break
            owned.update(added)
        owned.discard(os.getpid())
        for pid in owned:
            if pid in table:
                seen[pid] = table[pid][1]
        return {pid for pid in owned if pid in table and (include_zombies or table[pid][2] != 'Z')}

    def reap():
        # Let Popen collect the main child's status before collecting adoptees.
        child.poll()
        if child.returncode is not None:
            while True:
                try:
                    if os.waitpid(-1, os.WNOHANG)[0] == 0:
                        break
                except ChildProcessError:
                    break

    def send(sig):
        live = scan()  # record ancestry before killing parents
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        current_table = snapshot()
        for pid in live:
            # Recheck start time immediately before signalling.
            current = current_table.get(pid)
            if current and current[1] == seen[pid]:
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass

    def report(survivors):
        temporary = report_path + '.tmp'
        with open(temporary, 'w') as f:
            json.dump({'pgid': pgid, 'pids': sorted(seen), 'survivors': sorted(survivors)}, f)
        os.replace(temporary, report_path)

    while child.poll() is None and not stopping:
        report(scan())
        time.sleep(0.05)
    grace = 0.0
    if stopping:
        try:
            with open(report_path + '.grace') as f:
                grace = max(0.0, float(f.read()))
        except (OSError, ValueError):
            grace = 15.0
    send(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while scan() and time.monotonic() < deadline:
        reap()
        time.sleep(0.05)
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        reap()
        remaining = scan(include_zombies=True)
        if not remaining:
            break
        send(signal.SIGKILL)
        time.sleep(0.05)
    reap()
    report(scan())
    rc = child.poll()
    # Preserve signal exit codes in the parent's Popen interface.
    if rc is not None and rc < 0:
        if -rc not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(-rc, signal.SIG_DFL)
        os.kill(os.getpid(), -rc)
    os._exit(rc if rc is not None else 1)
