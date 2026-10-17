"""The per-machine scheduler daemon: HTTP JSON on 127.0.0.1, one per machine.

    python -m forge.sched.daemon            # run in the foreground
    python -m forge.sched.daemon ensure     # start it as a systemd --user unit if it is not up

On SIGTERM/SIGINT/SIGHUP (and at exit) it releases every grant and moves out every lease. Leases it
could not move out (landlord down) are persisted and retried on the next start.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from forge.sched import machine as machine_mod
from forge.sched.core import RequestError, Scheduler
from forge.sched.keeper import LeaseKeeper
from forge.sched.landlord import LandlordClient

PORT = int(os.environ.get("FORGE_SCHED_PORT", "7791"))
TICK_S = 5.0


def state_dir() -> str:
    d = os.path.join(machine_mod.forge_root(), "sched")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


class _Handler(BaseHTTPRequestHandler):
    sched: Scheduler = None  # set by serve()

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        try:
            if self.path == "/status":
                return self._send(200, self.sched.status())
            if self.path.startswith("/grant/"):
                return self._send(200, self.sched.get(self.path.split("/", 2)[2]))
            if self.path == "/health":
                return self._send(200, {"ok": True, "pid": os.getpid()})
            self._send(404, {"error": "not found"})
        except KeyError as e:
            self._send(404, {"error": f"unknown grant {e}"})

    def do_POST(self):
        try:
            b = self._body()
            if self.path == "/request":
                return self._send(200, self.sched.request(**b))
            if self.path == "/renew":
                return self._send(200, self.sched.renew(b["id"], int(b["minutes"])))
            if self.path == "/release":
                return self._send(200, self.sched.release(b["id"], b.get("outcome", "done")))
            self._send(404, {"error": "not found"})
        except (RequestError, TypeError, ValueError) as e:
            self._send(400, {"error": str(e)})
        except KeyError as e:
            self._send(404, {"error": f"unknown grant {e}"})


def serve() -> None:
    sd = state_dir()
    lockf = open(os.path.join(sd, "daemon.lock"), "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("forge.sched: another scheduler daemon is running on this machine", file=sys.stderr)
        sys.exit(1)
    m = machine_mod.load()
    tenant = f"forge-sched@{m.name}/recipe-forge"
    keeper = LeaseKeeper(os.path.join(sd, "leases.json"), LandlordClient(tenant=tenant), building=m.building)
    sched = Scheduler(m, keeper, os.path.join(sd, "grants.json"))
    _Handler.sched = sched
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), _Handler)
    httpd.daemon_threads = True
    stop = threading.Event()
    done = threading.Event()

    def _shutdown(reason: str):
        if done.is_set():
            return
        done.set()
        stop.set()
        ok = sched.shutdown(f"scheduler stopped ({reason})")
        print(f"forge.sched: shutdown ({reason}); all leases moved out: {ok}", flush=True)

    def _on_signal(signum, _frame):
        threading.Thread(target=httpd.shutdown, daemon=True).start()
        _shutdown(signal.Signals(signum).name)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _on_signal)

    def _ticker():
        while not stop.is_set():
            try:
                sched.tick()
            except Exception as e:  # keep ticking; the error is visible in status
                keeper.last_error = f"tick: {type(e).__name__}: {e}"
            stop.wait(TICK_S)

    threading.Thread(target=_ticker, daemon=True).start()
    print(f"forge.sched: {m.name} on 127.0.0.1:{PORT} timing={m.timing_gpus} work={m.work_gpus}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        _shutdown("exit")
        lockf.close()


def ensure() -> int:
    """Start the daemon as a transient systemd --user service unless it already answers."""
    from forge.sched.client import SchedClient
    if SchedClient().healthy():
        print("forge.sched: already running")
        return 0
    pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {"FORGE_ROOT": machine_mod.forge_root()}
    if "LANDLORD_URL" in os.environ:
        env["LANDLORD_URL"] = os.environ["LANDLORD_URL"]
    cmd = ["systemd-run", "--user", "--unit=forge-sched", "--collect", "-p", "KillSignal=SIGTERM",
           "-p", "TimeoutStopSec=90", f"--working-directory={pkg_parent}"]
    cmd += [f"--setenv={k}={v}" for k, v in env.items()]
    cmd += [sys.executable, "-m", "forge.sched.daemon"]
    subprocess.run(cmd, check=True)
    for _ in range(40):
        if SchedClient().healthy():
            print("forge.sched: started")
            return 0
        time.sleep(0.25)
    print("forge.sched: did not come up; see journalctl --user -u forge-sched", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ensure":
        sys.exit(ensure())
    serve()
