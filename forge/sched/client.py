"""Client for the per-machine scheduler daemon.

    from forge.sched.client import SchedClient
    sc = SchedClient()
    with sc.acquire("bench", owner="mission-7/worker-1", purpose="bench Bolt recipe, 1 GPU, 20 min",
                    minutes=20, where="any") as g:
        print(g.gpus, g.ports, g.env)          # g.wait_reason while queued; acquire blocks until active
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("FORGE_SCHED_PORT", "7791"))


class SchedError(RuntimeError):
    pass


class Grant:
    def __init__(self, client: "SchedClient", d: dict):
        self.client = client
        self.d = d

    def __getattr__(self, k):
        try:
            return self.d[k]
        except KeyError:
            raise AttributeError(k) from None

    def refresh(self) -> "Grant":
        self.d = self.client.get(self.d["id"]).d
        return self

    def renew(self, minutes: int) -> "Grant":
        self.d = self.client._post("/renew", {"id": self.d["id"], "minutes": minutes})
        return self

    def release(self, outcome: str = "done") -> None:
        if self.d.get("state") in ("released", "expired"):
            return
        self.d = self.client._post("/release", {"id": self.d["id"], "outcome": outcome})

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.release("done" if et is None else f"ended with {et.__name__}: {str(ev)[:200]}")

    def __repr__(self):
        return f"Grant({self.d.get('id')} {self.d.get('state')} gpus={self.d.get('gpus')} {self.d.get('wait_reason') or ''})"


class SchedClient:
    def __init__(self, port: int = PORT, host: str = "127.0.0.1", timeout: float = 60.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error")
            except ValueError:
                msg = str(e)
            raise SchedError(msg) from None
        except (urllib.error.URLError, OSError) as e:
            raise SchedError(f"scheduler not reachable on {self.base}: {e}") from None

    def _post(self, path: str, body: dict) -> dict:
        return self._req("POST", path, body)

    def healthy(self) -> bool:
        try:
            return bool(self._req("GET", "/health").get("ok"))
        except SchedError:
            return False

    def status(self) -> dict:
        return self._req("GET", "/status")

    def get(self, gid: str) -> Grant:
        return Grant(self, self._req("GET", f"/grant/{gid}"))

    def request(self, kind: str, owner: str, purpose: str, minutes: int = 60, vram_gb: float | None = None,
                gpu: int | None = None, where: str = "any", pct: int | None = None,
                holder_pid: int | None = None) -> Grant:
        """Queue a request and return at once (state queued|leasing|active)."""
        body = dict(kind=kind, owner=owner, purpose=purpose, minutes=minutes, vram_gb=vram_gb, gpu=gpu,
                    where=where, pct=pct, holder_pid=holder_pid if holder_pid is not None else os.getpid())
        return Grant(self, self._post("/request", body))

    def acquire(self, kind: str, owner: str, purpose: str, timeout: float | None = None, on_wait=None,
                **kw) -> Grant:
        """Request and block until active. on_wait(grant) is called when the wait reason changes.
        On timeout the request is withdrawn and SchedError says why it was still waiting."""
        g = self.request(kind, owner, purpose, **kw)
        end = None if timeout is None else time.time() + timeout
        last = None
        while g.state in ("queued", "leasing"):
            if on_wait and g.wait_reason != last:
                on_wait(g)
                last = g.wait_reason
            if end is not None and time.time() > end:
                reason = g.wait_reason
                g.release("withdrawn: waited too long")
                raise SchedError(f"timed out waiting for {kind}: {reason}")
            time.sleep(1.0)
            g.refresh()
        if g.state != "active":
            raise SchedError(f"grant {g.id} ended before it started: {g.outcome}")
        return g
