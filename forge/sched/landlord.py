"""Minimal landlord MCP client (streamable HTTP JSON-RPC, stdlib only).

Protocol: POST initialize -> read the mcp-session-id header -> POST notifications/initialized
-> POST tools/call with that header. Replies may be plain JSON or an SSE stream.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("LANDLORD_URL", "http://100.112.46.57:7788/mcp")
PROTOCOL = "2025-06-18"


class LandlordUnreachable(Exception):
    """Network-level failure: the coordinator is down or asleep. Retry later."""


class LandlordError(Exception):
    """The landlord answered but refused or failed the call. Do not blindly retry."""


class LandlordClient:
    def __init__(self, url: str = DEFAULT_URL, tenant: str | None = None, timeout: float = 20.0):
        self.url = url
        self.tenant = tenant
        self.timeout = timeout
        self._sid: str | None = None
        self._id = 0
        self._lock = threading.Lock()

    # -- transport ---------------------------------------------------------------
    def _post(self, payload: dict, expect_reply: bool = True):
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self._sid:
            headers["mcp-session-id"] = self._sid
            headers["mcp-protocol-version"] = PROTOCOL
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("mcp-session-id")
                body = resp.read().decode("utf-8", "replace")
                ctype = resp.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            if e.code in (400, 404) and self._sid:
                raise _SessionLost() from e
            raise LandlordUnreachable(f"HTTP {e.code}") from e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise LandlordUnreachable(str(e)) from e
        if sid:
            self._sid = sid
        if not expect_reply:
            return None
        return _parse_body(body, ctype, payload.get("id"))

    def _initialize(self):
        self._sid = None
        self._id += 1
        self._post({"jsonrpc": "2.0", "id": self._id, "method": "initialize",
                    "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                               "clientInfo": {"name": "forge-sched", "version": "1"}}})
        if not self._sid:
            raise LandlordUnreachable("no mcp-session-id in initialize reply")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, expect_reply=False)

    def call(self, tool: str, **args):
        """Call a landlord tool and return its decoded result (dict, list or str)."""
        with self._lock:
            for attempt in range(2):
                if not self._sid:
                    self._initialize()
                self._id += 1
                try:
                    msg = self._post({"jsonrpc": "2.0", "id": self._id, "method": "tools/call",
                                      "params": {"name": tool, "arguments": args}})
                except _SessionLost:
                    self._sid = None
                    continue
                return _tool_result(tool, msg)
            raise LandlordUnreachable("session lost twice")

    # -- tools -------------------------------------------------------------------
    def overview(self):
        return self.call("landlord_overview")

    def rent(self, pct: int, vram_gb: float, purpose: str, minutes: int, building: str | None = None,
             gpu: int | list[int] | None = None, gpus: int = 1):
        args = dict(pct=pct, vram_gb=vram_gb, purpose=purpose, minutes=minutes, gpus=gpus)
        if building:
            args["building"] = building
        if gpu is not None:
            args["gpu"] = gpu
        if self.tenant:
            args["tenant"] = self.tenant
        return self.call("rent_apartment", **args)

    def extend(self, lease_id: str, minutes: int, note: str):
        return self.call("extend_lease", lease_id=lease_id, minutes=minutes, note=note)

    def move_out(self, lease_id: str, outcome: str):
        return self.call("move_out", lease_id=lease_id, outcome=outcome)


class _SessionLost(Exception):
    pass


def _parse_body(body: str, ctype: str, want_id):
    if "text/event-stream" in ctype:
        last = None
        for line in body.splitlines():
            if line.startswith("data:"):
                try:
                    msg = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if msg.get("id") == want_id or want_id is None:
                    last = msg
        if last is None:
            raise LandlordUnreachable("empty SSE reply")
        return last
    try:
        return json.loads(body)
    except ValueError as e:
        raise LandlordUnreachable(f"bad JSON reply: {body[:200]}") from e


def _tool_result(tool: str, msg: dict):
    if "error" in msg:
        raise LandlordError(f"{tool}: {msg['error'].get('message', msg['error'])}")
    res = msg.get("result") or {}
    if res.get("structuredContent") is not None:
        out = res["structuredContent"]
        if isinstance(out, dict) and set(out) == {"result"}:
            out = out["result"]
    else:
        texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
        joined = "\n".join(texts)
        try:
            out = json.loads(joined)
        except ValueError:
            out = joined
    if res.get("isError"):
        raise LandlordError(f"{tool}: {out}")
    if isinstance(out, dict) and out.get("ok") is False:
        raise LandlordError(f"{tool}: {out.get('error') or out.get('reason') or out}")
    return out
