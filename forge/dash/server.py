"""FastAPI app: one static page, a JSON snapshot, and an SSE stream that pushes a new snapshot on change.

A single background task polls the reader (``poll_s``) and publishes; every SSE client just waits on the
published version, so N viewers cost one reader.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from forge.dash.reader import Reader

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
HEARTBEAT_S = 15.0


class Hub:
    def __init__(self, root: str, poll_s: float = 1.0):
        self.reader = Reader(root)
        self.poll_s = poll_s
        self.payload = "{}"
        self.digest = ""
        self.version = 0
        self.error: str | None = None
        self.cond = asyncio.Condition()
        self.lock = asyncio.Lock()

    def _build(self) -> str:
        snap = self.reader.snapshot()
        snap["server"] = {"version": self.version, "error": self.error, "poll_s": self.poll_s}
        return json.dumps(snap, default=str, separators=(",", ":"))

    async def refresh(self) -> None:
        async with self.lock:
            try:
                payload = await asyncio.to_thread(self._build)
                self.error = None
            except Exception as e:  # keep serving the last good snapshot
                self.error = f"{type(e).__name__}: {e}"[:300]
                return
            # generated_at changes every poll; hash the rest so clients only hear about real changes
            body = json.loads(payload)
            body.pop("generated_at", None)
            body.pop("server", None)
            digest = hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
            if digest != self.digest:
                self.digest, self.payload = digest, payload
                self.version += 1
                async with self.cond:
                    self.cond.notify_all()

    async def run(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(self.poll_s)


def create_app(root: str, poll_s: float = 1.0, fixture_root: str | None = None) -> FastAPI:
    """Serve ``root`` as the live view; with ``fixture_root``, also serve it under ``?src=fixture``."""
    app = FastAPI(title="recipe-forge dash", docs_url=None, redoc_url=None)
    hubs = {"live": Hub(root, poll_s)}
    if fixture_root:
        hubs["fixture"] = Hub(fixture_root, poll_s)
    app.state.hubs = hubs

    def pick(src: str | None) -> Hub:
        return hubs.get(src or "live") or hubs["live"]

    @app.on_event("startup")
    async def _start() -> None:
        app.state.tasks = []
        for hub in hubs.values():
            await hub.refresh()
            app.state.tasks.append(asyncio.create_task(hub.run()))

    @app.on_event("shutdown")
    async def _stop() -> None:
        for task in getattr(app.state, "tasks", []):
            task.cancel()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-cache"})

    @app.get("/static/{name}")
    async def static(name: str) -> Response:
        path = os.path.join(STATIC, os.path.basename(name))
        if not os.path.isfile(path):
            return Response(status_code=404)
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.get("/api/snapshot")
    async def snapshot(src: str | None = None) -> Response:
        hub = pick(src)
        if not hub.version:
            await hub.refresh()
        return Response(hub.payload, media_type="application/json")

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse({name: {"ok": h.error is None, "version": h.version, "error": h.error,
                                    "root": h.reader.root} for name, h in hubs.items()} | {"ts": time.time()})

    @app.get("/api/stream")
    async def stream(request: Request, src: str | None = None) -> StreamingResponse:
        hub = pick(src)

        async def gen():
            sent = -1
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    return
                if hub.version != sent:
                    sent = hub.version
                    yield f"id: {sent}\nevent: snapshot\ndata: {hub.payload}\n\n"
                    continue
                try:
                    async with hub.cond:
                        if hub.version == sent:  # checked under the lock, so a notify can't slip past
                            await asyncio.wait_for(hub.cond.wait(), HEARTBEAT_S)
                except asyncio.TimeoutError:
                    yield f": ping {time.time():.0f}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
