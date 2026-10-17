"""Server test in the real process: start `python -m forge.dash`, read the page, the snapshot and the SSE stream."""
import http.client
import json
import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port, path, timeout=5):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    return r.status, r.read()


def _sse_events(port, path, n, timeout=15, until=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    assert r.status == 200 and r.getheader("content-type").startswith("text/event-stream")
    events, buf = [], b""
    deadline = time.time() + timeout
    while len(events) < n and time.time() < deadline:
        buf += r.fp.read1(65536) if hasattr(r.fp, "read1") else r.read(1)
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            data = [ln[6:] for ln in block.split(b"\n") if ln.startswith(b"data: ")]
            if data:
                events.append(json.loads(b"\n".join(data)))
                if until and until(events[-1]):
                    n = len(events)
    c.close()
    return events


def test_server_end_to_end(tmp_path):
    port = _free_port()
    root = tmp_path / "root"
    root.mkdir()
    env = dict(os.environ, PYTHONPATH=REPO)
    proc = subprocess.Popen([sys.executable, "-m", "forge.dash", "--root", str(root), "--host", "127.0.0.1",
                             "--port", str(port), "--poll", "0.2"], env=env, cwd=str(tmp_path),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                status, body = _get(port, "/api/health", timeout=1)
                if status == 200:
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError(proc.stdout.read().decode(errors="replace") if proc.poll() is not None else "no start")
        health = json.loads(body)
        assert health["live"]["ok"] and health["fixture"]["ok"]

        status, page = _get(port, "/")
        assert status == 200 and b"recipe-forge" in page and b"/static/app.js" in page
        for name in ("app.js", "style.css"):
            assert _get(port, f"/static/{name}")[0] == 200
        assert _get(port, "/static/..%2Fserver.py")[0] == 404

        live = json.loads(_get(port, "/api/snapshot")[1])
        assert live["fixture"] is False and live["missions"] == []
        fx = json.loads(_get(port, "/api/snapshot?src=fixture")[1])
        assert fx["fixture"] is True and fx["leaderboard"] and fx["missions"]

        # the live root changes -> a new snapshot arrives on the stream
        events = _sse_events(port, "/api/stream", 1)
        assert events and events[0]["missions"] == []
        ev = root / "missions" / "m1" / "events.jsonl"
        ev.parent.mkdir(parents=True)
        ev.write_text(json.dumps({"t": time.time(), "kind": "mission_start", "recipe": "r1"}) + "\n")
        events = _sse_events(port, "/api/stream", 2, until=lambda e: bool(e["missions"]))
        assert events[-1]["missions"][0]["id"] == "m1"
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
