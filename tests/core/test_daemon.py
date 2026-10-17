"""Real-process test: scheduler daemon + client against a fake landlord MCP server (no GPU used).
Checks the MCP session handshake, grant flow over HTTP, and move-out on SIGTERM."""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeMCP(BaseHTTPRequestHandler):
    leases = {}
    moved_out = []
    sessions = set()

    def log_message(self, *a):
        pass

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if msg.get("method") == "initialize":
            sid = f"s{len(self.sessions)}"
            self.sessions.add(sid)
            return self._reply({"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": "2025-06-18"}}, sid)
        if self.headers.get("mcp-session-id") not in self.sessions:
            self.send_response(404)
            self.end_headers()
            return
        if msg.get("method") == "notifications/initialized":
            self.send_response(202)
            self.end_headers()
            return
        p = msg["params"]
        name, a = p["name"], p["arguments"]
        if name == "rent_apartment":
            lid = f"apt_{len(self.leases) + 1}"
            self.leases[lid] = a
            out = {"lease": {"id": lid, "gpu_idxs": [a["gpu"]], "minutes_left": a["minutes"],
                             "env": {"LANDLORD_LEASE_ID": lid}}}
        elif name == "move_out":
            self.moved_out.append(a["lease_id"])
            out = {"ok": True}
        elif name == "extend_lease":
            out = {"lease": {"id": a["lease_id"], "minutes_left": 60}}
        else:
            out = {"buildings": []}
        # answer as SSE, like streamable HTTP servers may
        body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'content': [{'type': 'text', 'text': json.dumps(out)}]}})}\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def _reply(self, obj, sid):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("mcp-session-id", sid)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipUnless(sys.platform.startswith("linux"), "daemon test runs on Linux")
class DaemonTest(unittest.TestCase):
    def test_daemon_grant_and_sigterm_moves_out(self):
        mcp_port, sched_port = free_port(), free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", mcp_port), FakeMCP)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        root = tempfile.mkdtemp(prefix="forge_daemon_test_")
        with open(os.path.join(root, "machine.json"), "w") as f:
            json.dump({"name": "t", "building": "t", "timing_gpus": [0, 1], "work_gpus": [2],
                       "vram_gb": {"0": 95, "1": 95, "2": 95}, "gpu_uuids": {"0": "a", "1": "b", "2": "c"},
                       "gate_cpus": [0], "agent_cpus": [1]}, f)
        env = {**os.environ, "FORGE_ROOT": root, "LANDLORD_URL": f"http://127.0.0.1:{mcp_port}/mcp",
               "FORGE_SCHED_PORT": str(sched_port)}
        pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        proc = subprocess.Popen([sys.executable, "-m", "forge.sched.daemon"], cwd=pkg, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            from forge.sched.client import SchedClient
            sc = SchedClient(port=sched_port)
            for _ in range(50):
                if sc.healthy():
                    break
                time.sleep(0.1)
            g = sc.acquire("bench", owner="test", purpose="daemon integration test grant", minutes=5, timeout=20)
            self.assertEqual(g.state, "active")
            self.assertEqual(g.gpus, [0])
            self.assertTrue(g.env["LANDLORD_LEASE_ID"].startswith("apt_"))
            g2 = sc.request("work", owner="test", purpose="daemon integration work grant", vram_gb=10)
            self.assertEqual(g2.state, "active")
            self.assertEqual(len(FakeMCP.leases), 2)
            proc.send_signal(signal.SIGTERM)
            out, _ = proc.communicate(timeout=60)
            self.assertIn("all leases moved out: True", out)
            self.assertEqual(sorted(FakeMCP.moved_out), sorted(FakeMCP.leases))
        finally:
            if proc.poll() is None:
                proc.kill()
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main()
