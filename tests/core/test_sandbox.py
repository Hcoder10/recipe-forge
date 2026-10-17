"""Sandbox tests. Linux with Landlock only. GPU test runs only with a lease:
FORGE_TEST_GPU=<idx> FORGE_TEST_LEASE_ID=<lease> python -m unittest tests.core.test_sandbox
"""
import os
import sys
import tempfile
import textwrap
import time
import unittest

from forge.sandbox import SandboxError, Spec, landlock, launch

ok = landlock.abi_version() >= 6
VENV = os.path.dirname(os.path.dirname(sys.executable))


@unittest.skipUnless(ok, "needs Linux Landlock ABI >= 6")
class SandboxTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".forge_sbx_test_")
        self.secret = os.path.join(self.root, "secret.txt")
        with open(self.secret, "w") as f:
            f.write("hidden")
        self.src = os.path.join(self.root, "src")
        os.makedirs(self.src)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def run_py(self, code, **kw):
        path = os.path.join(self.src, f"p{time.time_ns()}.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(code))
        spec = Spec(cmd=[sys.executable, path], job_dir=os.path.join(self.root, "jobs", os.path.basename(path)),
                    ro=[VENV, self.src] + kw.pop("ro", []), **kw)
        job = launch(spec)
        rc = job.wait(timeout=60)
        return rc, job.tail(20000), job

    def test_files_net_signals_env(self):
        rc, out, _ = self.run_py(f"""
            import os, socket
            def t(name, f):
                try:
                    f(); print(name, "ALLOWED")
                except Exception as e:
                    print(name, "DENIED")
            t("secret", lambda: open({self.secret!r}).read())
            t("home", lambda: os.listdir(os.path.expanduser("~cube") if False else "/home"))
            t("ws", lambda: open("x", "w").write("1"))
            t("tmp", lambda: open("/tmp/forge_sbx_probe", "w").write("1"))
            t("bind_in", lambda: socket.socket().bind(("127.0.0.1", int(os.environ["FORGE_PORT_LO"]))))
            t("bind_out", lambda: socket.socket().bind(("127.0.0.1", 23999)))
            t("connect_out", lambda: socket.create_connection(("127.0.0.1", 22), timeout=2))
            t("signal_parent", lambda: os.kill(1, 0))
            print("ENV", ",".join(sorted(os.environ)))
            print("CPUS", sorted(os.sched_getaffinity(0)))
        """, ports=(23100, 23109), cpus=[2, 3])
        self.assertEqual(rc, 0, out)
        for name, want in [("secret", "DENIED"), ("ws", "ALLOWED"), ("tmp", "DENIED"), ("bind_in", "ALLOWED"),
                           ("bind_out", "DENIED"), ("connect_out", "DENIED"), ("signal_parent", "DENIED")]:
            self.assertIn(f"{name} {want}", out)
        env = [l for l in out.splitlines() if l.startswith("ENV ")][0]
        self.assertNotIn("SSH_", env)
        self.assertNotIn("DBUS", env)
        self.assertIn("CPUS [2, 3]", out)

    def test_ro_inside_rw_refused(self):
        rw = os.path.join(self.root, "rw")
        with self.assertRaises(SandboxError):
            launch(Spec(cmd=["true"], job_dir=os.path.join(self.root, "j"), rw=[rw], ro=[os.path.join(rw, "x")]))

    def test_ro_is_read_only(self):
        rc, out, _ = self.run_py(f"""
            try:
                open({os.path.join(self.src, "w.txt")!r}, "w").write("x"); print("WROTE")
            except PermissionError:
                print("RO_OK")
        """)
        self.assertIn("RO_OK", out)

    def test_setsid_child_reaped(self):
        marker = os.path.join(self.root, "jobs", "child.pid")
        rc, out, job = self.run_py(f"""
            import subprocess, os
            p = subprocess.Popen(["sleep", "300"], start_new_session=True)
            print("CHILD", p.pid, flush=True)
        """)
        pid = int([l for l in out.splitlines() if l.startswith("CHILD")][0].split()[1])
        time.sleep(0.5)
        self.assertFalse(os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split()[2] != "Z",
                         "setsid child survived the scope")

    def test_timeout_stops_scope(self):
        t0 = time.time()
        path = os.path.join(self.src, "sleep.py")
        with open(path, "w") as f:
            f.write("import time\ntime.sleep(300)\n")
        job = launch(Spec(cmd=[sys.executable, path], job_dir=os.path.join(self.root, "jobs", "slp"),
                          ro=[VENV, self.src]))
        job.wait(timeout=2)
        self.assertLess(time.time() - t0, 30)
        self.assertIsNotNone(job.returncode)

    @unittest.skipUnless(os.environ.get("FORGE_TEST_LEASE_ID"), "needs a landlord lease for one GPU")
    def test_gpu_grant_is_index0(self):
        gpu = int(os.environ["FORGE_TEST_GPU"])
        import subprocess
        bus = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        rc, out, _ = self.run_py("""
            import torch
            torch.cuda.set_per_process_memory_fraction(0.02)
            p = torch.cuda.get_device_properties(0)
            print("COUNT", torch.cuda.device_count())
            print("BUS %04x:%02x:%02x" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id))
            print("SUM", torch.ones(8, device="cuda").sum().item())
        """, gpus=[gpu], env={"LANDLORD_LEASE_ID": os.environ["FORGE_TEST_LEASE_ID"]})
        self.assertEqual(rc, 0, out)
        self.assertIn("COUNT 1", out)
        self.assertIn("SUM 8.0", out)
        want = bus.split(":")[1].lower()     # 00000000:F1:00.0 -> f1
        self.assertIn(f":{want}:", out.lower())


if __name__ == "__main__":
    unittest.main()
