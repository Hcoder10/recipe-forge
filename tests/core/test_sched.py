import os
import tempfile
import unittest

from forge.sched.core import RequestError, Scheduler
from forge.sched.keeper import LeaseKeeper
from forge.sched.landlord import LandlordError, LandlordUnreachable
from forge.sched.machine import Machine


class FakeLandlord:
    def __init__(self):
        self.down = False
        self.refuse = set()
        self.leases = {}
        self.calls = []
        self.n = 0

    def _check(self):
        if self.down:
            raise LandlordUnreachable("down")

    def rent(self, pct, vram_gb, purpose, minutes, building=None, gpu=None, gpus=1):
        self._check()
        self.calls.append(("rent", gpu))
        if gpu in self.refuse:
            raise LandlordError(f"GPU {gpu} is not free")
        self.n += 1
        lid = f"apt_{self.n}"
        self.leases[lid] = {"gpu": gpu, "minutes": minutes, "purpose": purpose}
        return {"lease": {"id": lid, "gpu_idxs": [gpu], "minutes_left": minutes,
                          "env": {"LANDLORD_LEASE_ID": lid, "CUDA_VISIBLE_DEVICES": str(gpu)}}}

    def extend(self, lease_id, minutes, note):
        self._check()
        self.calls.append(("extend", lease_id))
        self.leases[lease_id]["minutes"] += minutes
        return {"lease": {"id": lease_id, "minutes_left": self.leases[lease_id]["minutes"]}}

    def move_out(self, lease_id, outcome):
        self._check()
        self.calls.append(("move_out", lease_id))
        if lease_id not in self.leases:
            raise LandlordError("no such lease")
        del self.leases[lease_id]
        return {"ok": True}


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def rig():
    return Machine(name="rig", building="rig", timing_gpus=[0, 1], work_gpus=[2],
                   vram_gb={0: 95.6, 1: 95.6, 2: 95.6}, gpu_uuids={0: "u0", 1: "u1", 2: "u2"},
                   gate_cpus=[0, 1, 32, 33], agent_cpus=list(range(2, 32)))


class SchedTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ll = FakeLandlord()
        self.clock = Clock()
        self.alive = {}
        self.keeper = LeaseKeeper(os.path.join(self.dir, "leases.json"), self.ll, building="rig", clock=self.clock)
        self.s = self.make()

    def make(self):
        return Scheduler(rig(), self.keeper, os.path.join(self.dir, "grants.json"), clock=self.clock,
                         alive=lambda pid, start=None: self.alive.get(pid, True))

    def req(self, kind, **kw):
        return self.s.request(kind, owner="t", purpose="unit test of the scheduler", **kw)

    def test_bench_gets_timing_gpu_with_lease_env(self):
        g = self.req("bench")
        self.assertEqual(g["state"], "active")
        self.assertEqual(g["gpus"], [0])
        self.assertEqual(g["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertTrue(g["env"]["LANDLORD_LEASE_ID"].startswith("apt_"))
        self.assertEqual(len(g["ports"]), 2)
        self.assertEqual(g["cpus"], list(range(2, 32)))

    def test_gate_exclusive_and_drains_timing(self):
        b = self.req("bench")
        gate = self.req("gate")
        self.assertEqual(gate["state"], "queued")
        self.assertIn("timing GPUs", gate["wait_reason"])
        # while a gate waits, a new bench must not take the free timing GPU 1; it goes to the work GPU
        b2 = self.req("bench")
        self.assertEqual(b2["gpus"], [2])
        b3 = self.req("bench", where="timing")
        self.assertEqual(b3["state"], "queued")
        self.assertIn("gate is waiting", b3["wait_reason"])
        self.s.release(b["id"])
        self.s.tick()
        gate = self.s.get(gate["id"])
        self.assertEqual(gate["state"], "active")
        self.assertEqual(gate["gpus"], [0, 1])
        self.assertEqual(len(gate["leases"]), 2)
        self.assertEqual(gate["cpus"], [0, 1, 32, 33])
        self.assertEqual(self.s.get(b3["id"])["state"], "queued")

    def test_work_vram_accounting(self):
        a = self.req("work", vram_gb=60)
        b = self.req("work", vram_gb=30)
        c = self.req("work", vram_gb=10)
        self.assertEqual((a["gpus"], b["gpus"]), ([2], [2]))
        self.assertEqual(c["state"], "queued")
        self.assertIn("GB free", c["wait_reason"])
        self.s.release(b["id"])
        self.s.tick()
        self.assertEqual(self.s.get(c["id"])["state"], "active")

    def test_never_fits_is_refused(self):
        with self.assertRaises(RequestError):
            self.req("work", vram_gb=500)
        with self.assertRaises(RequestError):
            self.s.request("bench", owner="t", purpose="")

    def test_landlord_down_queues_then_recovers(self):
        self.ll.down = True
        g = self.req("bench")
        self.assertEqual(g["state"], "queued")
        self.assertIn("unreachable", g["wait_reason"])
        self.ll.down = False
        self.clock.t += 11
        self.s.tick()
        self.assertEqual(self.s.get(g["id"])["state"], "active")

    def test_landlord_refusal_queues(self):
        self.ll.refuse = {0, 1, 2}
        g = self.req("bench")
        self.assertEqual(g["state"], "queued")
        self.assertIn("refused", g["wait_reason"])
        self.ll.refuse = set()
        self.clock.t += 31
        self.s.tick()
        self.assertEqual(self.s.get(g["id"])["state"], "active")

    def test_partial_gate_rent_returns_first_lease(self):
        self.ll.refuse = {1}
        g = self.req("gate")
        self.assertEqual(g["state"], "queued")
        self.assertEqual(self.ll.leases, {})       # GPU 0's lease was moved out again

    def test_extend_and_move_out_survive_outage(self):
        g = self.req("bench", minutes=20)
        lid = g["leases"]["0"]
        self.clock.t += 10 * 60            # 10 min left < 12 min threshold
        self.s.tick()
        self.assertIn(("extend", lid), self.ll.calls)
        self.ll.down = True
        self.s.release(g["id"], "finished")
        self.assertIn(lid, self.ll.leases)
        self.assertIn(lid, self.keeper.pending_out)
        # a restarted keeper still knows it must move out
        k2 = LeaseKeeper(os.path.join(self.dir, "leases.json"), self.ll, building="rig", clock=self.clock)
        self.assertIn(lid, k2.pending_out)
        self.ll.down = False
        k2.tick(set())
        self.assertNotIn(lid, self.ll.leases)
        self.assertEqual(k2.pending_out, {})

    def test_dead_holder_releases(self):
        g = self.req("bench", holder_pid=4242)
        self.alive[4242] = False
        self.s.tick()
        self.assertEqual(self.s.get(g["id"])["state"], "released")
        self.assertEqual(self.ll.leases, {})

    def test_deadline_expires(self):
        g = self.req("bench", minutes=5)
        self.clock.t += 6 * 60
        self.s.tick()
        self.assertEqual(self.s.get(g["id"])["state"], "expired")
        self.assertEqual(self.ll.leases, {})

    def test_shutdown_moves_out_everything(self):
        self.req("bench")
        self.req("work", vram_gb=10)
        self.assertEqual(len(self.ll.leases), 2)
        self.assertTrue(self.s.shutdown("test over"))
        self.assertEqual(self.ll.leases, {})

    def test_state_survives_restart(self):
        g = self.req("bench")
        s2 = self.make()
        self.assertEqual(s2.get(g["id"])["state"], "active")
        self.assertEqual(s2.get(g["id"])["gpus"], [0])


if __name__ == "__main__":
    unittest.main()
