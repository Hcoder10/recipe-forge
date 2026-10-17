"""Gate tests with fake servers (no GPU): statistics, lossless judging, ledger flow, retention."""
import hashlib
import math
import os
import random
import tempfile
import unittest

TMP = tempfile.mkdtemp(prefix="forge_gate_test_")
os.environ["FORGE_ROOT"] = TMP

from forge.gate import records as L  # noqa: E402
from forge.gate import lock as gate_lock  # noqa: E402
from forge.gate import lossless, pool, runner, service, stats  # noqa: E402

VOCAB = 50


def _dist(prev: int, shift: float = 0.0) -> dict[int, float]:
    """Deterministic next-token distribution over 6 tokens given the previous token."""
    rng = random.Random(prev)
    toks = rng.sample(range(VOCAB), 6)
    logits = [3.0, 1.0, 0.5, 0.2, 0.1, 0.0]
    if shift:
        logits = [1.0, 3.0] + logits[2:]    # lossy: swaps the top two
    z = math.log(sum(math.exp(x) for x in logits))
    return {t: x - z for t, x in zip(toks, logits)}


class FakeServer:
    def __init__(self, side, name, speed, lossy, noise_seed):
        self.side, self.name, self.speed, self.lossy = side, name, speed, lossy
        self.rng = random.Random(noise_seed)
        self.unit = name
        self.job = None

    def start(self):
        self.jitter = math.exp(self.rng.gauss(0, 0.004))     # per-server-lifetime noise
        return 1.0

    def model_name(self):
        return "m"

    def _shift(self, prev):
        return 2.5 if self.lossy and prev % 3 == 0 else 0.0

    def complete(self, model, prompt_ids, max_tokens, logprobs=None, prompt_logprobs=None, timeout=0):
        if prompt_logprobs:
            plp = [None] + [{str(k): {"logprob": v} for k, v in _dist(prompt_ids[i - 1], self._shift(prompt_ids[i - 1])).items()}
                            for i in range(1, len(prompt_ids))]
            return {"choices": [{"text": "", "prompt_logprobs": plp}], "usage": {"completion_tokens": 1}}, 0.01
        prev, out, lps = prompt_ids[-1], [], []
        for _ in range(max_tokens):
            d = _dist(prev, self._shift(prev))
            t = max(d, key=d.get)
            out.append(t)
            lps.append(d[t])
            prev = t
        dt = len(out) / (100.0 * self.speed * self.jitter)
        return {"choices": [{"text": " ".join(f"w{t}" for t in out),
                             "logprobs": {"tokens": [f"token_id:{t}" for t in out], "token_logprobs": lps}}],
                "usage": {"completion_tokens": len(out)}}, dt

    def stop(self):
        pass

    def log_tail(self, n=0):
        return "fake log"

    def cleanup(self):
        pass


def chat_ids(messages):
    h = hashlib.sha256(messages[0]["content"].encode()).digest()
    return [b % VOCAB for b in h[:24]]


class FakeCtx:
    speed_b = 1.0
    lossy_b = False

    def __init__(self, req, grant, run_dir, log):
        self.req = req
        self.n = 0

    def make_server(self, side, name):
        self.n += 1
        if side == "B":
            return FakeServer(side, name, self.speed_b, self.lossy_b, hash((self.req["gate_id"], name)))
        return FakeServer(side, name, 1.0, False, hash((self.req["gate_id"], name)))

    chat_ids = staticmethod(chat_ids)

    @staticmethod
    def retok(text):
        return text.split()

    @staticmethod
    def quiet():
        return None


class Grant:
    gpus = [0]
    gpu_uuids = ["u0"]
    ports = [20000, 20099]
    cpus = [0]
    leases = {"0": "apt_x"}


RECIPE = {"id": "fake", "model": "org/fake", "serve": {"args": []},
          "variants": [{"name": "c1", "concurrency": 1, "max_tokens": 16, "metric": "decode_tok_s"}]}


def fake_python():
    d = os.path.join(TMP, "venv", "bin")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "python")
    open(p, "w").write("#fake")
    return p


class StatsTest(unittest.TestCase):
    def test_t_ppf(self):
        self.assertAlmostEqual(stats.t_ppf(0.95, 1), 6.3138, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.999, 7), 4.7853, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.975, 1000), 1.9623, places=3)

    def test_paired(self):
        null = [0.004, -0.003, 0.002, -0.005, 0.001, 0.003, -0.002, -0.001]
        self.assertTrue(stats.paired_test([0.05, 0.06, 0.055, 0.052], null, 0.001)["win"])
        self.assertFalse(stats.paired_test([0.05, -0.001, 0.055, 0.052], null, 0.001)["win"])
        self.assertFalse(stats.paired_test([0.004, 0.003, 0.005, 0.002], null, 0.001)["win"])
        self.assertFalse(stats.paired_test([0.05] * 4, null[:1], 0.001)["win"])


class PoolTest(unittest.TestCase):
    def test_deterministic_disjoint(self):
        p = pool.build(TMP)
        self.assertEqual(oct(os.stat(os.path.join(TMP, "gate_hidden")).st_mode & 0o777), "0o700")
        p2 = pool.build(TMP)
        self.assertEqual(p["sha256"], p2["sha256"])
        s = pool.Slices(p)
        ids = lambda xs: {x["id"] for x in xs}
        r0, r1 = ids(s.rep(0, 8)), ids(s.rep(1, 8))
        c0 = ids(s.confirm(0, 0, 8))
        self.assertEqual(len(r0), 8)
        self.assertFalse(r0 & r1)
        self.assertFalse(c0 & r0)
        self.assertFalse(ids(s.warmup()) & r0)
        self.assertFalse(ids(s.probes()) & (r0 | r1))
        self.assertEqual(ids(s.rep(0, 8)), r0)


class LosslessTest(unittest.TestCase):
    def test_judge(self):
        same = lossless.probe_compare([[_dist(i) for i in range(30)]], [[_dist(i) for i in range(30)]])
        self.assertEqual(same["top1_agree"], 1.0)
        bad = lossless.probe_compare([[_dist(i) for i in range(30)]], [[_dist(i, 2.5) for i in range(30)]])
        self.assertLess(bad["top1_agree"], 0.5)
        b = lossless.band([same, same], [{"flip_rate": 0.0, "max_regret": 0.0, "lp_gap_mean": 0.0}])
        self.assertTrue(lossless.judge([same], [{"flip_rate": 0.0, "max_regret": 0.1, "lp_gap_mean": 0.0}], b)["pass"])
        self.assertFalse(lossless.judge([bad], [{"flip_rate": 0.0, "max_regret": 0.1, "lp_gap_mean": 0.0}], b)["pass"])


class GateFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gate_lock.write("main")
        cls.py = fake_python()
        cls.bundle = os.path.join(TMP, "bundle")
        os.makedirs(cls.bundle, exist_ok=True)
        with open(os.path.join(cls.bundle, "manifest.json"), "w") as f:
            f.write('{"config_delta": {"--x": "1"}}')

    def drain(self, max_runs=10):
        out = []
        for _ in range(max_runs):
            req = L.next_request()
            if req is None:
                break
            L.append(req, status="running")
            v = service.run_one(req, Grant(), log=lambda m: None, ctx_factory=FakeCtx)
            L.finish(req, v)
            out.append((req, v))
        return out

    def test_flow(self):
        import forge.gate as gate
        FakeCtx.speed_b, FakeCtx.lossy_b = 1.10, False
        row = gate.request("fake:c1", self.bundle, mission_id="m1", machine="rig", requester="t", claim="10%",
                           recipe=RECIPE, python=self.py)
        self.assertEqual(row["status"], "queued")
        dup = gate.request("fake:c1", self.bundle, mission_id="m1", recipe=RECIPE, python=self.py)
        self.assertTrue(dup.get("dedup"))
        runs = self.drain()
        kinds = [r["kind"] for r, _ in runs]
        self.assertEqual(kinds[:2], ["reference", "reference"])      # setup controls first
        self.assertIn("candidate", kinds)
        self.assertIn("confirm", kinds)
        st = gate.status(row["gate_id"])
        self.assertEqual(st["verdict"], "win", st)
        self.assertTrue(st["confirmed"], st)
        self.assertGreater(st["delta_pct"], 5)
        run_dir = os.path.join(TMP, "gate", "runs", row["gate_id"])
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(run_dir) for f in fs)
        self.assertLess(size, 20 * 1024 * 1024)
        self.assertFalse(os.path.exists(os.path.join(run_dir, "bundle")))
        self.assertNotIn("Background material", open(os.path.join(TMP, "gate", "ledger.jsonl")).read())

        # a lossy candidate is caught; a same-speed candidate is a loss
        b2 = os.path.join(TMP, "bundle2")
        os.makedirs(b2, exist_ok=True)
        open(os.path.join(b2, "manifest.json"), "w").write('{"config_delta": {"--y": "1"}}')
        FakeCtx.speed_b, FakeCtx.lossy_b = 1.10, True
        lossy = gate.request("fake:c1", b2, mission_id="m2", recipe=RECIPE, python=self.py)
        self.drain()
        self.assertEqual(gate.status(lossy["gate_id"])["verdict"], "lossy")
        b3 = os.path.join(TMP, "bundle3")
        os.makedirs(b3, exist_ok=True)
        open(os.path.join(b3, "manifest.json"), "w").write('{"config_delta": {"--z": "1"}}')
        FakeCtx.speed_b, FakeCtx.lossy_b = 1.0, False
        same = gate.request("fake:c1", b3, mission_id="m3", recipe=RECIPE, python=self.py)
        self.drain()
        self.assertEqual(gate.status(same["gate_id"])["verdict"], "loss")

    def test_lock_blocks_changed_code(self):
        ok, _, _ = gate_lock.verify("main")
        self.assertTrue(ok)
        ok, msg, _ = gate_lock.verify("nonexistent")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
