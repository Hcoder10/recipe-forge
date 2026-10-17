"""CPU-only regressions for complete gate evidence; never print measurements."""
import math
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from forge.gate import records, stats


class EvidenceTest(unittest.TestCase):
    def test_interval_uses_reference_noise(self):
        result = stats.paired_test([0.0] * 4, [0.01, -0.01] * 4, 0.001)
        self.assertTrue("ci" in result, "confidence interval missing")
        ci = result["ci"]
        radius = stats.t_ppf(0.9995, 8) * 0.01 / math.sqrt(4)
        self.assertAlmostEqual(ci["lower_log_gain"], -radius)
        self.assertAlmostEqual(ci["upper_log_gain"], radius)
        self.assertEqual(ci["confidence"], 0.999)

    def test_finish_retains_interval_and_lossless_evidence(self):
        ci = {"confidence": 0.999, "lower_log_gain": -0.1, "upper_log_gain": 0.1}
        lossless = {"pass": True, "band": {"probe": {}}, "fails": []}
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"FORGE_ROOT": root}):
            records.finish({"gate_id": "evidence-test", "kind": "candidate"},
                           {"verdict": "loss", "test": {"ci": ci}, "lossless": lossless,
                            "pairs": [{"log_gain": 0.0}], "minutes": 1.0})
            row = records.latest()["evidence-test"]
            self.assertTrue(row.get("ci") == ci, "ledger confidence interval missing")
            self.assertTrue(row.get("lossless") == lossless, "ledger lossless band evidence missing")


@unittest.skipUnless(os.environ.get("FORGE_TEST_GATE_ID"), "operator live audit is opt-in")
class LiveEvidenceTest(unittest.TestCase):
    def test_completed_candidate(self):
        from forge.gate import lock
        from forge.sched.client import SchedClient
        gid = os.environ["FORGE_TEST_GATE_ID"]
        root = Path(os.environ["FORGE_ROOT"])
        row = records.latest()[gid]
        v = json.loads((root / "gate/runs" / gid / "verdict.json").read_text())
        req = records.get_request(gid)
        self.assertTrue(row.get("status") == "done", "candidate unfinished")
        self.assertTrue(row.get("verdict") == "loss", "unexpected no-op verdict")
        self.assertTrue(len(row.get("pairs", [])) == 4, "pair evidence incomplete")
        self.assertTrue(row.get("lossless", {}).get("pass") is True, "lossless check failed")
        self.assertTrue(bool(row.get("lossless", {}).get("limits")), "band limits missing")
        ci = row.get("ci") or {}
        self.assertTrue(all(isinstance(ci.get(k), (int, float)) and math.isfinite(ci[k])
                            for k in ("lower_log_gain", "upper_log_gain")), "finite CI missing")
        self.assertTrue(ci["lower_log_gain"] <= ci["upper_log_gain"], "CI bounds reversed")
        self.assertTrue(v.get("lock", {}).get("ok") is True and lock.verify("main")[0],
                        "integrity lock failed")
        self.assertTrue(v.get("bundle_sha256") == req["bundle_hash"], "bundle hash mismatch")
        self.assertTrue(all(v.get(k) for k in ("venv_sha256", "recipe_sha256", "pool_sha256")),
                        "integrity evidence incomplete")
        sched = SchedClient().status()
        self.assertFalse(any(g.get("owner") == "gate/" + gid for g in sched.get("grants", [])),
                         "candidate grant still held")
        self.assertFalse(sched.get("keeper", {}).get("pending_out"), "lease move-out pending")
        self.assertTrue(gid not in json.dumps(sched.get("keeper", {})), "candidate lease remains")


if __name__ == "__main__":
    unittest.main()
