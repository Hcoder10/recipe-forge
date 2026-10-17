"""Synthetic quick-bench checks; never launch a server or use a GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from forge.tools import bench, benchjob


class BenchNoiseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        prompts = self.root / "prompts.jsonl"
        prompts.write_text('{"prompt":"fixture"}\n')
        self.spec = {
            "work_dir": str(self.root / "work"), "prompts": str(prompts),
            "ref_cache_dir": str(self.root / "refs"), "recipe": {}, "workload": {},
            "candidate": {}, "confirm_pct": 2.0,
        }
        self.gpu = {"index": "0", "uuid": "GPU-fixture", "power_limit": "test"}

    def run_job(self, values, now=1000):
        path = self.root / "spec.json"
        path.write_text(json.dumps(self.spec))
        output = io.StringIO()
        measurements = [v if isinstance(v, Exception) else {"value": v} for v in values]
        with patch.object(benchjob.sys, "argv", ["benchjob", "--spec", str(path)]), \
                patch.object(benchjob, "gpu_identity", return_value=self.gpu), \
                patch.object(benchjob.time, "time", return_value=now), \
                patch.object(benchjob, "serve_and_measure", side_effect=measurements) as measure, \
                contextlib.redirect_stdout(output):
            benchjob.main()
        result = json.loads(output.getvalue().removeprefix("BENCH_RESULT "))
        return result, [call.args[4] for call in measure.call_args_list]

    def test_arithmetic_mean_and_fresh_baselines(self):
        for cached, candidate, fresh, baseline in ((100, 110, 101, 100.5), (100, 110, 105, 105),
                                                    (102, 110, 100, 101)):
            result = benchjob.comparison(cached, candidate, fresh)
            self.assertTrue(result["confirmed"])
            self.assertEqual(result["delta_pct"], round(100 * (candidate - baseline) / baseline, 3))
            self.assertEqual(result["delta_pct_vs_cached"], round(100 * (candidate - cached) / cached, 3))
            self.assertEqual(result["noise_pct"], round(100 * abs(cached - fresh) / fresh, 3))

    def test_verdict_and_zero_reference(self):
        for value, hint in ((100.5, "within_noise"), (101, "gain"), (99, "loss")):
            self.assertEqual(benchjob.comparison(100, value)["verdict_hint"], hint)
        self.assertEqual(benchjob.comparison(100, 104, 105)["verdict_hint"], "within_noise")
        self.assertIsNone(benchjob.comparison(0, 100)["delta_pct"])
        self.assertIsNone(benchjob.comparison(100, 100, 0)["delta_pct"])

    def test_cache_confirmation_order_and_refresh(self):
        first, labels = self.run_job([100, 100])
        self.assertEqual(labels, ["ref", "cand"])
        self.assertFalse(first["reference_cached"])
        result, labels = self.run_job([103, 101], now=1100)
        self.assertEqual(labels, ["cand", "ref-confirm"])
        self.assertTrue(result["reference_cached"])
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["reference_age_s"], 100)
        self.assertEqual(result["reference_gpu"], self.gpu)
        cache = self.root / "refs" / (result["ref_key"] + ".json")
        stored = json.loads(cache.read_text())
        self.assertEqual(stored["value"], 101)
        self.assertEqual(stored["measured_at"], 1100)
        result, labels = self.run_job([101], now=1101)
        self.assertEqual(labels, ["cand"])
        self.assertEqual(result["reference"]["value"], 101)
        self.assertNotIn("confirmed", result)

    def test_both_trigger_directions_and_threshold_override(self):
        self.run_job([100, 100])
        for candidate in (102, 98):
            result, labels = self.run_job([candidate, 100])
            self.assertEqual(labels, ["cand", "ref-confirm"])
            self.assertTrue(result["confirmed"])
        self.spec["confirm_pct"] = 4
        _, labels = self.run_job([103])
        self.assertEqual(labels, ["cand"])

    def test_expired_forced_and_missing_provenance_caches(self):
        first, _ = self.run_job([100, 100])
        cache = self.root / "refs" / (first["ref_key"] + ".json")
        stored = json.loads(cache.read_text())
        for changes in ({"measured_at": 0}, {"gpu": {}},
                        {"gpu": {"index": "9", "uuid": "GPU-other"}}):
            cache.write_text(json.dumps({**stored, **changes}))
            result, labels = self.run_job([100, 100], now=2000)
            self.assertEqual(labels, ["ref", "cand"])
            self.assertFalse(result["reference_cached"])
        self.spec["fresh_reference"] = True
        _, labels = self.run_job([100, 100])
        self.assertEqual(labels, ["ref", "cand"])

    def test_gpu_identity_matching(self):
        self.assertTrue(benchjob.same_gpu({"index": 0}, self.gpu))
        self.assertTrue(benchjob.same_gpu({"uuid": "GPU-fixture", "index": "9"}, self.gpu))
        self.assertFalse(benchjob.same_gpu({}, self.gpu))
        self.assertFalse(benchjob.same_gpu({"uuid": "unknown-0"}, {"uuid": "unknown-0"}))
        with patch.object(benchjob.subprocess, "run", return_value=SimpleNamespace(stdout="0, GPU-fixture, test, 200 W\n")):
            identity = benchjob.gpu_identity()
        self.assertEqual(identity["index"], "0")
        self.assertEqual(identity["uuid"], "GPU-fixture")
        self.assertEqual(identity["power_limit"], "200 W")

    def test_confirmation_failure_preserves_cache(self):
        first, _ = self.run_job([100, 100])
        cache = self.root / "refs" / (first["ref_key"] + ".json")
        before = cache.read_bytes()
        result, _ = self.run_job([103, RuntimeError("fixture failure")])
        self.assertFalse(result["ok"])
        self.assertTrue(result["infra"])
        self.assertEqual(cache.read_bytes(), before)

    def test_threshold_env_default_and_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(benchjob.confirmation_threshold(), 2.0)
        with patch.dict(os.environ, {"FORGE_BENCH_CONFIRM_PCT": "5"}):
            self.assertEqual(benchjob.confirmation_threshold(), 5)
            self.assertEqual(benchjob.confirmation_threshold(3), 3)
        for invalid in (-1, "nan", "inf"):
            with self.assertRaises(ValueError):
                benchjob.confirmation_threshold(invalid)

    def test_runner_passes_threshold(self):
        executor = Mock()
        executor.workspace.return_value = "/workspace"
        executor.run.return_value = SimpleNamespace(infra_error=None, stdout='BENCH_RESULT {"ok": true}')
        mission = SimpleNamespace(id="fixture", python="python", bench_prompts="prompts",
                                  recipe={"model": "fixture", "serve": {"args": []}, "workload": {}})
        runner = bench.RemoteBenchRunner(executor, "/forge", "python")
        with patch.dict(os.environ, {"FORGE_BENCH_CONFIRM_PCT": "5"}):
            runner.run(mission)
            self.assertEqual(json.loads(executor.write_text.call_args.args[1])["confirm_pct"], 5)
            runner.run(mission, confirm_pct=3)
            self.assertEqual(json.loads(executor.write_text.call_args.args[1])["confirm_pct"], 3)


if __name__ == "__main__":
    unittest.main()
