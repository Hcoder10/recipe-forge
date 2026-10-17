"""Harness regression checks with synthetic evidence; no GPU, server or LLM calls."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from forge.agents.mission import MissionRunner
from forge.agents.mock import MockBrain, _call, _msg
from forge.tools.bench import MockBenchRunner
from forge.tools.builtin import t_bench, t_decide, t_make_bundle, t_request_gate
from forge.tools.executor import LocalExecutor
from forge.tools.gateclient import MockGate
from forge.tools.registry import EnvBlocked, ToolError
from test_mission_mock import make_spec


class HarnessProgressTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.bench = MockBenchRunner(noise=0)
        self.r = MissionRunner(make_spec(root, time_box_min=75),
                               executor=LocalExecutor(root / "target"), bench=self.bench,
                               gate=MockGate(root / "gate.jsonl"), mock_llm=MockBrain(), root=root / "state")
        self.ctx = self.r.agents["worker-1"].ctx
        self.other = self.r.agents["worker-2"].ctx

    def test_freeze_and_submit_without_second_benchmark(self):
        result = t_bench(self.ctx, config_delta={"fixture_option": "fixture"})
        bid = result["bundle"]
        self.assertEqual(self.bench.calls[0]["bundle_dir"], self.r.mission.bundle_dir(bid))
        self.assertIsNone(self.bench.calls[0]["config_delta"])
        request = t_request_gate(self.ctx, bid, "synthetic evidence", result["bench_id"])
        self.assertEqual(request["status"], "pending_review")
        self.assertEqual(len(self.bench.calls), 1)

    def test_duplicate_across_workers_and_explicit_repeats(self):
        first = t_bench(self.ctx, config_delta={"fixture_option": "fixture"})
        second = t_bench(self.other, config_delta={"--fixture-option": "fixture"})
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["bench_id"], second["bench_id"])
        self.assertEqual(len(self.bench.calls), 1)
        t_bench(self.other, bundle=first["bundle"], repeat_reason="independent confirmation")
        t_bench(self.ctx, bundle=first["bundle"], fresh_reference=True)
        self.assertEqual(len(self.bench.calls), 3)

    def test_prompt_count_and_expiry_invalidate_dedupe(self):
        t_bench(self.ctx, n_prompts=2)
        t_bench(self.ctx, n_prompts=3)
        for entry in self.r.mission.experiments.values():
            entry["t"] -= 601
        t_bench(self.ctx, n_prompts=2)
        self.assertEqual(len(self.bench.calls), 3)

    def test_inflight_duplicate_and_failed_job_are_not_relaunched(self):
        entered, release = threading.Event(), threading.Event()
        original = self.bench.run
        def slow(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return original(*args, **kwargs)
        with patch.object(self.bench, "run", side_effect=slow):
            thread = threading.Thread(target=lambda: t_bench(self.ctx))
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                result = t_bench(self.other)
                self.assertEqual(result["status"], "running")
                self.assertTrue(result["duplicate"])
            finally:
                release.set()
                thread.join(3)
        with patch.object(self.bench, "run", side_effect=EnvBlocked("fixture unavailable")):
            with self.assertRaises(EnvBlocked):
                t_bench(self.ctx, n_prompts=7)
        t_bench(self.ctx, n_prompts=7)
        self.assertEqual(len(self.bench.calls), 2)

    def test_bundle_overrides_are_rejected(self):
        bundle = t_make_bundle(self.ctx, config_delta={"fixture": "value"})["bundle"]
        with self.assertRaises(ToolError):
            t_bench(self.ctx, bundle=bundle, env={"FIXTURE": "value"})
        self.assertFalse(self.bench.calls)

    def test_config_freeze_excludes_unmeasured_edits(self):
        m = self.r.mission
        self.ctx.executor.write_text(m.overlay_root(self.ctx.agent) + "/vllm/x.py", "A = 2\n")
        result = t_bench(self.ctx, config_delta={"fixture": "value"})
        self.assertEqual(m.bundles[result["bundle"]]["manifest"]["overlays"], {})

    def test_stop_guard_keeps_triggers_open_and_operator_stop_works(self):
        self.r.services.team.assign("lead", "worker-1", "inspect and measure", "runtime")
        tid = self.r.decisions.trigger("fixture", "unfinished")
        with self.assertRaises(ToolError):
            t_decide(self.r.agents["lead"].ctx, "stop", [tid], "fixture")
        self.assertIsNone(self.r.decisions.stop_requested)
        self.assertEqual(self.r.decisions.pending()[0]["id"], tid)
        self.r.end("operator stop: fixture")
        self.assertTrue(self.r.finished.is_set())

    def test_stop_after_blocked_report_or_final_quarter(self):
        team = self.r.services.team
        team.assign("lead", "worker-1", "inspect and measure", "runtime")
        team.report("worker-1", "fixture unavailable", "blocked")
        self.assertEqual(team.stop_blockers(), [])
        team.assign("lead", "worker-2", "inspect and measure", "kernels")
        self.r.mission.t_start = time.time() - 60 * 60
        self.assertEqual(team.stop_blockers(), [])

    def test_unmeasured_done_lane_blocks_early_stop(self):
        team = self.r.services.team
        team.assign("lead", "worker-1", "inspect and measure", "runtime")
        team.report("worker-1", "inspection complete", "done")
        self.assertIn("runtime", team.stop_blockers())

    def test_infra_failure_stops_batch_and_parks_worker(self):
        worker = self.r.agents["worker-1"]
        self.r.services.team.assign("lead", worker.name, "fixture", "config")
        self.r.client.mock = lambda *args, **kw: _msg(_call("bench"), _call("bench"))
        with patch.object(self.bench, "run", side_effect=EnvBlocked("fixture unavailable")) as run:
            result = worker.step("fixture")
        self.assertTrue(result["infra_blocked"])
        self.assertEqual(run.call_count, 1)
        self.assertGreater(worker.infra_retry_at, time.time())
        self.assertEqual(sum(m["role"] == "tool" for m in worker.history), 2)
        worker.infra_retry_at = 0
        with patch.object(worker, "step", side_effect=lambda status: (self.r.end("fixture") or result)):
            self.r._worker_loop(worker.name)
        self.assertIsNone(self.r.services.team.task_of(worker.name))


if __name__ == "__main__":
    unittest.main()
