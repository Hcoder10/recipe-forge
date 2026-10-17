"""Persistence policy tests; no GPU, LLM, or measured findings."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from forge.agents import playbook
from forge.agents.decisions import Decisions
from forge.agents.detector import Detector
from forge.agents.mission import MissionRunner, Team


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        r = self.r = MissionRunner.__new__(MissionRunner)
        r.finished = threading.Event()
        r.paused = threading.Event()
        r.end_reason = ""
        r.log = Mock()
        r.mission = SimpleNamespace(worker_names=["worker-1"], minutes_left=Mock(return_value=30))
        r.meter = SimpleNamespace(budget_usd=100, cost_usd=0)
        r.agents = {name: Mock() for name in ("lead", "worker-1")}
        r.wakers = {name: threading.Event() for name in r.agents}
        r.decisions = Decisions(self.root / "decisions.jsonl", r.log)
        r.detector = Detector(r.decisions, Mock(), r.log)
        r.services = SimpleNamespace(team=Team(r))

    def request_stop(self):
        tid = self.r.decisions.trigger("worker_idle", "Assign work.", source="worker-1")
        self.r.decisions.decide("lead", "stop", [tid], "T1")
        return tid

    def test_stop_refused_and_trigger_restored(self):
        tid = self.request_stop()
        self.assertFalse(self.r._handle_lead_stop())
        self.assertFalse(self.r.finished.is_set())
        self.assertIsNone(self.r.decisions.stop_requested)
        self.assertEqual(self.r.decisions.pending()[0]["id"], tid)
        self.r.agents["lead"].add_event_message.assert_called_with(
            "Stop refused: 30 minutes remain. Pick the next ledger row or a new approach.")

    def test_stop_time_boundary(self):
        self.r.mission.minutes_left.return_value = 5
        self.request_stop()
        self.assertFalse(self.r._handle_lead_stop())
        self.r.mission.minutes_left.return_value = 4.99
        self.request_stop()
        self.assertTrue(self.r._handle_lead_stop())

    def test_stop_budget_boundary(self):
        self.r.meter.cost_usd = 94.99
        self.request_stop()
        self.assertFalse(self.r._handle_lead_stop())
        self.r.meter.cost_usd = 95
        self.request_stop()
        self.assertTrue(self.r._handle_lead_stop())

    def test_tool_stop_guard_budget_exception_and_refusal_event(self):
        m = self.r.mission
        m.time_box_min, m.criteria, m.id, m.benches = 30, {}, "fixture", {}
        self.r.services.gate = Mock()
        self.r.services.team.assign("lead", "worker-1", "Inspect the row.", "runtime")
        with patch("forge.agents.mission.done_mod.audit", return_value={"met": False}):
            self.assertIn("worker-1", self.r.services.team.stop_blockers())
        self.r.agents["lead"].add_event_message.assert_called_with(
            "Stop refused: 30 minutes remain. Pick the next ledger row or a new approach.")
        self.r.meter.cost_usd = 95
        self.assertEqual(self.r.services.team.stop_blockers(), [])

    def test_unlimited_budget_does_not_allow_early_stop(self):
        self.r.meter.budget_usd = None
        self.request_stop()
        self.assertFalse(self.r._handle_lead_stop())

    def test_operator_stop_is_preserved(self):
        self.request_stop()
        self.r.end("operator stop: requested")
        self.assertTrue(self.r._handle_lead_stop())
        self.assertEqual(self.r.end_reason, "operator stop: requested")

    def test_failure_trigger_and_assignment_closure(self):
        for status in ("failed", "refuted"):
            with self.subTest(status=status):
                team = self.r.services.team
                team.assign("lead", "worker-1", "Inspect the assigned row.", "runtime")
                team.report("worker-1", "Consult evidence identifiers.", status)
                self.r._supervise_workers()
                self.r._supervise_workers()
                pending = [t for t in self.r.decisions.pending() if t["kind"] == "worker_idle_after_failure"]
                self.assertEqual(len(pending), 1)
                self.assertTrue(self.r.wakers["lead"].is_set())
                team.assign("lead", "worker-1", "Test the next assumption.", "runtime")
                self.r._supervise_workers()
                self.assertFalse(any(t["kind"] == "worker_idle_after_failure" for t in self.r.decisions.pending()))

    def test_failure_can_be_closed_with_decide(self):
        self.r.services.team.report("worker-1", "Consult evidence identifiers.", "failed")
        self.r._supervise_workers()
        ids = [t["id"] for t in self.r.decisions.pending()]
        self.r.decisions.decide("lead", "replan", ids, "", "Choose the next row.")
        self.r._supervise_workers()
        self.assertEqual(self.r.decisions.pending(), [])

    def test_idle_deadline_and_no_duplicate(self):
        with patch("forge.agents.mission.time.monotonic", return_value=100):
            self.r._supervise_workers()
        with patch("forge.agents.mission.time.monotonic", return_value=219):
            self.r._supervise_workers()
        self.assertEqual(self.r.decisions.pending(), [])
        with patch("forge.agents.mission.time.monotonic", return_value=220):
            self.r._supervise_workers()
        self.assertEqual(self.r.decisions.pending()[0]["kind"], "worker_idle")
        with patch("forge.agents.mission.time.monotonic", return_value=400):
            self.r._supervise_workers()
        self.assertEqual(len(self.r.decisions.pending()), 1)

    def test_active_and_paused_workers_not_flagged_idle(self):
        self.r.services.team.assign("lead", "worker-1", "Inspect.", "runtime")
        self.r._supervise_workers()
        self.assertEqual(self.r.decisions.pending(), [])
        self.r.paused.set()
        self.r.services.team.report("worker-1", "Consult evidence identifiers.", "failed")
        self.r._supervise_workers()
        self.assertFalse(any(t["kind"] == "worker_idle_after_failure" for t in self.r.decisions.pending()))

    def test_playbook_roundtrip_hash_and_procedural_text(self):
        pb = playbook.load(None, self.root)
        digest = playbook.save(pb, self.root)
        self.assertEqual(digest, playbook.content_hash(pb))
        self.assertEqual(playbook.load(digest, self.root), pb)
        custom = {**pb, "protocol": pb["protocol"] + " Review evidence identifiers."}
        custom_hash = playbook.save(custom, self.root)
        self.assertNotEqual(custom_hash, digest)
        self.assertEqual(playbook.load(custom_hash, self.root)["protocol"], custom["protocol"])
        for field in ("lead", "worker", "verifier", "protocol", "method", "techniques"):
            self.assertNotRegex(pb[field], r"\d")
        for section in ("Persistence", "Approaches beyond flags", "Think like a performance engineer"):
            self.assertIn(section, pb["techniques"])
        self.assertIn("If available, gpu_job and gpu_job_status", pb["techniques"])

    def test_repeated_error_nudge_uses_no_raw_error(self):
        d = self.r.detector
        for index in range(3):
            d.observe("worker-1", "worker", "read", {"path": str(index)},
                      {"ok": False, "class": "agent_error", "error": "opaque fixture"})
        text = " ".join(d.take_nudges("worker-1"))
        self.assertIn("Change approach: the same error repeated.", text)
        self.assertNotIn("opaque fixture", text)

    def test_different_errors_do_not_count_as_same_error(self):
        d = self.r.detector
        for index in range(3):
            d.observe("worker-1", "worker", "read", {},
                      {"ok": False, "class": "agent_error", "error": str(index)})
        self.assertEqual(d.counts["repeat_failure"], 0)

    def test_measurement_deadline_and_reset(self):
        d = self.r.detector
        start = d.started_at
        d.check_measurement("worker-1", now=start + 899)
        self.assertEqual(d.counts["no_measurement"], 0)
        d.check_measurement("worker-1", now=start + 900)
        self.assertEqual(d.counts["no_measurement"], 1)
        for tool in ("bench", "kernel_test"):
            with patch("forge.agents.detector.time.monotonic", return_value=start + 1000):
                d.observe("worker-1", "worker", tool, {}, {"ok": True})
            d.check_measurement("worker-1", now=start + 1899)
            self.assertEqual(d.counts["no_measurement"], 1)


if __name__ == "__main__":
    unittest.main()
